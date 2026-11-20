using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

public class Chunk : MonoBehaviour
{
    public ComputeShader MarchingShader;

    public MeshFilter MeshFilter;
    public MeshCollider MeshCollider;

    public NoiseGenerator NoiseGenerator;

    [Range(0, 9)]
    public int LOD;

    public float ChunkScale = GridMetrics.Scale;
    public bool UseSmoothShading = false;

    [Header("Performance & Mesh Telemetry")]
    [SerializeField] int rawVertexCount;
    [SerializeField] int weldedVertexCount;
    [SerializeField] float vertexReductionPercent;
    [SerializeField] int triangleCount;
    [SerializeField] float lastMeshRebuildMs;
    [SerializeField] bool usesLruWeightCache;

    public int RawVertexCount => rawVertexCount;
    public int WeldedVertexCount => weldedVertexCount;
    public float VertexReductionPercent => vertexReductionPercent;
    public int TriangleCount => triangleCount;
    public float LastMeshRebuildMs => lastMeshRebuildMs;
    public bool UsesLruWeightCache => usesLruWeightCache;
    public bool IsColliderDirty => _colliderDirty;
    public bool IsQueuedForCollider { get; set; }
    public bool IsAsyncBuildInFlight => _activeAsyncSlot != null;
    public EmptyChunkState LastEmptyState { get; private set; }
    public Bounds LocalMeshBounds => _mesh != null ? _mesh.bounds : default;
    public bool IsMeshVisible => gameObject.activeInHierarchy && _meshRenderer != null && _meshRenderer.enabled && triangleCount > 0 && _baseSurfaceIndexCount >= 3;
    public Bounds WorldMeshBounds
    {
        get
        {
            Bounds b = LocalMeshBounds;
            b.center += transform.position;
            return b;
        }
    }
    public bool IsAsyncStage2Ready => _activeAsyncSlot != null && !_activeAsyncSlot.Cancelled && _activeAsyncSlot.Stage == 2 && _activeAsyncSlot.TriangleRequest.done;

    public enum EmptyChunkState : byte
    {
        None = 0,
        AllAir = 1,
        AllSolid = 2
    }

    public enum AsyncBuildPollResult
    {
        StillPending,
        CompletedWithMesh,
        CompletedEmpty,
        CancelledOrFailed
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Triangle
    {
        public Vector3 a;
        public Vector3 b;
        public Vector3 c;

        public static int SizeOf => sizeof(float) * 3 * 3;
    }

    class AsyncReadbackSlot
    {
        public ComputeBuffer TrianglesBuffer;
        public ComputeBuffer CountBuffer;
        public int TriangleCapacity;
        public bool InUse;
        public bool Cancelled;
        public int Stage; // 0 = Idle, 1 = WaitingForCount, 2 = WaitingForTriangles
        public int RequestToken;
        public int TriCount;
        public Chunk OwnerChunk;
        public AsyncGPUReadbackRequest CountRequest;
        public AsyncGPUReadbackRequest TriangleRequest;
    }

    const float NormalizedQuantizeResolution = 10000f;
    const float BoundaryQuantizeScale = 20f;
    const int MaxLruWeightBuffers = 8;
    const int AsyncSlotCount = 16;

    // Shared scratch buffers for passive chunks + bounded 8-slot LRU cache for actively sculpted chunks
    static ComputeBuffer s_trianglesBuffer;
    static ComputeBuffer s_trianglesCountBuffer;
    static ComputeBuffer s_scratchWeightsBuffer;
    static ComputeBuffer s_strokesBuffer;
    static int s_allocatedTriangleCapacity = -1;
    static int s_allocatedWeightCount = -1;
    static int s_allocatedStrokeCapacity = -1;
    static int s_allocatedLruWeightCount = -1;
    static int s_activeInstanceCount;

    static readonly AsyncReadbackSlot[] s_asyncSlots = CreateAsyncSlots();

    static AsyncReadbackSlot[] CreateAsyncSlots()
    {
        AsyncReadbackSlot[] slots = new AsyncReadbackSlot[AsyncSlotCount];
        for (int i = 0; i < AsyncSlotCount; i++)
        {
            slots[i] = new AsyncReadbackSlot();
        }
        return slots;
    }

    public static int MaxAsyncReadbackSlots => AsyncSlotCount;

    public static int ActiveAsyncReadbackCount
    {
        get
        {
            int count = 0;
            for (int i = 0; i < s_asyncSlots.Length; i++)
            {
                if (s_asyncSlots[i].InUse)
                {
                    count++;
                }
            }
            return count;
        }
    }

    static readonly Stack<ComputeBuffer> s_freeLruWeightBuffers = new Stack<ComputeBuffer>(MaxLruWeightBuffers);
    static readonly List<Chunk> s_lruChunks = new List<Chunk>(MaxLruWeightBuffers);
    static int s_totalCreatedLruBuffers;

    static readonly Dictionary<Vector3Int, int> s_vertexIndexMap = new Dictionary<Vector3Int, int>(16384);
    static readonly Dictionary<int, int> s_skirtVertexMap = new Dictionary<int, int>(512);
    static readonly List<Vector3> s_weldedVertices = new List<Vector3>(16384);
    static readonly List<Vector3> s_accumulatedNormals = new List<Vector3>(16384);
    static readonly List<int> s_weldedTriangles = new List<int>(49152);
    static readonly int[] s_triCountCPU = new int[2];
    static readonly Stopwatch s_rebuildTimer = new Stopwatch();
    static Triangle[] s_triangleBufferCPU = new Triangle[4096];
    static PlanetOctree.TerraformStroke[] s_strokeArrayCPU = new PlanetOctree.TerraformStroke[64];

    readonly List<int> _boundaryVertexIndices = new List<int>(512);
    readonly List<int> _boundarySkirtVertexIndices = new List<int>(512);
    readonly List<Vector3Int> _boundaryWorldKeys = new List<Vector3Int>(512);
    readonly List<Vector3> _localBoundaryNormals = new List<Vector3>(512);
    readonly List<Vector3> _meshNormals = new List<Vector3>(8192);
    readonly List<Vector3> _meshVertices = new List<Vector3>(2048);
    readonly List<int> _meshTriangles = new List<int>(4096);
    readonly List<PlanetOctree.TerraformStroke> _localStrokes = new List<PlanetOctree.TerraformStroke>();

    int _baseSurfaceIndexCount;
    MeshRenderer _meshRenderer;
    ComputeBuffer _cachedWeightsBuffer;
    int _cachedWeightsLod = -1;
    bool _hasContributedBoundaryNormals;
    AsyncReadbackSlot _activeAsyncSlot;
    int _asyncBuildToken;

    Mesh _mesh;
    int _marchKernel = -1;
    int _updateWeightsKernel = -1;
    int _updateWeightsBatchKernel = -1;
    bool _isInitialized;
    bool _colliderDirty;
    bool _pendingValidateRebuild;

    private void Awake()
    {
        s_activeInstanceCount++;
        EnsureComponentsAndMesh();
    }

    private void Start()
    {
        if (!_isInitialized)
        {
            InitializeAndGenerate();
        }
    }

    private void Update()
    {
        if (_pendingValidateRebuild && _isInitialized && MarchingShader != null && NoiseGenerator != null)
        {
            _pendingValidateRebuild = false;
            CancelAsyncBuild();
            ReleaseCachedWeightBuffer();
            RebuildWithStrokes(_localStrokes, true);
        }
    }

    private void OnValidate()
    {
        if (Application.isPlaying && _isInitialized)
        {
            _pendingValidateRebuild = true;
        }
    }

    private void OnDestroy()
    {
        CancelAsyncBuild();
        ReleaseCachedWeightBuffer();
        s_activeInstanceCount = Mathf.Max(0, s_activeInstanceCount - 1);
        if (s_activeInstanceCount == 0)
        {
            ReleaseSharedBuffers();
        }

        if (_mesh != null)
        {
            Destroy(_mesh);
            _mesh = null;
        }
    }

    public void InitializeAndGenerate()
    {
        EnsureComponentsAndMesh();
        RebuildWithStrokes(_localStrokes, true);
        _isInitialized = true;
    }

    public void InitializeOctreeChunk(
        ComputeShader marchingShader,
        NoiseGenerator noiseGenerator,
        Material material,
        float chunkScale,
        int lod,
        bool buildMesh = true)
    {
        MarchingShader = marchingShader;
        NoiseGenerator = noiseGenerator;
        ChunkScale = chunkScale;
        LOD = Mathf.Clamp(lod, 0, GridMetrics.LastLod);

        ReleaseCachedWeightBuffer();
        EnsureComponentsAndMesh();
        if (material != null)
        {
            MeshRenderer renderer = GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = material;
            }
        }

        _marchKernel = -1;
        _updateWeightsKernel = -1;
        _updateWeightsBatchKernel = -1;

        if (buildMesh)
        {
            RebuildWithStrokes(null, true);
        }
        _isInitialized = true;
    }

    public void RebuildWithStrokes(List<PlanetOctree.TerraformStroke> strokes, bool updateCollider = true)
    {
        if (MarchingShader == null || NoiseGenerator == null)
        {
            return;
        }

        CancelAsyncBuild();
        ReleaseCachedWeightBuffer();
        EnsureComponentsAndMesh();
        EnsureSharedBuffers();
        NoiseGenerator.PopulateNoiseBuffer(s_scratchWeightsBuffer, LOD, transform.position, ChunkScale);

        if (strokes != null && strokes.Count > 0)
        {
            ApplyTerraformBatchKernelOnly(strokes, s_scratchWeightsBuffer);
        }

        ConstructMesh(s_scratchWeightsBuffer);
        AssignMeshAndCollider(updateCollider);
    }

    public static bool HasFreeAsyncSlot()
    {
        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            return false;
        }

        for (int i = 0; i < s_asyncSlots.Length; i++)
        {
            if (!s_asyncSlots[i].InUse)
            {
                return true;
            }
        }
        return false;
    }

    static AsyncReadbackSlot AcquireFreeAsyncSlot()
    {
        for (int i = 0; i < s_asyncSlots.Length; i++)
        {
            if (!s_asyncSlots[i].InUse)
            {
                return s_asyncSlots[i];
            }
        }
        return null;
    }

    public static void ReclaimCancelledAsyncSlots()
    {
        for (int i = 0; i < s_asyncSlots.Length; i++)
        {
            AsyncReadbackSlot slot = s_asyncSlots[i];
            if (!slot.InUse || !slot.Cancelled)
            {
                continue;
            }

            bool isDone = slot.Stage == 1
                ? slot.CountRequest.done
                : (slot.Stage == 2 ? slot.TriangleRequest.done : true);

            if (isDone)
            {
                ResetAsyncSlot(slot);
            }
        }
    }

    public static void WaitForAllAsyncReadbacks()
    {
        AsyncGPUReadback.WaitAllRequests();
        for (int i = 0; i < s_asyncSlots.Length; i++)
        {
            AsyncReadbackSlot slot = s_asyncSlots[i];
            if (slot.OwnerChunk != null)
            {
                slot.OwnerChunk._activeAsyncSlot = null;
                slot.OwnerChunk._asyncBuildToken++;
            }
            ResetAsyncSlot(slot);
        }
    }

    static void ResetAsyncSlot(AsyncReadbackSlot slot)
    {
        slot.InUse = false;
        slot.Cancelled = false;
        slot.Stage = 0;
        slot.OwnerChunk = null;
        slot.TriCount = 0;
    }

    static void EnsureAsyncSlotCapacity(AsyncReadbackSlot slot, int maxTriangles)
    {
        if (slot.TrianglesBuffer == null || slot.TriangleCapacity < maxTriangles)
        {
            slot.TrianglesBuffer?.Release();
            slot.TrianglesBuffer = new ComputeBuffer(maxTriangles, Triangle.SizeOf, ComputeBufferType.Append);
            slot.TriangleCapacity = maxTriangles;
        }

        if (slot.CountBuffer == null)
        {
            slot.CountBuffer = new ComputeBuffer(2, sizeof(int), ComputeBufferType.Raw);
        }
    }

    public void CancelAsyncBuild()
    {
        _asyncBuildToken++;
        if (_activeAsyncSlot != null)
        {
            _activeAsyncSlot.Cancelled = true;
            _activeAsyncSlot.OwnerChunk = null;
            _activeAsyncSlot = null;
        }
    }

    public bool TryBeginAsyncRebuildWithStrokes(List<PlanetOctree.TerraformStroke> strokes)
    {
        if (!SystemInfo.supportsAsyncGPUReadback || MarchingShader == null || NoiseGenerator == null)
        {
            return false;
        }

        AsyncReadbackSlot slot = AcquireFreeAsyncSlot();
        if (slot == null)
        {
            return false;
        }

        CancelAsyncBuild();
        ReleaseCachedWeightBuffer();
        EnsureComponentsAndMesh();
        EnsureSharedBuffers();
        CacheKernels();

        int lodPoints = GridMetrics.PointsPerChunk(LOD);
        int cellsPerAxis = Mathf.Max(1, lodPoints - 1);
        int maxTriangles = 5 * cellsPerAxis * cellsPerAxis * cellsPerAxis;
        EnsureAsyncSlotCapacity(slot, maxTriangles);

        NoiseGenerator.PopulateNoiseBuffer(s_scratchWeightsBuffer, LOD, transform.position, ChunkScale);
        if (strokes != null && strokes.Count > 0)
        {
            ApplyTerraformBatchKernelOnly(strokes, s_scratchWeightsBuffer);
        }

        float isoLevel = NoiseGenerator != null ? NoiseGenerator.IsoLevel : 0.5f;
        MarchingShader.SetBuffer(_marchKernel, "_Triangles", slot.TrianglesBuffer);
        MarchingShader.SetBuffer(_marchKernel, "_ChunkMeta", slot.CountBuffer);
        MarchingShader.SetBuffer(_marchKernel, "_Weights", s_scratchWeightsBuffer);
        MarchingShader.SetInt("_ChunkSize", lodPoints);
        MarchingShader.SetInt("_LODSize", lodPoints);
        MarchingShader.SetFloat("_IsoLevel", isoLevel);
        MarchingShader.SetFloat("_Scale", ChunkScale);
        MarchingShader.SetVector("_ChunkOffset", transform.position);

        slot.TrianglesBuffer.SetCounterValue(0);
        int groups = GridMetrics.ThreadGroups(LOD);
        MarchingShader.Dispatch(_marchKernel, groups, groups, groups);

        ComputeBuffer.CopyCount(slot.TrianglesBuffer, slot.CountBuffer, 0);
        slot.CountRequest = AsyncGPUReadback.Request(slot.CountBuffer);

        _asyncBuildToken++;
        slot.InUse = true;
        slot.Cancelled = false;
        slot.Stage = 1;
        slot.OwnerChunk = this;
        slot.RequestToken = _asyncBuildToken;
        slot.TriCount = 0;
        _activeAsyncSlot = slot;
        return true;
    }

    public AsyncBuildPollResult AdvanceAsyncStage1()
    {
        AsyncReadbackSlot slot = _activeAsyncSlot;
        if (slot == null)
        {
            return AsyncBuildPollResult.CancelledOrFailed;
        }

        if (slot.Cancelled || slot.RequestToken != _asyncBuildToken || slot.OwnerChunk != this)
        {
            _activeAsyncSlot = null;
            return AsyncBuildPollResult.CancelledOrFailed;
        }

        if (slot.Stage != 1 || !slot.CountRequest.done)
        {
            return AsyncBuildPollResult.StillPending;
        }

        if (slot.CountRequest.hasError)
        {
            ResetAsyncSlot(slot);
            _activeAsyncSlot = null;
            return AsyncBuildPollResult.CancelledOrFailed;
        }

        NativeArray<int> meta = slot.CountRequest.GetData<int>();
        int triCount = Mathf.Min(meta[0], slot.TriangleCapacity);
        triangleCount = triCount;

        if (triCount <= 0)
        {
            LastEmptyState = (meta.Length > 1 && meta[1] == 1) ? EmptyChunkState.AllAir : EmptyChunkState.AllSolid;
            BuildEmptyMesh();
            ResetAsyncSlot(slot);
            _activeAsyncSlot = null;
            return AsyncBuildPollResult.CompletedEmpty;
        }

        LastEmptyState = EmptyChunkState.None;
        slot.TriCount = triCount;
        slot.TriangleRequest = AsyncGPUReadback.Request(slot.TrianglesBuffer, triCount * Triangle.SizeOf, 0);
        slot.Stage = 2;
        return AsyncBuildPollResult.StillPending;
    }

    public AsyncBuildPollResult TryFinalizeAsyncStage2(bool updateCollider)
    {
        AsyncReadbackSlot slot = _activeAsyncSlot;
        if (slot == null)
        {
            return AsyncBuildPollResult.CancelledOrFailed;
        }

        if (slot.Cancelled || slot.RequestToken != _asyncBuildToken || slot.OwnerChunk != this)
        {
            _activeAsyncSlot = null;
            return AsyncBuildPollResult.CancelledOrFailed;
        }

        if (slot.Stage != 2 || !slot.TriangleRequest.done)
        {
            return AsyncBuildPollResult.StillPending;
        }

        if (slot.TriangleRequest.hasError)
        {
            ResetAsyncSlot(slot);
            _activeAsyncSlot = null;
            return AsyncBuildPollResult.CancelledOrFailed;
        }

        NativeArray<Triangle> tris = slot.TriangleRequest.GetData<Triangle>();
        BuildMeshFromNativeTriangles(tris, slot.TriCount);
        AssignMeshAndCollider(updateCollider);
        ResetAsyncSlot(slot);
        _activeAsyncSlot = null;
        return AsyncBuildPollResult.CompletedWithMesh;
    }

    /// <summary>
    /// Applies a single sculpting stroke in O(1) GPU time using an 8-slot LRU cache of VRAM weight buffers,
    /// avoiding noise recomputation and full stroke replay on consecutive sculpting frames.
    /// </summary>
    public void ApplyIncrementalStroke(
        PlanetOctree.TerraformStroke newStroke,
        List<PlanetOctree.TerraformStroke> allIntersectingStrokes,
        bool updateCollider = false)
    {
        if (MarchingShader == null || NoiseGenerator == null)
        {
            return;
        }

        CancelAsyncBuild();
        EnsureComponentsAndMesh();
        EnsureSharedBuffers();
        CacheKernels();

        int pointsForLod = GridMetrics.PointsPerChunk(LOD);
        int requiredWeights = pointsForLod * pointsForLod * pointsForLod;
        bool cacheHit = _cachedWeightsBuffer != null && _cachedWeightsLod == LOD && _cachedWeightsBuffer.count >= requiredWeights;
        if (cacheHit)
        {
            TouchLruEntry();
            DispatchSingleStrokeKernel(_cachedWeightsBuffer, newStroke);
        }
        else
        {
            AcquireLruWeightBuffer();
            NoiseGenerator.PopulateNoiseBuffer(_cachedWeightsBuffer, LOD, transform.position, ChunkScale);
            if (allIntersectingStrokes != null && allIntersectingStrokes.Count > 0)
            {
                ApplyTerraformBatchKernelOnly(allIntersectingStrokes, _cachedWeightsBuffer);
            }
            else
            {
                DispatchSingleStrokeKernel(_cachedWeightsBuffer, newStroke);
            }
            _cachedWeightsLod = LOD;
        }

        ConstructMesh(_cachedWeightsBuffer);
        AssignMeshAndCollider(updateCollider);
    }

    void AssignMeshAndCollider(bool updateCollider)
    {
        if (MeshFilter != null)
        {
            MeshFilter.sharedMesh = _mesh;
        }

        if (updateCollider)
        {
            _colliderDirty = true;
            FlushCollider();
        }
        else
        {
            _colliderDirty = true;
        }
    }

    public void ClearChunkData()
    {
        CancelAsyncBuild();
        ReleaseCachedWeightBuffer();

        if (_mesh != null)
        {
            _mesh.Clear();
        }
        if (MeshFilter != null)
        {
            MeshFilter.sharedMesh = null;
        }
        if (MeshCollider != null)
        {
            MeshCollider.sharedMesh = null;
        }

        _boundaryVertexIndices.Clear();
        _boundarySkirtVertexIndices.Clear();
        _boundaryWorldKeys.Clear();
        _localBoundaryNormals.Clear();
        _meshNormals.Clear();
        _meshVertices.Clear();
        _meshTriangles.Clear();
        _baseSurfaceIndexCount = 0;
        _hasContributedBoundaryNormals = false;
        _colliderDirty = false;
        IsQueuedForCollider = false;
        LastEmptyState = EmptyChunkState.None;
        rawVertexCount = 0;
        weldedVertexCount = 0;
        vertexReductionPercent = 0f;
        triangleCount = 0;
    }

    void AcquireLruWeightBuffer()
    {
        int pointsForLod = GridMetrics.PointsPerChunk(LOD);
        int weightCount = pointsForLod * pointsForLod * pointsForLod;

        if (_cachedWeightsBuffer != null)
        {
            if (_cachedWeightsBuffer.count >= weightCount)
            {
                TouchLruEntry();
                return;
            }
            ReleaseCachedWeightBuffer();
        }

        if (s_allocatedLruWeightCount < weightCount)
        {
            while (s_freeLruWeightBuffers.Count > 0)
            {
                s_freeLruWeightBuffers.Pop()?.Release();
                s_totalCreatedLruBuffers = Mathf.Max(0, s_totalCreatedLruBuffers - 1);
            }
            s_allocatedLruWeightCount = weightCount;
        }

        int allocSize = Mathf.Max(weightCount, s_allocatedLruWeightCount);

        if (s_freeLruWeightBuffers.Count > 0)
        {
            _cachedWeightsBuffer = s_freeLruWeightBuffers.Pop();
        }
        else if (s_totalCreatedLruBuffers < MaxLruWeightBuffers)
        {
            _cachedWeightsBuffer = new ComputeBuffer(allocSize, sizeof(float));
            s_totalCreatedLruBuffers++;
        }
        else if (s_lruChunks.Count > 0)
        {
            Chunk victim = s_lruChunks[0];
            s_lruChunks.RemoveAt(0);
            _cachedWeightsBuffer = victim._cachedWeightsBuffer;
            victim._cachedWeightsBuffer = null;
            victim._cachedWeightsLod = -1;
            victim.usesLruWeightCache = false;

            if (_cachedWeightsBuffer != null && _cachedWeightsBuffer.count < allocSize)
            {
                _cachedWeightsBuffer.Release();
                _cachedWeightsBuffer = new ComputeBuffer(allocSize, sizeof(float));
            }
        }
        else
        {
            _cachedWeightsBuffer = new ComputeBuffer(allocSize, sizeof(float));
        }

        s_lruChunks.Add(this);
        usesLruWeightCache = true;
    }

    void TouchLruEntry()
    {
        int idx = s_lruChunks.IndexOf(this);
        if (idx >= 0 && idx < s_lruChunks.Count - 1)
        {
            s_lruChunks.RemoveAt(idx);
            s_lruChunks.Add(this);
        }
        usesLruWeightCache = true;
    }

    void ReleaseCachedWeightBuffer()
    {
        if (_cachedWeightsBuffer != null)
        {
            s_lruChunks.Remove(this);
            if (_cachedWeightsBuffer.count >= s_allocatedLruWeightCount)
            {
                s_freeLruWeightBuffers.Push(_cachedWeightsBuffer);
            }
            else
            {
                _cachedWeightsBuffer.Release();
                s_totalCreatedLruBuffers = Mathf.Max(0, s_totalCreatedLruBuffers - 1);
            }
            _cachedWeightsBuffer = null;
        }
        _cachedWeightsLod = -1;
        usesLruWeightCache = false;
    }

    void EnsureComponentsAndMesh()
    {
        if (MeshFilter == null)
        {
            MeshFilter = GetComponent<MeshFilter>();
        }
        if (_meshRenderer == null)
        {
            _meshRenderer = GetComponent<MeshRenderer>();
        }
        if (MeshCollider == null)
        {
            MeshCollider = GetComponent<MeshCollider>();
        }

        if (MeshCollider != null)
        {
            MeshCollider.cookingOptions = MeshColliderCookingOptions.CookForFasterSimulation | MeshColliderCookingOptions.UseFastMidphase;
        }

        if (_mesh == null)
        {
            _mesh = new Mesh
            {
                name = "ProceduralChunkMesh",
                indexFormat = UnityEngine.Rendering.IndexFormat.UInt32
            };
            _mesh.MarkDynamic();
        }
    }

    void CacheKernels()
    {
        if (MarchingShader == null)
        {
            return;
        }
        if (_marchKernel < 0)
        {
            _marchKernel = MarchingShader.FindKernel("March");
        }
        if (_updateWeightsKernel < 0)
        {
            _updateWeightsKernel = MarchingShader.FindKernel("UpdateWeights");
        }
        if (_updateWeightsBatchKernel < 0)
        {
            _updateWeightsBatchKernel = MarchingShader.FindKernel("UpdateWeightsBatch");
        }
    }

    void EnsureSharedBuffers()
    {
        int pointsForLod = GridMetrics.PointsPerChunk(LOD);
        int requiredWeights = pointsForLod * pointsForLod * pointsForLod;

        if (s_scratchWeightsBuffer == null || s_allocatedWeightCount < requiredWeights)
        {
            s_scratchWeightsBuffer?.Release();
            int allocWeights = Mathf.Max(requiredWeights, s_allocatedWeightCount);
            s_scratchWeightsBuffer = new ComputeBuffer(allocWeights, sizeof(float));
            s_allocatedWeightCount = allocWeights;
        }

        int cellsPerAxis = Mathf.Max(1, pointsForLod - 1);
        int maxTriangles = 5 * cellsPerAxis * cellsPerAxis * cellsPerAxis;

        if (s_trianglesBuffer == null || s_allocatedTriangleCapacity < maxTriangles)
        {
            s_trianglesBuffer?.Release();
            s_trianglesBuffer = new ComputeBuffer(maxTriangles, Triangle.SizeOf, ComputeBufferType.Append);
            s_allocatedTriangleCapacity = maxTriangles;
        }

        if (s_trianglesCountBuffer == null)
        {
            s_trianglesCountBuffer = new ComputeBuffer(2, sizeof(int), ComputeBufferType.Raw);
        }
    }

    static void EnsureSharedStrokeBuffer(int strokeCount)
    {
        if (s_strokesBuffer == null || s_allocatedStrokeCapacity < strokeCount)
        {
            s_strokesBuffer?.Release();
            int newCapacity = Mathf.Max(strokeCount, 64);
            s_strokesBuffer = new ComputeBuffer(newCapacity, sizeof(float) * 5);
            s_allocatedStrokeCapacity = newCapacity;
        }

        if (s_strokeArrayCPU.Length < strokeCount)
        {
            s_strokeArrayCPU = new PlanetOctree.TerraformStroke[Mathf.Max(strokeCount, s_strokeArrayCPU.Length * 2)];
        }
    }

    public void FlushCollider()
    {
        IsQueuedForCollider = false;
        if (!_colliderDirty || MeshCollider == null)
        {
            return;
        }

        MeshCollider.sharedMesh = null;
        if (triangleCount > 0 && _mesh != null)
        {
            MeshCollider.sharedMesh = _mesh;
        }
        _colliderDirty = false;
    }

    void DispatchSingleStrokeKernel(ComputeBuffer targetWeightsBuffer, PlanetOctree.TerraformStroke stroke)
    {
        MarchingShader.SetBuffer(_updateWeightsKernel, "_Weights", targetWeightsBuffer);
        MarchingShader.SetInt("_ChunkSize", GridMetrics.PointsPerChunk(LOD));
        MarchingShader.SetVector("_ChunkOffset", transform.position);
        MarchingShader.SetVector("_HitPosition", stroke.HitPosition);
        MarchingShader.SetFloat("_BrushSize", stroke.BrushSize);
        MarchingShader.SetFloat("_TerraformStrength", stroke.Strength);
        MarchingShader.SetFloat("_Scale", ChunkScale);

        int groups = GridMetrics.ThreadGroups(LOD);
        MarchingShader.Dispatch(_updateWeightsKernel, groups, groups, groups);
    }

    public void ApplyTerraformBatchKernelOnly(List<PlanetOctree.TerraformStroke> strokes, ComputeBuffer targetWeightsBuffer)
    {
        if (strokes == null || strokes.Count == 0 || MarchingShader == null || targetWeightsBuffer == null)
        {
            return;
        }

        EnsureSharedBuffers();
        EnsureSharedStrokeBuffer(strokes.Count);
        CacheKernels();

        for (int i = 0; i < strokes.Count; i++)
        {
            s_strokeArrayCPU[i] = strokes[i];
        }
        s_strokesBuffer.SetData(s_strokeArrayCPU, 0, 0, strokes.Count);

        MarchingShader.SetBuffer(_updateWeightsBatchKernel, "_Weights", targetWeightsBuffer);
        MarchingShader.SetBuffer(_updateWeightsBatchKernel, "_StrokeBuffer", s_strokesBuffer);
        MarchingShader.SetInt("_StrokeCount", strokes.Count);
        MarchingShader.SetInt("_ChunkSize", GridMetrics.PointsPerChunk(LOD));
        MarchingShader.SetVector("_ChunkOffset", transform.position);
        MarchingShader.SetFloat("_Scale", ChunkScale);

        int groups = GridMetrics.ThreadGroups(LOD);
        MarchingShader.Dispatch(_updateWeightsBatchKernel, groups, groups, groups);
    }

    public void EditWeights(Vector3 hitPosition, float brushSize, bool add, bool updateCollider = true)
    {
        if (MarchingShader == null)
        {
            return;
        }

        PlanetOctree.TerraformStroke stroke = new PlanetOctree.TerraformStroke
        {
            HitPosition = hitPosition,
            BrushSize = brushSize,
            Strength = add ? 1f : -1f
        };
        _localStrokes.Add(stroke);
        ApplyIncrementalStroke(stroke, _localStrokes, updateCollider);
    }

    public void RemoveBoundaryNormals(Dictionary<Vector3Int, Vector3> sharedBoundaryNormals)
    {
        if (!_hasContributedBoundaryNormals || sharedBoundaryNormals == null)
        {
            return;
        }

        for (int i = 0; i < _boundaryWorldKeys.Count; i++)
        {
            Vector3Int key = _boundaryWorldKeys[i];
            if (sharedBoundaryNormals.TryGetValue(key, out Vector3 existing))
            {
                Vector3 remaining = existing - _localBoundaryNormals[i];
                if (remaining.sqrMagnitude <= 1e-12f)
                {
                    sharedBoundaryNormals.Remove(key);
                }
                else
                {
                    sharedBoundaryNormals[key] = remaining;
                }
            }
        }
        _hasContributedBoundaryNormals = false;
    }

    public void ContributeBoundaryNormals(Dictionary<Vector3Int, Vector3> sharedBoundaryNormals)
    {
        if (!UseSmoothShading || sharedBoundaryNormals == null || _boundaryVertexIndices.Count == 0)
        {
            return;
        }

        if (_hasContributedBoundaryNormals)
        {
            RemoveBoundaryNormals(sharedBoundaryNormals);
        }

        for (int i = 0; i < _boundaryWorldKeys.Count; i++)
        {
            Vector3Int key = _boundaryWorldKeys[i];
            Vector3 localNormal = _localBoundaryNormals[i];
            if (sharedBoundaryNormals.TryGetValue(key, out Vector3 existing))
            {
                sharedBoundaryNormals[key] = existing + localNormal;
            }
            else
            {
                sharedBoundaryNormals.Add(key, localNormal);
            }
        }
        _hasContributedBoundaryNormals = true;
    }

    public void ApplyBoundaryNormals(Dictionary<Vector3Int, Vector3> sharedBoundaryNormals)
    {
        if (!UseSmoothShading || sharedBoundaryNormals == null || _boundaryVertexIndices.Count == 0 || _mesh == null || _meshNormals.Count == 0)
        {
            return;
        }

        bool anyUpdated = false;
        for (int i = 0; i < _boundaryVertexIndices.Count; i++)
        {
            int vertIndex = _boundaryVertexIndices[i];
            if ((uint)vertIndex >= (uint)_meshNormals.Count)
            {
                continue;
            }

            Vector3Int key = _boundaryWorldKeys[i];
            if (sharedBoundaryNormals.TryGetValue(key, out Vector3 summedNormal) && summedNormal.sqrMagnitude > 1e-12f)
            {
                Vector3 newNormal = summedNormal.normalized;
                if ((newNormal - _meshNormals[vertIndex]).sqrMagnitude > 1e-6f)
                {
                    _meshNormals[vertIndex] = newNormal;
                    if (i < _boundarySkirtVertexIndices.Count)
                    {
                        int skirtIdx = _boundarySkirtVertexIndices[i];
                        if ((uint)skirtIdx < (uint)_meshNormals.Count)
                        {
                            _meshNormals[skirtIdx] = newNormal;
                        }
                    }
                    anyUpdated = true;
                }
            }
        }

        if (anyUpdated)
        {
            _mesh.SetNormals(_meshNormals);
        }
    }

    void ConstructMesh(ComputeBuffer weightsBuffer)
    {
        CacheKernels();

        int lodPoints = GridMetrics.PointsPerChunk(LOD);
        float isoLevel = NoiseGenerator != null ? NoiseGenerator.IsoLevel : 0.5f;

        MarchingShader.SetBuffer(_marchKernel, "_Triangles", s_trianglesBuffer);
        MarchingShader.SetBuffer(_marchKernel, "_ChunkMeta", s_trianglesCountBuffer);
        MarchingShader.SetBuffer(_marchKernel, "_Weights", weightsBuffer);

        MarchingShader.SetInt("_ChunkSize", lodPoints);
        MarchingShader.SetInt("_LODSize", lodPoints);
        MarchingShader.SetFloat("_IsoLevel", isoLevel);
        MarchingShader.SetFloat("_Scale", ChunkScale);
        MarchingShader.SetVector("_ChunkOffset", transform.position);

        s_trianglesBuffer.SetCounterValue(0);

        int groups = GridMetrics.ThreadGroups(LOD);
        MarchingShader.Dispatch(_marchKernel, groups, groups, groups);

        int triCount = ReadTriangleCount();
        triangleCount = triCount;

        if (triCount > 0)
        {
            if (s_triangleBufferCPU.Length < triCount)
            {
                int newSize = Mathf.Max(triCount, s_triangleBufferCPU.Length * 2);
                s_triangleBufferCPU = new Triangle[newSize];
            }
            s_trianglesBuffer.GetData(s_triangleBufferCPU, 0, 0, triCount);
        }

        BuildMeshFromTriangles(triCount);
    }

    int ReadTriangleCount()
    {
        ComputeBuffer.CopyCount(s_trianglesBuffer, s_trianglesCountBuffer, 0);
        s_trianglesCountBuffer.GetData(s_triCountCPU);
        int triCount = Mathf.Min(s_triCountCPU[0], s_allocatedTriangleCapacity);
        LastEmptyState = triCount <= 0
            ? (s_triCountCPU[1] == 1 ? EmptyChunkState.AllAir : EmptyChunkState.AllSolid)
            : EmptyChunkState.None;
        return triCount;
    }

    static Vector3Int QuantizeVertex(Vector3 v, float quantizeFactor)
    {
        return new Vector3Int(
            Mathf.RoundToInt(v.x * quantizeFactor),
            Mathf.RoundToInt(v.y * quantizeFactor),
            Mathf.RoundToInt(v.z * quantizeFactor)
        );
    }

    static Vector3Int QuantizeWorldPosition(Vector3 worldPos)
    {
        return new Vector3Int(
            Mathf.RoundToInt(worldPos.x * BoundaryQuantizeScale),
            Mathf.RoundToInt(worldPos.y * BoundaryQuantizeScale),
            Mathf.RoundToInt(worldPos.z * BoundaryQuantizeScale)
        );
    }

    bool IsBoundaryVertex(Vector3 localPos)
    {
        float eps = ChunkScale * 0.002f;
        return localPos.x <= eps || localPos.x >= ChunkScale - eps ||
               localPos.y <= eps || localPos.y >= ChunkScale - eps ||
               localPos.z <= eps || localPos.z >= ChunkScale - eps;
    }

    void PrepareMeshBuildState(int triCount)
    {
        s_rebuildTimer.Restart();

        s_vertexIndexMap.Clear();
        s_skirtVertexMap.Clear();
        s_weldedVertices.Clear();
        s_accumulatedNormals.Clear();
        s_weldedTriangles.Clear();
        _boundaryVertexIndices.Clear();
        _boundarySkirtVertexIndices.Clear();
        _boundaryWorldKeys.Clear();
        _localBoundaryNormals.Clear();
        _hasContributedBoundaryNormals = false;

        rawVertexCount = triCount * 3;
    }

    void BuildEmptyMesh()
    {
        PrepareMeshBuildState(0);
        triangleCount = 0;
        _baseSurfaceIndexCount = 0;
        _mesh.Clear();
        _meshNormals.Clear();
        _meshVertices.Clear();
        _meshTriangles.Clear();
        _colliderDirty = false;
        if (MeshCollider != null)
        {
            MeshCollider.sharedMesh = null;
        }
        weldedVertexCount = 0;
        vertexReductionPercent = 0f;
        s_rebuildTimer.Stop();
        lastMeshRebuildMs = (float)s_rebuildTimer.Elapsed.TotalMilliseconds;
    }

    void ProcessSingleTriangle(Triangle tri, float quantizeFactor)
    {
        // Area-weighted 3D cross-product face normal (Physics 12 right-hand rule)
        Vector3 faceNormal = Vector3.Cross(tri.b - tri.a, tri.c - tri.a);
        if (faceNormal.sqrMagnitude <= 1e-12f)
        {
            return;
        }

        if (UseSmoothShading)
        {
            AddWeldedVertex(tri.a, faceNormal, quantizeFactor);
            AddWeldedVertex(tri.b, faceNormal, quantizeFactor);
            AddWeldedVertex(tri.c, faceNormal, quantizeFactor);
        }
        else
        {
            Vector3 unitNormal = faceNormal.normalized;
            int baseIndex = s_weldedVertices.Count;

            s_weldedVertices.Add(tri.a);
            s_weldedVertices.Add(tri.b);
            s_weldedVertices.Add(tri.c);

            s_accumulatedNormals.Add(unitNormal);
            s_accumulatedNormals.Add(unitNormal);
            s_accumulatedNormals.Add(unitNormal);

            s_weldedTriangles.Add(baseIndex);
            s_weldedTriangles.Add(baseIndex + 1);
            s_weldedTriangles.Add(baseIndex + 2);
        }
    }

    void FinalizeWeldedMesh()
    {
        if (s_weldedVertices.Count == 0 || s_weldedTriangles.Count == 0)
        {
            triangleCount = 0;
            _baseSurfaceIndexCount = 0;
            _mesh.Clear();
            _meshNormals.Clear();
            _meshVertices.Clear();
            _meshTriangles.Clear();
            weldedVertexCount = 0;
            vertexReductionPercent = 0f;
            s_rebuildTimer.Stop();
            lastMeshRebuildMs = (float)s_rebuildTimer.Elapsed.TotalMilliseconds;
            return;
        }

        Vector3 chunkOrigin = transform.position;
        if (UseSmoothShading)
        {
            for (int i = 0; i < s_weldedVertices.Count; i++)
            {
                Vector3 rawNormalSum = s_accumulatedNormals[i];
                if (IsBoundaryVertex(s_weldedVertices[i]))
                {
                    _boundaryVertexIndices.Add(i);
                    _boundaryWorldKeys.Add(QuantizeWorldPosition(chunkOrigin + s_weldedVertices[i]));
                    _localBoundaryNormals.Add(rawNormalSum);
                }

                s_accumulatedNormals[i] = rawNormalSum.sqrMagnitude > 1e-12f ? rawNormalSum.normalized : Vector3.up;
            }
        }

        int baseIndexCount = s_weldedTriangles.Count;
        _baseSurfaceIndexCount = baseIndexCount;
        _meshVertices.Clear();
        _meshVertices.AddRange(s_weldedVertices);
        _meshTriangles.Clear();
        _meshTriangles.AddRange(s_weldedTriangles);

        // Generate watertight LOD-seam skirts for both smooth and flat-shaded chunks
        float eps = ChunkScale * 0.002f;
        float cellSize = ChunkScale / Mathf.Max(1, GridMetrics.PointsPerChunk(LOD) - 1);
        float skirtDepth = cellSize * 1.50f;
        float inset = cellSize * 0.02f;

        for (int t = 0; t < baseIndexCount; t += 3)
        {
            int i0 = s_weldedTriangles[t];
            int i1 = s_weldedTriangles[t + 1];
            int i2 = s_weldedTriangles[t + 2];

            TryAddSkirtEdge(i0, i1, eps, chunkOrigin, skirtDepth, inset);
            TryAddSkirtEdge(i1, i2, eps, chunkOrigin, skirtDepth, inset);
            TryAddSkirtEdge(i2, i0, eps, chunkOrigin, skirtDepth, inset);
        }

        if (UseSmoothShading)
        {
            for (int i = 0; i < _boundaryVertexIndices.Count; i++)
            {
                _boundarySkirtVertexIndices.Add(
                    s_skirtVertexMap.TryGetValue(_boundaryVertexIndices[i], out int skirtIdx) ? skirtIdx : -1
                );
            }
        }

        _meshNormals.Clear();
        _meshNormals.AddRange(s_accumulatedNormals);

        _mesh.Clear();
        _mesh.SetVertices(s_weldedVertices);
        _mesh.SetNormals(_meshNormals);
        _mesh.SetTriangles(s_weldedTriangles, 0, true);

        weldedVertexCount = s_weldedVertices.Count;
        vertexReductionPercent = rawVertexCount > 0
            ? (1f - (float)weldedVertexCount / rawVertexCount) * 100f
            : 0f;

        s_rebuildTimer.Stop();
        lastMeshRebuildMs = (float)s_rebuildTimer.Elapsed.TotalMilliseconds;
    }

    public bool RaycastMesh(Ray worldRay, float maxDistance, out float hitDistance)
    {
        hitDistance = float.MaxValue;
        if (!IsMeshVisible || _mesh == null)
        {
            return false;
        }

        Vector3 origin = worldRay.origin - transform.position;
        Vector3 dir = worldRay.direction;
        Ray localRay = new Ray(origin, dir);
        if (!_mesh.bounds.IntersectRay(localRay, out float boxDist) || boxDist > maxDistance)
        {
            return false;
        }

        bool hit = false;
        float bestDist = maxDistance;
        int indexCount = Mathf.Min(_baseSurfaceIndexCount, _meshTriangles.Count);

        for (int i = 0; i + 2 < indexCount; i += 3)
        {
            Vector3 v0 = _meshVertices[_meshTriangles[i]];
            Vector3 v1 = _meshVertices[_meshTriangles[i + 1]];
            Vector3 v2 = _meshVertices[_meshTriangles[i + 2]];

            Vector3 e1 = v1 - v0;
            Vector3 e2 = v2 - v0;
            Vector3 pvec = Vector3.Cross(dir, e2);
            float det = Vector3.Dot(e1, pvec);
            if (Mathf.Abs(det) < 1e-7f)
            {
                continue;
            }

            float invDet = 1f / det;
            Vector3 tvec = origin - v0;
            float u = Vector3.Dot(tvec, pvec) * invDet;
            if (u < 0f || u > 1f)
            {
                continue;
            }

            Vector3 qvec = Vector3.Cross(tvec, e1);
            float v = Vector3.Dot(dir, qvec) * invDet;
            if (v < 0f || u + v > 1f)
            {
                continue;
            }

            float t = Vector3.Dot(e2, qvec) * invDet;
            if (t > 0.01f && t < bestDist)
            {
                bestDist = t;
                hit = true;
            }
        }

        hitDistance = bestDist;
        return hit;
    }

    void BuildMeshFromTriangles(int triCount)
    {
        if (triCount <= 0)
        {
            BuildEmptyMesh();
            return;
        }

        PrepareMeshBuildState(triCount);
        float quantizeFactor = NormalizedQuantizeResolution / Mathf.Max(ChunkScale, 0.001f);

        for (int i = 0; i < triCount; i++)
        {
            ProcessSingleTriangle(s_triangleBufferCPU[i], quantizeFactor);
        }

        FinalizeWeldedMesh();
    }

    void BuildMeshFromNativeTriangles(NativeArray<Triangle> tris, int triCount)
    {
        if (triCount <= 0)
        {
            BuildEmptyMesh();
            return;
        }

        PrepareMeshBuildState(triCount);
        float quantizeFactor = NormalizedQuantizeResolution / Mathf.Max(ChunkScale, 0.001f);

        for (int i = 0; i < triCount; i++)
        {
            ProcessSingleTriangle(tris[i], quantizeFactor);
        }

        FinalizeWeldedMesh();
    }

    int GetSharedBoundaryFace(Vector3 a, Vector3 b, float eps)
    {
        if (a.x <= eps && b.x <= eps) return 0;
        if (a.x >= ChunkScale - eps && b.x >= ChunkScale - eps) return 1;
        if (a.y <= eps && b.y <= eps) return 2;
        if (a.y >= ChunkScale - eps && b.y >= ChunkScale - eps) return 3;
        if (a.z <= eps && b.z <= eps) return 4;
        if (a.z >= ChunkScale - eps && b.z >= ChunkScale - eps) return 5;
        return -1;
    }

    void TryAddSkirtEdge(int i0, int i1, float eps, Vector3 chunkOrigin, float skirtDepth, float inset)
    {
        if (i0 == i1 || GetSharedBoundaryFace(s_weldedVertices[i0], s_weldedVertices[i1], eps) < 0)
        {
            return;
        }

        int s0 = GetOrCreateSkirtVertex(i0, chunkOrigin, skirtDepth, inset);
        int s1 = GetOrCreateSkirtVertex(i1, chunkOrigin, skirtDepth, inset);

        s_weldedTriangles.Add(i1);
        s_weldedTriangles.Add(i0);
        s_weldedTriangles.Add(s0);

        s_weldedTriangles.Add(i1);
        s_weldedTriangles.Add(s0);
        s_weldedTriangles.Add(s1);
    }

    int GetOrCreateSkirtVertex(int boundaryVertIdx, Vector3 chunkOrigin, float skirtDepth, float inset)
    {
        if (s_skirtVertexMap.TryGetValue(boundaryVertIdx, out int existingSkirtIdx))
        {
            return existingSkirtIdx;
        }

        Vector3 pos = s_weldedVertices[boundaryVertIdx];
        Vector3 downDir;
        if (NoiseGenerator != null && NoiseGenerator.UseSphericalPlanet)
        {
            Vector3 radial = (chunkOrigin + pos) - NoiseGenerator.PlanetCenter;
            downDir = radial.sqrMagnitude > 1e-8f ? -radial.normalized : Vector3.down;
        }
        else
        {
            Vector3 n = s_accumulatedNormals[boundaryVertIdx];
            downDir = n.sqrMagnitude > 1e-8f ? -n.normalized : Vector3.down;
        }

        Vector3 skirtPos = pos + downDir * skirtDepth;
        skirtPos.x = Mathf.Clamp(skirtPos.x, inset, ChunkScale - inset);
        skirtPos.y = Mathf.Clamp(skirtPos.y, inset, ChunkScale - inset);
        skirtPos.z = Mathf.Clamp(skirtPos.z, inset, ChunkScale - inset);

        int skirtIdx = s_weldedVertices.Count;
        s_weldedVertices.Add(skirtPos);
        s_accumulatedNormals.Add(s_accumulatedNormals[boundaryVertIdx]);
        s_skirtVertexMap.Add(boundaryVertIdx, skirtIdx);
        return skirtIdx;
    }

    static int AddWeldedVertex(Vector3 position, Vector3 faceNormal, float quantizeFactor)
    {
        Vector3Int key = QuantizeVertex(position, quantizeFactor);
        if (s_vertexIndexMap.TryGetValue(key, out int sharedIndex))
        {
            s_weldedTriangles.Add(sharedIndex);
            s_accumulatedNormals[sharedIndex] += faceNormal;
            return sharedIndex;
        }

        int newIndex = s_weldedVertices.Count;
        s_vertexIndexMap.Add(key, newIndex);
        s_weldedVertices.Add(position);
        s_accumulatedNormals.Add(faceNormal);
        s_weldedTriangles.Add(newIndex);
        return newIndex;
    }

    static void ReleaseSharedBuffers()
    {
        WaitForAllAsyncReadbacks();

        for (int i = 0; i < s_asyncSlots.Length; i++)
        {
            s_asyncSlots[i].TrianglesBuffer?.Release();
            s_asyncSlots[i].TrianglesBuffer = null;
            s_asyncSlots[i].CountBuffer?.Release();
            s_asyncSlots[i].CountBuffer = null;
            s_asyncSlots[i].TriangleCapacity = 0;
        }

        s_trianglesBuffer?.Release();
        s_trianglesBuffer = null;

        s_trianglesCountBuffer?.Release();
        s_trianglesCountBuffer = null;

        s_scratchWeightsBuffer?.Release();
        s_scratchWeightsBuffer = null;

        s_strokesBuffer?.Release();
        s_strokesBuffer = null;

        while (s_freeLruWeightBuffers.Count > 0)
        {
            s_freeLruWeightBuffers.Pop()?.Release();
        }
        for (int i = s_lruChunks.Count - 1; i >= 0; i--)
        {
            Chunk c = s_lruChunks[i];
            if (c != null && c._cachedWeightsBuffer != null)
            {
                c._cachedWeightsBuffer.Release();
                c._cachedWeightsBuffer = null;
                c._cachedWeightsLod = -1;
                c.usesLruWeightCache = false;
            }
        }
        s_lruChunks.Clear();
        s_totalCreatedLruBuffers = 0;

        s_allocatedTriangleCapacity = -1;
        s_allocatedWeightCount = -1;
        s_allocatedStrokeCapacity = -1;
        s_allocatedLruWeightCount = -1;
    }
}
