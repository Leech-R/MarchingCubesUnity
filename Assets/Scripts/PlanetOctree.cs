using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

public class PlanetOctree : MonoBehaviour
{
    [Header("Mode")]
    public bool EnableOctreeMode = true;
    public bool UseSphericalPlanet = true;

    [Header("Core References")]
    public Transform Viewer;
    public NoiseGenerator NoiseGenerator;
    public ComputeShader MarchingShader;
    public Material TerrainMaterial;
    public Chunk ChunkPrefab;

    [Header("Planet Geometry")]
    public Vector3 PlanetCenter = Vector3.zero;
    public float PlanetRadius = 30000f;
    public float RootSize = 65536f;

    [Header("Spherical Shell Culling & Spatial Buckets")]
    public float ShellMargin = 100f;
    public float StrokeBucketSize = 128f;

    [Header("Astroneer Partial Loading (Horizon & Frustum)")]
    [Range(0, 3)]
    public int MinShellDepth = 2;
    public float SurfaceRenderDistance = 10000f;
    public float MaxDetailDistance = 450f;
    public bool EnableHorizonCulling = true;
    public float HorizonMarginDegrees = 4.0f;
    public float UndergroundLoadRadius = 450f;
    public bool EnableFrustumCulling = true;
    public bool UnloadOutOfFrustumChunks = true;
    public float ImmediateBubbleRadius = 320f;
    public float FrustumUnloadDistance = 900f;
    public float ColliderActiveDistance = 260f;
    public bool AutoBakeNearbyColliders = false;
    [Range(2, 10)]
    public int ProgressiveGateDepth = 6;

    [Header("Octree LOD & Async Streaming Budget")]
    [Range(0, 14)]
    public int MaxDepth = 8;
    public float SplitDistanceMultiplier = 1.20f;
    [Range(0, 9)]
    public int BaseLOD = 1;
    [Range(0, 9)]
    public int MaxLeafLOD = 1;
    public bool UseAsyncGpuStreaming = true;
    public float UpdateMoveThreshold = 36f;
    public float RefreshMinInterval = 0.14f;
    public float FrameBuildBudgetMs = 1.5f;
    public int MaxAsyncDispatchesPerFrame = 3;
    public float VelocityLookAheadSeconds = 0.25f;
    [SerializeField] bool smoothShadingEnabled = false;

    [Header("Octree Telemetry")]
    [SerializeField] int activeChunkCount;
    [SerializeField] int pooledChunkCount;
    [SerializeField] int culledNodeCount;
    [SerializeField] int culledShellCount;
    [SerializeField] int culledHorizonCount;
    [SerializeField] int culledFrustumCount;
    [SerializeField] int emptyLeafCount;
    [SerializeField] int recordedStrokeCount;
    [SerializeField] int pendingBuildCount;

    public int ActiveChunkCount => activeChunkCount;
    public int PooledChunkCount => pooledChunkCount;
    public int CulledNodeCount => culledNodeCount;
    public int CulledShellCount => culledShellCount;
    public int CulledHorizonCount => culledHorizonCount;
    public int CulledFrustumCount => culledFrustumCount;
    public int EmptyLeafCount => emptyLeafCount;
    public int RecordedStrokeCount => recordedStrokeCount;
    public int PendingBuildCount => pendingBuildCount;
    public int InFlightAsyncCount => _inFlightNodes.Count;
    public float EffectiveRenderDistance => _cachedEffectiveRenderDist;
    public bool SmoothShadingEnabled { get => smoothShadingEnabled; private set => smoothShadingEnabled = value; }
    public bool ShowOctreeBounds { get; set; }
    public bool FreezeStreaming { get; set; }
    public bool PartialCullingEnabled => EnableHorizonCulling || UnloadOutOfFrustumChunks;

    Material _wireframeMaterial;

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct TerraformStroke
    {
        public Vector3 HitPosition;
        public float BrushSize;
        public float Strength;
    }

    class OctreeNode
    {
        public Vector3 Center;
        public float Size;
        public int Depth;
        public OctreeNode Parent;
        public OctreeNode[] Children;
        public Chunk AssignedChunk;
        public Chunk PendingChunk;
        public bool IsActiveLeaf;
        public bool IsQueuedForBuild;
        public Chunk.EmptyChunkState EmptyState;
        public bool IsEmptyConfirmed => EmptyState != Chunk.EmptyChunkState.None;
        public bool WasSubdivided;
        public bool HasMeshBounds;
        public Bounds LastMeshWorldBounds;

        public OctreeNode(Vector3 center, float size, int depth, OctreeNode parent)
        {
            Center = center;
            Size = size;
            Depth = depth;
            Parent = parent;
        }
    }

    struct RaycastCandidate
    {
        public Chunk Chunk;
        public float BoxDist;
    }

    readonly Stack<Chunk> _chunkPool = new Stack<Chunk>();
    readonly List<TerraformStroke> _strokeLog = new List<TerraformStroke>(1024);
    readonly Dictionary<Vector3Int, List<int>> _spatialStrokeBuckets = new Dictionary<Vector3Int, List<int>>(256);
    readonly HashSet<int> _queriedStrokeIndices = new HashSet<int>();
    readonly List<TerraformStroke> _reusableIntersectingStrokes = new List<TerraformStroke>(128);
    readonly List<Chunk> _activeChunks = new List<Chunk>(256);
    readonly List<Chunk> _modifiedChunksScratch = new List<Chunk>(32);
    readonly List<OctreeNode> _sculptNodesScratch = new List<OctreeNode>(32);
    readonly Dictionary<Vector3Int, Vector3> _sharedBoundaryNormals = new Dictionary<Vector3Int, Vector3>(16384);
    readonly List<OctreeNode> _pendingBuildList = new List<OctreeNode>(256);
    readonly List<OctreeNode> _inFlightNodes = new List<OctreeNode>(16);
    readonly Queue<Chunk> _colliderQueue = new Queue<Chunk>(64);
    readonly List<RaycastCandidate> _raycastCandidates = new List<RaycastCandidate>(32);
    readonly Stopwatch _frameTimer = new Stopwatch();
    readonly Plane[] _frustumPlanes = new Plane[6];

    OctreeNode _root;
    Camera _viewerCamera;
    bool _hasFrustumPlanes;
    bool _hasStrokeBounds;
    Vector3 _allStrokesMin;
    Vector3 _allStrokesMax;
    Vector3 _lastUpdateViewerPos;
    Vector3 _lastUpdateViewerForward;
    Vector3 _prevFrameViewerPos;
    Vector3 _viewerVelocity;
    Vector3 _lodFocusPos;
    bool _hasPrevViewerPos;
    bool _hasEvaluatedOnce;
    bool _needsProgressiveRefresh;
    int _spawnCounter;
    float _nextAllowedRefreshTime;

    // Cached per-refresh spherical shell & horizon constants
    Vector3 _cachedCamVec;
    float _cachedCamDist;
    float _cachedROccl;
    float _cachedRMin;
    float _cachedRMax;
    float _cachedCamHorizonAngle;
    float _cachedHorizonMarginRad;
    float _cachedEffectiveRenderDist = 10000f;

    private void Awake()
    {
        if (!EnableOctreeMode)
        {
            return;
        }

        // Keep template scene Chunk disabled so it does not render overlapping with Octree leaves
        if (ChunkPrefab != null && ChunkPrefab.gameObject.scene.IsValid())
        {
            ChunkPrefab.gameObject.SetActive(false);
        }
    }

    private void Start()
    {
        if (!EnableOctreeMode)
        {
            return;
        }

        if (Viewer == null && Camera.main != null)
        {
            Viewer = Camera.main.transform;
        }
        CacheViewerCamera();

        if (NoiseGenerator != null)
        {
            NoiseGenerator.UseSphericalPlanet = UseSphericalPlanet;
            NoiseGenerator.PlanetCenter = PlanetCenter;
            NoiseGenerator.PlanetRadius = PlanetRadius;
        }

        if (Viewer != null)
        {
            _prevFrameViewerPos = Viewer.position;
            _hasPrevViewerPos = true;
        }

        _root = new OctreeNode(PlanetCenter, RootSize, 0, null);
        RefreshOctree(synchronousInitialBuild: true);
    }

    private void Update()
    {
        if (!EnableOctreeMode || Viewer == null)
        {
            return;
        }

        if (FreezeStreaming)
        {
            _viewerVelocity = Vector3.zero;
            _prevFrameViewerPos = Viewer.position;
            _hasPrevViewerPos = true;
        }
        else
        {
            if (_hasPrevViewerPos && Time.deltaTime > 1e-5f)
            {
                Vector3 instantVel = (Viewer.position - _prevFrameViewerPos) / Time.deltaTime;
                _viewerVelocity = Vector3.Lerp(_viewerVelocity, instantVel, 0.35f);
            }
            _prevFrameViewerPos = Viewer.position;
            _hasPrevViewerPos = true;

            float altitude = GetViewerAltitude(Viewer.position);
            float adaptiveMoveThreshold = Mathf.Clamp(altitude * 0.03f, UpdateMoveThreshold, 500f);
            bool movedFar = Vector3.Distance(Viewer.position, _lastUpdateViewerPos) >= adaptiveMoveThreshold;
            bool turnedFar = Vector3.Dot(Viewer.forward, _lastUpdateViewerForward) < 0.975f;

            if (!_hasEvaluatedOnce || (Time.unscaledTime >= _nextAllowedRefreshTime && (_needsProgressiveRefresh || movedFar || turnedFar)))
            {
                RefreshOctree(synchronousInitialBuild: false);
            }
        }

        ProcessBuildAndColliderQueues();
    }

    float GetViewerAltitude(Vector3 viewerPos)
    {
        return UseSphericalPlanet
            ? Mathf.Max(0f, Vector3.Distance(viewerPos, PlanetCenter) - PlanetRadius)
            : Mathf.Max(0f, viewerPos.y - GridMetrics.GroundLevel);
    }

    Vector3 ComputeLodFocusPosition(Vector3 viewerPos)
    {
        float altitude = GetViewerAltitude(viewerPos);
        float lookAheadCap = Mathf.Clamp(altitude * 0.45f, 160f, 3000f);
        return viewerPos + Vector3.ClampMagnitude(_viewerVelocity * VelocityLookAheadSeconds, lookAheadCap);
    }

    void CacheViewerCamera()
    {
        if (Viewer != null)
        {
            _viewerCamera = Viewer.GetComponent<Camera>();
        }
        if (_viewerCamera == null)
        {
            _viewerCamera = Camera.main;
        }
    }

    private void OnDestroy()
    {
        if (_wireframeMaterial != null)
        {
            Destroy(_wireframeMaterial);
            _wireframeMaterial = null;
        }
    }

    private void OnRenderObject()
    {
        if (!ShowOctreeBounds || !EnableOctreeMode || _root == null || Camera.current == null)
        {
            return;
        }

        if (_wireframeMaterial == null)
        {
            Shader lineShader = Shader.Find("Hidden/Internal-Colored");
            if (lineShader == null)
            {
                return;
            }
            _wireframeMaterial = new Material(lineShader) { hideFlags = HideFlags.HideAndDontSave };
            _wireframeMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            _wireframeMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            _wireframeMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            _wireframeMaterial.SetInt("_ZWrite", 0);
        }

        _wireframeMaterial.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity);
        GL.Begin(GL.LINES);
        DrawNodeWireframeRecursive(_root);
        GL.End();
        GL.PopMatrix();
    }

    static Color GetDepthColor(int depth)
    {
        float t = Mathf.Clamp01((depth - 2f) / 10f);
        Color col = Color.HSVToRGB(Mathf.Lerp(0.55f, 0.0f, t), 0.85f, 1.0f);
        col.a = Mathf.Lerp(0.35f, 0.98f, t);
        return col;
    }

    static void DrawNodeWireframeRecursive(OctreeNode node)
    {
        if (node == null)
        {
            return;
        }

        if (node.AssignedChunk != null && node.AssignedChunk.TryGetComponent<MeshRenderer>(out MeshRenderer mr) && mr.enabled)
        {
            DrawWireBox(node.Center, node.Size, GetDepthColor(node.Depth));
        }

        if (node.Children != null)
        {
            for (int i = 0; i < 8; i++)
            {
                DrawNodeWireframeRecursive(node.Children[i]);
            }
        }
    }

    static void DrawWireBox(Vector3 center, float size, Color color)
    {
        float h = size * 0.5f;
        Vector3 c000 = center + new Vector3(-h, -h, -h);
        Vector3 c100 = center + new Vector3(h, -h, -h);
        Vector3 c110 = center + new Vector3(h, h, -h);
        Vector3 c010 = center + new Vector3(-h, h, -h);
        Vector3 c001 = center + new Vector3(-h, -h, h);
        Vector3 c101 = center + new Vector3(h, -h, h);
        Vector3 c111 = center + new Vector3(h, h, h);
        Vector3 c011 = center + new Vector3(-h, h, h);

        GL.Color(color);
        GL.Vertex(c000); GL.Vertex(c100);
        GL.Vertex(c100); GL.Vertex(c110);
        GL.Vertex(c110); GL.Vertex(c010);
        GL.Vertex(c010); GL.Vertex(c000);

        GL.Vertex(c001); GL.Vertex(c101);
        GL.Vertex(c101); GL.Vertex(c111);
        GL.Vertex(c111); GL.Vertex(c011);
        GL.Vertex(c011); GL.Vertex(c001);

        GL.Vertex(c000); GL.Vertex(c001);
        GL.Vertex(c100); GL.Vertex(c101);
        GL.Vertex(c110); GL.Vertex(c111);
        GL.Vertex(c010); GL.Vertex(c011);
    }

    Vector3Int WorldToBucket(Vector3 pos)
    {
        float invSize = 1f / Mathf.Max(StrokeBucketSize, 0.1f);
        return new Vector3Int(
            Mathf.FloorToInt(pos.x * invSize),
            Mathf.FloorToInt(pos.y * invSize),
            Mathf.FloorToInt(pos.z * invSize)
        );
    }

    public void RecordTerraformStroke(Vector3 hitPosition, float brushSize, bool add)
    {
        float deltaStrength = add ? 1f : -1f;
        Vector3 strokeMin = hitPosition - Vector3.one * brushSize;
        Vector3 strokeMax = hitPosition + Vector3.one * brushSize;
        if (!_hasStrokeBounds)
        {
            _allStrokesMin = strokeMin;
            _allStrokesMax = strokeMax;
            _hasStrokeBounds = true;
        }
        else
        {
            _allStrokesMin = Vector3.Min(_allStrokesMin, strokeMin);
            _allStrokesMax = Vector3.Max(_allStrokesMax, strokeMax);
        }

        if (_strokeLog.Count > 0)
        {
            int lastIdx = _strokeLog.Count - 1;
            TerraformStroke last = _strokeLog[lastIdx];
            float mergeThreshold = brushSize * 0.05f;
            if (Mathf.Abs(last.BrushSize - brushSize) < 1e-3f &&
                (last.HitPosition - hitPosition).sqrMagnitude <= mergeThreshold * mergeThreshold)
            {
                last.Strength += deltaStrength;
                _strokeLog[lastIdx] = last;
                return;
            }
        }

        int strokeIndex = _strokeLog.Count;
        _strokeLog.Add(new TerraformStroke
        {
            HitPosition = hitPosition,
            BrushSize = brushSize,
            Strength = deltaStrength
        });

        Vector3Int minBucket = WorldToBucket(hitPosition - Vector3.one * brushSize);
        Vector3Int maxBucket = WorldToBucket(hitPosition + Vector3.one * brushSize);

        for (int x = minBucket.x; x <= maxBucket.x; x++)
        {
            for (int y = minBucket.y; y <= maxBucket.y; y++)
            {
                for (int z = minBucket.z; z <= maxBucket.z; z++)
                {
                    Vector3Int key = new Vector3Int(x, y, z);
                    if (!_spatialStrokeBuckets.TryGetValue(key, out List<int> bucket))
                    {
                        bucket = new List<int>(8);
                        _spatialStrokeBuckets.Add(key, bucket);
                    }
                    bucket.Add(strokeIndex);
                }
            }
        }

        recordedStrokeCount = _strokeLog.Count;
    }

    public void TogglePartialCulling()
    {
        bool next = !PartialCullingEnabled;
        EnableHorizonCulling = next;
        UnloadOutOfFrustumChunks = next;
        RefreshOctree(synchronousInitialBuild: false);
    }

    public void RefreshOctree()
    {
        RefreshOctree(synchronousInitialBuild: false);
    }

    public void SetSmoothShading(bool smooth)
    {
        if (SmoothShadingEnabled == smooth)
        {
            return;
        }

        SmoothShadingEnabled = smooth;
        _sharedBoundaryNormals.Clear();
        Vector3 viewerPos = Viewer != null ? Viewer.position : _lastUpdateViewerPos;

        for (int i = 0; i < _activeChunks.Count; i++)
        {
            Chunk chunk = _activeChunks[i];
            if (chunk == null || !chunk.gameObject.activeInHierarchy)
            {
                continue;
            }

            chunk.UseSmoothShading = smooth;
            Vector3 chunkOrigin = chunk.transform.position;
            Vector3 chunkMax = chunkOrigin + Vector3.one * chunk.ChunkScale;
            CollectIntersectingStrokes(chunkOrigin, chunkMax, _reusableIntersectingStrokes);
            chunk.RebuildWithStrokes(_reusableIntersectingStrokes, updateCollider: false);
            if (AutoBakeNearbyColliders && GetDistanceToChunk(chunk, viewerPos) <= ColliderActiveDistance)
            {
                EnqueueCollider(chunk);
            }

            if (smooth)
            {
                chunk.ContributeBoundaryNormals(_sharedBoundaryNormals);
            }
        }

        if (smooth)
        {
            for (int i = 0; i < _activeChunks.Count; i++)
            {
                Chunk chunk = _activeChunks[i];
                if (chunk != null && chunk.gameObject.activeInHierarchy)
                {
                    chunk.ApplyBoundaryNormals(_sharedBoundaryNormals);
                }
            }
        }
    }

    public void ToggleNoiseType()
    {
        if (NoiseGenerator == null)
        {
            return;
        }

        NoiseGenerator.UsePerlinNoise = !NoiseGenerator.UsePerlinNoise;
        RebuildWholePlanetSynchronously();
    }

    public void ClearAllStrokesAndRebuild()
    {
        _strokeLog.Clear();
        _spatialStrokeBuckets.Clear();
        _hasStrokeBounds = false;
        recordedStrokeCount = 0;
        RebuildWholePlanetSynchronously();
    }

    public void GetMeshTelemetry(
        out int totalTriangles,
        out int totalRawVerts,
        out int totalWeldedVerts,
        out float reductionPercent,
        out int lruCachedChunks)
    {
        totalTriangles = 0;
        totalRawVerts = 0;
        totalWeldedVerts = 0;
        lruCachedChunks = 0;

        for (int i = 0; i < _activeChunks.Count; i++)
        {
            Chunk chunk = _activeChunks[i];
            if (chunk == null)
            {
                continue;
            }

            totalTriangles += chunk.TriangleCount;
            totalRawVerts += chunk.RawVertexCount;
            totalWeldedVerts += chunk.WeldedVertexCount;
            if (chunk.UsesLruWeightCache)
            {
                lruCachedChunks++;
            }
        }

        reductionPercent = totalRawVerts > 0
            ? (1f - (float)totalWeldedVerts / totalRawVerts) * 100f
            : 0f;
    }

    void RebuildWholePlanetSynchronously()
    {
        Chunk.WaitForAllAsyncReadbacks();
        _pendingBuildList.Clear();
        _inFlightNodes.Clear();
        _colliderQueue.Clear();
        _sharedBoundaryNormals.Clear();

        if (_root != null)
        {
            ReleaseSubtree(_root);
            _root.Children = null;
            _root.IsQueuedForBuild = false;
            _root.EmptyState = Chunk.EmptyChunkState.None;
            _root.WasSubdivided = false;
            _root.HasMeshBounds = false;
        }

        RefreshOctree(synchronousInitialBuild: true);
    }

    void RefreshOctree(bool synchronousInitialBuild)
    {
        if (_root == null || Viewer == null || MarchingShader == null || NoiseGenerator == null)
        {
            return;
        }

        if (_viewerCamera == null)
        {
            CacheViewerCamera();
        }
        if (EnableFrustumCulling && _viewerCamera != null)
        {
            GeometryUtility.CalculateFrustumPlanes(_viewerCamera, _frustumPlanes);
            _hasFrustumPlanes = true;
        }
        else
        {
            _hasFrustumPlanes = false;
        }

        _lastUpdateViewerPos = Viewer.position;
        _lastUpdateViewerForward = Viewer.forward;
        _lodFocusPos = synchronousInitialBuild
            ? Viewer.position
            : ComputeLodFocusPosition(Viewer.position);
        _hasEvaluatedOnce = true;
        _needsProgressiveRefresh = false;
        _nextAllowedRefreshTime = Time.unscaledTime + RefreshMinInterval;

        float amp = NoiseGenerator.Amplitude;
        float caveD = NoiseGenerator.CaveDepth;
        float alt = GetViewerAltitude(Viewer.position);
        _cachedEffectiveRenderDist = Mathf.Max(SurfaceRenderDistance, alt * 2.4f + SurfaceRenderDistance * 0.5f);
        Shader.SetGlobalFloat("_GlobalRenderDistance", _cachedEffectiveRenderDist);
        _cachedCamVec = Viewer.position - PlanetCenter;
        _cachedCamDist = _cachedCamVec.magnitude;
        _cachedROccl = Mathf.Max(1f, PlanetRadius - amp * 0.55f);
        _cachedRMin = Mathf.Max(0f, PlanetRadius - Mathf.Max(amp * 0.5f, caveD) - ShellMargin);
        _cachedRMax = PlanetRadius + amp + ShellMargin;
        _cachedCamHorizonAngle = _cachedCamDist > _cachedROccl + 1f
            ? Mathf.Acos(Mathf.Clamp(_cachedROccl / _cachedCamDist, -1f, 1f))
            : -1f;
        _cachedHorizonMarginRad = HorizonMarginDegrees * Mathf.Deg2Rad;

        activeChunkCount = 0;
        culledNodeCount = 0;
        culledShellCount = 0;
        culledHorizonCount = 0;
        culledFrustumCount = 0;
        emptyLeafCount = 0;
        _activeChunks.Clear();

        UpdateNode(_root, Viewer.position, Viewer.forward, synchronousInitialBuild);
        culledNodeCount = culledShellCount + culledHorizonCount + culledFrustumCount;

        if (synchronousInitialBuild)
        {
            if (SmoothShadingEnabled)
            {
                _sharedBoundaryNormals.Clear();
                for (int i = 0; i < _activeChunks.Count; i++)
                {
                    _activeChunks[i].ContributeBoundaryNormals(_sharedBoundaryNormals);
                }
                for (int i = 0; i < _activeChunks.Count; i++)
                {
                    _activeChunks[i].ApplyBoundaryNormals(_sharedBoundaryNormals);
                }
            }
        }
        else if (AutoBakeNearbyColliders)
        {
            for (int i = 0; i < _activeChunks.Count; i++)
            {
                Chunk chunk = _activeChunks[i];
                if (chunk != null && chunk.IsColliderDirty && !chunk.IsQueuedForCollider &&
                    GetDistanceToChunk(chunk, Viewer.position) <= ColliderActiveDistance)
                {
                    EnqueueCollider(chunk);
                }
            }
        }

        pooledChunkCount = _chunkPool.Count;
        pendingBuildCount = _pendingBuildList.Count;
    }

    void EnqueueCollider(Chunk chunk)
    {
        if (chunk == null || chunk.IsQueuedForCollider)
        {
            return;
        }
        chunk.IsQueuedForCollider = true;
        _colliderQueue.Enqueue(chunk);
    }

    static float GetDistanceToChunk(Chunk chunk, Vector3 point)
    {
        Vector3 min = chunk.transform.position;
        Vector3 max = min + Vector3.one * chunk.ChunkScale;
        Vector3 closest = Vector3.Max(min, Vector3.Min(point, max));
        return Vector3.Distance(point, closest);
    }

    static float GetDistanceToNode(OctreeNode node, Vector3 point)
    {
        float half = node.Size * 0.5f;
        Vector3 min = node.Center - Vector3.one * half;
        Vector3 max = node.Center + Vector3.one * half;
        Vector3 closest = Vector3.Max(min, Vector3.Min(point, max));
        return Vector3.Distance(point, closest);
    }

    OctreeNode PopHighestPriorityPendingNode()
    {
        Vector3 viewerPos = Viewer != null ? Viewer.position : _lastUpdateViewerPos;
        Vector3 viewerFwd = Viewer != null ? Viewer.forward : _lastUpdateViewerForward;
        Vector3 focusPos = FreezeStreaming
            ? viewerPos
            : ComputeLodFocusPosition(viewerPos);

        int bestIndex = -1;
        float bestScore = float.MaxValue;

        for (int i = _pendingBuildList.Count - 1; i >= 0; i--)
        {
            OctreeNode node = _pendingBuildList[i];
            if (node == null || !node.IsActiveLeaf || node.AssignedChunk != null || node.PendingChunk != null || node.IsEmptyConfirmed)
            {
                if (node != null)
                {
                    node.IsQueuedForBuild = false;
                }
                int last = _pendingBuildList.Count - 1;
                _pendingBuildList[i] = _pendingBuildList[last];
                _pendingBuildList.RemoveAt(last);
                if (bestIndex == last)
                {
                    bestIndex = i;
                }
                continue;
            }

            float half = node.Size * 0.5f;
            Vector3 min = node.Center - Vector3.one * half;
            Vector3 max = node.Center + Vector3.one * half;
            Vector3 closest = Vector3.Max(min, Vector3.Min(viewerPos, max));
            float dist = Vector3.Distance(viewerPos, closest);
            Vector3 closestFocus = Vector3.Max(min, Vector3.Min(focusPos, max));
            float distFocus = Vector3.Distance(focusPos, closestFocus);
            float lodDist = Mathf.Min(dist, distFocus);

            bool inFront = dist < 1f || Vector3.Dot(viewerFwd, (closest - viewerPos) / dist) > 0f;
            float score = lodDist * (inFront ? 0.40f : 1.40f) + node.Depth * 10f;

            if (node.Parent != null && node.Parent.HasMeshBounds)
            {
                Vector3 pMin = node.Parent.LastMeshWorldBounds.min;
                Vector3 pMax = node.Parent.LastMeshWorldBounds.max;
                if (max.x >= pMin.x && min.x <= pMax.x &&
                    max.y >= pMin.y && min.y <= pMax.y &&
                    max.z >= pMin.z && min.z <= pMax.z)
                {
                    score *= 0.25f;
                }
            }

            if (score < bestScore)
            {
                bestScore = score;
                bestIndex = i;
            }
        }

        if (bestIndex < 0)
        {
            return null;
        }

        OctreeNode best = _pendingBuildList[bestIndex];
        int lastIdx = _pendingBuildList.Count - 1;
        _pendingBuildList[bestIndex] = _pendingBuildList[lastIdx];
        _pendingBuildList.RemoveAt(lastIdx);
        best.IsQueuedForBuild = false;
        return best;
    }

    static void RecordNodeMeshBounds(OctreeNode node, Chunk chunk)
    {
        if (node == null || chunk == null)
        {
            return;
        }
        Bounds localBounds = chunk.LocalMeshBounds;
        Vector3 center = chunk.transform.position + localBounds.center;
        Vector3 size = localBounds.size + Vector3.one * (node.Size * 0.35f);
        node.LastMeshWorldBounds = new Bounds(center, size);
        node.HasMeshBounds = true;
    }

    void HandleCancelledInFlightNode(OctreeNode node, Chunk finished)
    {
        ReleaseChunk(finished);
        if (node.IsActiveLeaf && node.AssignedChunk == null && !node.IsEmptyConfirmed && !node.IsQueuedForBuild)
        {
            node.IsQueuedForBuild = true;
            _pendingBuildList.Add(node);
        }
    }

    void HandleCompletedEmptyNode(OctreeNode node, Chunk finished, Vector3 viewerPos)
    {
        Chunk.EmptyChunkState emptyState = finished.LastEmptyState != Chunk.EmptyChunkState.None
            ? finished.LastEmptyState
            : Chunk.EmptyChunkState.AllAir;
        ReleaseChunk(finished);
        node.EmptyState = emptyState;
        CompleteLeafBuild(node, viewerPos, enqueueCollider: false, pendingColliderChunks: null);
    }

    void ProcessBuildAndColliderQueues()
    {
        _modifiedChunksScratch.Clear();
        _frameTimer.Restart();
        bool activeDirty = false;
        Vector3 viewerPos = Viewer != null ? Viewer.position : _lastUpdateViewerPos;

        Chunk.ReclaimCancelledAsyncSlots();

        // Phase 1 (non-blocking, always runs): Advance Stage-1 count readbacks & prune empty chunks immediately
        for (int i = _inFlightNodes.Count - 1; i >= 0; i--)
        {
            OctreeNode node = _inFlightNodes[i];
            if (node == null || node.PendingChunk == null)
            {
                _inFlightNodes.RemoveAt(i);
                continue;
            }

            Chunk.AsyncBuildPollResult res = node.PendingChunk.AdvanceAsyncStage1();
            if (res == Chunk.AsyncBuildPollResult.StillPending)
            {
                continue;
            }

            _inFlightNodes.RemoveAt(i);
            Chunk finished = node.PendingChunk;
            node.PendingChunk = null;

            if (res == Chunk.AsyncBuildPollResult.CancelledOrFailed)
            {
                HandleCancelledInFlightNode(node, finished);
            }
            else if (res == Chunk.AsyncBuildPollResult.CompletedEmpty)
            {
                HandleCompletedEmptyNode(node, finished, viewerPos);
                activeDirty = true;
            }
        }

        // Phase 2 (CPU-budgeted): Finalize closest ready Stage-2 meshes within FrameBuildBudgetMs
        while (_frameTimer.Elapsed.TotalMilliseconds < FrameBuildBudgetMs)
        {
            int readyIndex = -1;
            float closestDist = float.MaxValue;
            for (int i = 0; i < _inFlightNodes.Count; i++)
            {
                OctreeNode candidate = _inFlightNodes[i];
                if (candidate == null || candidate.PendingChunk == null || !candidate.PendingChunk.IsAsyncStage2Ready)
                {
                    continue;
                }
                float d = GetDistanceToNode(candidate, viewerPos);
                if (d < closestDist)
                {
                    closestDist = d;
                    readyIndex = i;
                }
            }

            if (readyIndex < 0)
            {
                break;
            }

            OctreeNode node = _inFlightNodes[readyIndex];
            _inFlightNodes.RemoveAt(readyIndex);
            Chunk finished = node.PendingChunk;
            node.PendingChunk = null;

            Chunk.AsyncBuildPollResult res = finished.TryFinalizeAsyncStage2(updateCollider: false);
            if (res == Chunk.AsyncBuildPollResult.CancelledOrFailed)
            {
                HandleCancelledInFlightNode(node, finished);
            }
            else if (res == Chunk.AsyncBuildPollResult.CompletedEmpty)
            {
                HandleCompletedEmptyNode(node, finished, viewerPos);
                activeDirty = true;
            }
            else if (res == Chunk.AsyncBuildPollResult.CompletedWithMesh)
            {
                node.AssignedChunk = finished;
                CompleteLeafBuild(node, viewerPos, enqueueCollider: true, pendingColliderChunks: null);
                activeDirty = true;
            }
        }

        // Phase 3 (always refills GPU queue): Keep up to 16 async GPU slots fed regardless of Phase 2 CPU budget
        if (_pendingBuildList.Count > 0)
        {
            bool useAsync = UseAsyncGpuStreaming && SystemInfo.supportsAsyncGPUReadback;
            if (useAsync)
            {
                int dispatches = 0;
                while (dispatches < MaxAsyncDispatchesPerFrame && Chunk.HasFreeAsyncSlot() && _pendingBuildList.Count > 0)
                {
                    OctreeNode node = PopHighestPriorityPendingNode();
                    if (node == null)
                    {
                        break;
                    }

                    Chunk chunk = PrepareChunkForNode(node);
                    Vector3 chunkOrigin = chunk.transform.position;
                    Vector3 chunkMax = chunkOrigin + Vector3.one * chunk.ChunkScale;
                    CollectIntersectingStrokes(chunkOrigin, chunkMax, _reusableIntersectingStrokes);

                    if (chunk.TryBeginAsyncRebuildWithStrokes(_reusableIntersectingStrokes))
                    {
                        node.PendingChunk = chunk;
                        _inFlightNodes.Add(node);
                        dispatches++;
                    }
                    else
                    {
                        chunk.RebuildWithStrokes(_reusableIntersectingStrokes, updateCollider: false);
                        FinalizeSyncBuiltNode(node, chunk, viewerPos, allowCollider: false);
                        CompleteLeafBuild(node, viewerPos, enqueueCollider: false, pendingColliderChunks: null);
                        activeDirty = true;
                    }
                }
            }
            else
            {
                while (_frameTimer.Elapsed.TotalMilliseconds < FrameBuildBudgetMs && _pendingBuildList.Count > 0)
                {
                    OctreeNode node = PopHighestPriorityPendingNode();
                    if (node == null)
                    {
                        break;
                    }

                    BuildChunkForNodeSync(node, updateCollider: false);
                    CompleteLeafBuild(node, viewerPos, enqueueCollider: false, pendingColliderChunks: null);
                    activeDirty = true;
                }
            }
        }

        if (activeDirty || _modifiedChunksScratch.Count > 0)
        {
            _activeChunks.Clear();
            activeChunkCount = 0;
            emptyLeafCount = 0;
            CollectActiveChunks(_root);
            if (SmoothShadingEnabled)
            {
                SyncBoundaryNormalsAround(_modifiedChunksScratch);
            }
        }

        // Phase 4: Lazy distance-gated physics collider bake (at most 1 nearby chunk per frame)
        if (_colliderQueue.Count > 0 && _frameTimer.Elapsed.TotalMilliseconds < FrameBuildBudgetMs)
        {
            while (_colliderQueue.Count > 0)
            {
                Chunk chunkToBake = _colliderQueue.Dequeue();
                if (chunkToBake == null)
                {
                    continue;
                }

                chunkToBake.IsQueuedForCollider = false;
                if (!chunkToBake.gameObject.activeInHierarchy || !chunkToBake.IsColliderDirty)
                {
                    continue;
                }

                if (Viewer != null && GetDistanceToChunk(chunkToBake, viewerPos) > ColliderActiveDistance * 1.25f)
                {
                    continue;
                }

                chunkToBake.FlushCollider();
                break;
            }
        }

        pendingBuildCount = _pendingBuildList.Count;
        pooledChunkCount = _chunkPool.Count;
    }

    public void EnsureAllCollidersBaked()
    {
        while (_colliderQueue.Count > 0)
        {
            Chunk chunk = _colliderQueue.Dequeue();
            if (chunk != null)
            {
                chunk.IsQueuedForCollider = false;
                if (chunk.gameObject.activeInHierarchy)
                {
                    chunk.FlushCollider();
                }
            }
        }
    }

    public void EnsureCollidersBakedAlongRay(Ray ray, float maxDistance, float padRadius)
    {
        for (int i = 0; i < _activeChunks.Count; i++)
        {
            Chunk chunk = _activeChunks[i];
            if (chunk == null || !chunk.gameObject.activeInHierarchy || !chunk.IsColliderDirty)
            {
                continue;
            }

            float scale = chunk.ChunkScale;
            Vector3 center = chunk.transform.position + Vector3.one * (scale * 0.5f);
            Bounds paddedBounds = new Bounds(center, Vector3.one * (scale + padRadius * 2f));
            if (paddedBounds.IntersectRay(ray, out float hitDist) && hitDist <= maxDistance)
            {
                chunk.FlushCollider();
            }
        }
    }

    public bool RaycastTerrain(Ray ray, float maxDistance, out Vector3 hitPoint)
    {
        _raycastCandidates.Clear();
        for (int i = 0; i < _activeChunks.Count; i++)
        {
            Chunk chunk = _activeChunks[i];
            if (chunk == null || !chunk.IsMeshVisible)
            {
                continue;
            }

            if (chunk.WorldMeshBounds.IntersectRay(ray, out float boxDist) && boxDist <= maxDistance)
            {
                _raycastCandidates.Add(new RaycastCandidate
                {
                    Chunk = chunk,
                    BoxDist = boxDist
                });
            }
        }

        // In-place insertion sort by box distance ascending (zero GC allocations)
        for (int i = 1; i < _raycastCandidates.Count; i++)
        {
            RaycastCandidate key = _raycastCandidates[i];
            int j = i - 1;
            while (j >= 0 && _raycastCandidates[j].BoxDist > key.BoxDist)
            {
                _raycastCandidates[j + 1] = _raycastCandidates[j];
                j--;
            }
            _raycastCandidates[j + 1] = key;
        }

        float bestDist = maxDistance;
        bool hitAny = false;

        for (int i = 0; i < _raycastCandidates.Count; i++)
        {
            RaycastCandidate candidate = _raycastCandidates[i];
            if (candidate.BoxDist > bestDist)
            {
                break;
            }

            if (candidate.Chunk.RaycastMesh(ray, bestDist, out float triDist) && triDist < bestDist)
            {
                bestDist = triDist;
                hitAny = true;
            }
        }

        hitPoint = hitAny ? (ray.origin + ray.direction * bestDist) : default;
        return hitAny;
    }

    void CollectActiveChunks(OctreeNode node)
    {
        if (node == null)
        {
            return;
        }

        if (node.IsEmptyConfirmed)
        {
            emptyLeafCount++;
        }

        if (node.AssignedChunk != null)
        {
            _activeChunks.Add(node.AssignedChunk);
            activeChunkCount++;
        }

        if (node.Children != null)
        {
            for (int i = 0; i < 8; i++)
            {
                CollectActiveChunks(node.Children[i]);
            }
        }
    }

    static bool ChunksTouch(Chunk a, Chunk b)
    {
        if (a == b)
        {
            return true;
        }

        Vector3 aMin = a.transform.position;
        Vector3 aMax = aMin + Vector3.one * a.ChunkScale;
        Vector3 bMin = b.transform.position;
        Vector3 bMax = bMin + Vector3.one * b.ChunkScale;
        float eps = 0.25f;

        return aMax.x + eps >= bMin.x && aMin.x - eps <= bMax.x &&
               aMax.y + eps >= bMin.y && aMin.y - eps <= bMax.y &&
               aMax.z + eps >= bMin.z && aMin.z - eps <= bMax.z;
    }

    public void SyncBoundaryNormalsAround(List<Chunk> modifiedChunks)
    {
        if (modifiedChunks == null || modifiedChunks.Count == 0 || _activeChunks.Count == 0)
        {
            return;
        }

        Vector3 viewerPos = Viewer != null ? Viewer.position : _lastUpdateViewerPos;
        for (int i = 0; i < _activeChunks.Count; i++)
        {
            Chunk active = _activeChunks[i];
            if (active == null || (Viewer != null && GetDistanceToChunk(active, viewerPos) > 1200f))
            {
                continue;
            }

            for (int m = 0; m < modifiedChunks.Count; m++)
            {
                if (ChunksTouch(active, modifiedChunks[m]))
                {
                    active.ApplyBoundaryNormals(_sharedBoundaryNormals);
                    break;
                }
            }
        }
    }

    public void ApplySculptStrokeToChunk(Chunk chunk, Vector3 hitPosition, float brushSize, bool add, bool updateCollider)
    {
        if (chunk == null)
        {
            return;
        }

        TerraformStroke newStroke = new TerraformStroke
        {
            HitPosition = hitPosition,
            BrushSize = brushSize,
            Strength = add ? 1f : -1f
        };

        Vector3 chunkOrigin = chunk.transform.position;
        Vector3 chunkMax = chunkOrigin + Vector3.one * chunk.ChunkScale;
        CollectIntersectingStrokes(chunkOrigin, chunkMax, _reusableIntersectingStrokes);

        chunk.RemoveBoundaryNormals(_sharedBoundaryNormals);
        chunk.ApplyIncrementalStroke(newStroke, _reusableIntersectingStrokes, updateCollider);
        chunk.ContributeBoundaryNormals(_sharedBoundaryNormals);
    }

    public void ApplySculptAtWorldPoint(Vector3 hitPosition, float brushSize, bool add, HashSet<Chunk> pendingColliderChunks)
    {
        RecordTerraformStroke(hitPosition, brushSize, add);
        if (_root == null)
        {
            return;
        }

        _sculptNodesScratch.Clear();
        CollectIntersectingLeafNodes(_root, hitPosition, brushSize, _sculptNodesScratch);
        if (_sculptNodesScratch.Count == 0)
        {
            return;
        }

        _modifiedChunksScratch.Clear();
        bool structureChanged = false;
        Vector3 viewerPos = Viewer != null ? Viewer.position : _lastUpdateViewerPos;

        for (int i = 0; i < _sculptNodesScratch.Count; i++)
        {
            OctreeNode node = _sculptNodesScratch[i];

            if (node.PendingChunk != null)
            {
                ReleaseChunk(node.PendingChunk);
                _inFlightNodes.Remove(node);
                node.PendingChunk = null;
            }

            if (node.IsEmptyConfirmed)
            {
                if ((!add && node.EmptyState == Chunk.EmptyChunkState.AllAir) ||
                    (add && node.EmptyState == Chunk.EmptyChunkState.AllSolid))
                {
                    continue;
                }
                node.EmptyState = Chunk.EmptyChunkState.None;
                BuildChunkForNodeSync(node, updateCollider: false);
                CompleteLeafBuild(node, viewerPos, enqueueCollider: false, pendingColliderChunks);
                structureChanged = true;
                continue;
            }

            if (node.AssignedChunk != null && node.AssignedChunk.gameObject.activeInHierarchy)
            {
                ApplySculptStrokeToChunk(node.AssignedChunk, hitPosition, brushSize, add, updateCollider: false);
                RecordNodeMeshBounds(node, node.AssignedChunk);
                _modifiedChunksScratch.Add(node.AssignedChunk);
                pendingColliderChunks?.Add(node.AssignedChunk);
            }
            else if (node.IsActiveLeaf)
            {
                node.IsQueuedForBuild = false;
                _pendingBuildList.Remove(node);
                BuildChunkForNodeSync(node, updateCollider: false);
                CompleteLeafBuild(node, viewerPos, enqueueCollider: false, pendingColliderChunks);
                structureChanged = true;
            }
        }

        if (structureChanged)
        {
            _activeChunks.Clear();
            activeChunkCount = 0;
            emptyLeafCount = 0;
            CollectActiveChunks(_root);
        }

        if (SmoothShadingEnabled)
        {
            SyncBoundaryNormalsAround(_modifiedChunksScratch);
        }
        pendingBuildCount = _pendingBuildList.Count;
        pooledChunkCount = _chunkPool.Count;
    }

    void CompleteLeafBuild(OctreeNode node, Vector3 viewerPos, bool enqueueCollider, HashSet<Chunk> pendingColliderChunks)
    {
        if (node.Children != null)
        {
            ReleaseSubtree(node, preserveAssignedChunk: true);
            node.Children = null;
        }

        if (node.AssignedChunk != null)
        {
            RecordNodeMeshBounds(node, node.AssignedChunk);

            if (HasActiveAncestorChunk(node.Parent))
            {
                SetChunkRendererEnabled(node.AssignedChunk, false);
            }

            if (AutoBakeNearbyColliders && enqueueCollider && GetDistanceToNode(node, viewerPos) <= ColliderActiveDistance)
            {
                EnqueueCollider(node.AssignedChunk);
            }

            if (SmoothShadingEnabled)
            {
                node.AssignedChunk.ContributeBoundaryNormals(_sharedBoundaryNormals);
            }
            _modifiedChunksScratch.Add(node.AssignedChunk);
            pendingColliderChunks?.Add(node.AssignedChunk);

            if (node.Depth < MaxDepth &&
                Mathf.Min(GetDistanceToNode(node, viewerPos), GetDistanceToNode(node, _lodFocusPos)) <= GetSplitDistanceForNode(node))
            {
                _needsProgressiveRefresh = true;
            }
        }

        TryReleaseCoveredParentChunk(node.Parent);
    }

    static void CollectIntersectingLeafNodes(OctreeNode node, Vector3 sphereCenter, float radius, List<OctreeNode> results)
    {
        if (node == null)
        {
            return;
        }

        float half = node.Size * 0.5f;
        Vector3 boxMin = node.Center - Vector3.one * half;
        Vector3 boxMax = node.Center + Vector3.one * half;
        Vector3 clamped = new Vector3(
            Mathf.Clamp(sphereCenter.x, boxMin.x, boxMax.x),
            Mathf.Clamp(sphereCenter.y, boxMin.y, boxMax.y),
            Mathf.Clamp(sphereCenter.z, boxMin.z, boxMax.z)
        );
        if ((clamped - sphereCenter).sqrMagnitude > radius * radius)
        {
            return;
        }

        if (node.IsActiveLeaf)
        {
            results.Add(node);
            return;
        }

        if (node.Children != null)
        {
            for (int i = 0; i < 8; i++)
            {
                CollectIntersectingLeafNodes(node.Children[i], sphereCenter, radius, results);
            }
        }

        if (node.AssignedChunk != null)
        {
            results.Add(node);
        }
    }

    int MapDepthToPowerOfTwoLod(int depth, float distToViewer)
    {
        return Mathf.Clamp(MaxLeafLOD, 1, GridMetrics.LastLod);
    }

    float GetSplitDistanceForNode(OctreeNode node)
    {
        int levelsFromMax = MaxDepth - node.Depth;
        if (levelsFromMax <= 1)
        {
            return node.Size * Mathf.Min(SplitDistanceMultiplier * 0.72f, 0.90f);
        }
        if (levelsFromMax == 2)
        {
            return node.Size * Mathf.Min(SplitDistanceMultiplier, 1.20f);
        }
        return node.Size * Mathf.Max(SplitDistanceMultiplier, 1.35f);
    }

    bool IntersectsSurfaceShell(OctreeNode node)
    {
        if (NoiseGenerator == null)
        {
            return true;
        }

        float half = node.Size * 0.5f;
        Vector3 min = node.Center - Vector3.one * half;
        Vector3 max = node.Center + Vector3.one * half;

        if (!UseSphericalPlanet)
        {
            float groundY = GridMetrics.GroundLevel;
            float terrainHalfBand = NoiseGenerator.Amplitude + ShellMargin;
            if (max.y >= groundY - terrainHalfBand && min.y <= groundY + terrainHalfBand)
            {
                return true;
            }
            return HasIntersectingStroke(min, max);
        }

        Vector3 closest = new Vector3(
            Mathf.Clamp(PlanetCenter.x, min.x, max.x),
            Mathf.Clamp(PlanetCenter.y, min.y, max.y),
            Mathf.Clamp(PlanetCenter.z, min.z, max.z)
        );
        float minDist = Vector3.Distance(closest, PlanetCenter);

        Vector3 farthestOffset = new Vector3(
            Mathf.Abs(node.Center.x - PlanetCenter.x) + half,
            Mathf.Abs(node.Center.y - PlanetCenter.y) + half,
            Mathf.Abs(node.Center.z - PlanetCenter.z) + half
        );
        float maxDist = farthestOffset.magnitude;

        if (minDist <= _cachedRMax && maxDist >= _cachedRMin)
        {
            return true;
        }

        return HasIntersectingStroke(min, max);
    }

    bool IsBeyondPlanetHorizon(OctreeNode node, float distToViewer)
    {
        if (!EnableHorizonCulling || !UseSphericalPlanet)
        {
            return false;
        }

        if (_cachedCamDist <= _cachedROccl + 1f)
        {
            return distToViewer > UndergroundLoadRadius;
        }

        Vector3 nodeVec = node.Center - PlanetCenter;
        float nodeDist = nodeVec.magnitude;
        float nodeRadius = node.Size * 0.8660254f;

        if (nodeDist <= nodeRadius + 1f)
        {
            return false;
        }

        float nodeAngularRadius = Mathf.Asin(Mathf.Min(1f, nodeRadius / nodeDist));
        float rPeak = Mathf.Min(nodeDist + nodeRadius, _cachedRMax);
        float peakHorizonAngle = rPeak > _cachedROccl
            ? Mathf.Acos(Mathf.Clamp(_cachedROccl / rPeak, -1f, 1f))
            : 0f;
        float maxVisibleAngle = _cachedCamHorizonAngle + peakHorizonAngle + nodeAngularRadius + _cachedHorizonMarginRad;

        float dot = Vector3.Dot(_cachedCamVec, nodeVec) / Mathf.Max(_cachedCamDist * nodeDist, 1e-4f);
        float actualAngle = Mathf.Acos(Mathf.Clamp(dot, -1f, 1f));
        return actualAngle > maxVisibleAngle;
    }

    bool HasIntersectingStroke(Vector3 boxMin, Vector3 boxMax)
    {
        if (_strokeLog.Count == 0 || !_hasStrokeBounds ||
            boxMax.x < _allStrokesMin.x || boxMin.x > _allStrokesMax.x ||
            boxMax.y < _allStrokesMin.y || boxMin.y > _allStrokesMax.y ||
            boxMax.z < _allStrokesMin.z || boxMin.z > _allStrokesMax.z)
        {
            return false;
        }

        Vector3 clampedMin = Vector3.Max(boxMin, _allStrokesMin);
        Vector3 clampedMax = Vector3.Min(boxMax, _allStrokesMax);
        Vector3Int minBucket = WorldToBucket(clampedMin);
        Vector3Int maxBucket = WorldToBucket(clampedMax);
        long bucketSpan = (long)(maxBucket.x - minBucket.x + 1) *
                          (long)(maxBucket.y - minBucket.y + 1) *
                          (long)(maxBucket.z - minBucket.z + 1);
        if (bucketSpan > _strokeLog.Count)
        {
            for (int i = 0; i < _strokeLog.Count; i++)
            {
                if (StrokeIntersectsBox(_strokeLog[i], boxMin, boxMax))
                {
                    return true;
                }
            }
            return false;
        }

        for (int x = minBucket.x; x <= maxBucket.x; x++)
        {
            for (int y = minBucket.y; y <= maxBucket.y; y++)
            {
                for (int z = minBucket.z; z <= maxBucket.z; z++)
                {
                    if (!_spatialStrokeBuckets.TryGetValue(new Vector3Int(x, y, z), out List<int> bucket))
                    {
                        continue;
                    }

                    for (int i = 0; i < bucket.Count; i++)
                    {
                        if (StrokeIntersectsBox(_strokeLog[bucket[i]], boxMin, boxMax))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    void CollectIntersectingStrokes(Vector3 boxMin, Vector3 boxMax, List<TerraformStroke> results)
    {
        results.Clear();
        if (_strokeLog.Count == 0 || !_hasStrokeBounds ||
            boxMax.x < _allStrokesMin.x || boxMin.x > _allStrokesMax.x ||
            boxMax.y < _allStrokesMin.y || boxMin.y > _allStrokesMax.y ||
            boxMax.z < _allStrokesMin.z || boxMin.z > _allStrokesMax.z)
        {
            return;
        }

        Vector3 clampedMin = Vector3.Max(boxMin, _allStrokesMin);
        Vector3 clampedMax = Vector3.Min(boxMax, _allStrokesMax);
        Vector3Int minBucket = WorldToBucket(clampedMin);
        Vector3Int maxBucket = WorldToBucket(clampedMax);
        long bucketSpan = (long)(maxBucket.x - minBucket.x + 1) *
                          (long)(maxBucket.y - minBucket.y + 1) *
                          (long)(maxBucket.z - minBucket.z + 1);
        if (bucketSpan > _strokeLog.Count)
        {
            for (int i = 0; i < _strokeLog.Count; i++)
            {
                TerraformStroke stroke = _strokeLog[i];
                if (StrokeIntersectsBox(stroke, boxMin, boxMax))
                {
                    results.Add(stroke);
                }
            }
            return;
        }

        _queriedStrokeIndices.Clear();
        for (int x = minBucket.x; x <= maxBucket.x; x++)
        {
            for (int y = minBucket.y; y <= maxBucket.y; y++)
            {
                for (int z = minBucket.z; z <= maxBucket.z; z++)
                {
                    if (!_spatialStrokeBuckets.TryGetValue(new Vector3Int(x, y, z), out List<int> bucket))
                    {
                        continue;
                    }

                    for (int i = 0; i < bucket.Count; i++)
                    {
                        int idx = bucket[i];
                        if (_queriedStrokeIndices.Add(idx))
                        {
                            TerraformStroke stroke = _strokeLog[idx];
                            if (StrokeIntersectsBox(stroke, boxMin, boxMax))
                            {
                                results.Add(stroke);
                            }
                        }
                    }
                }
            }
        }
    }

    static bool StrokeIntersectsBox(TerraformStroke stroke, Vector3 boxMin, Vector3 boxMax)
    {
        Vector3 clamped = new Vector3(
            Mathf.Clamp(stroke.HitPosition.x, boxMin.x, boxMax.x),
            Mathf.Clamp(stroke.HitPosition.y, boxMin.y, boxMax.y),
            Mathf.Clamp(stroke.HitPosition.z, boxMin.z, boxMax.z)
        );
        return (clamped - stroke.HitPosition).sqrMagnitude <= stroke.BrushSize * stroke.BrushSize;
    }

    Chunk PrepareChunkForNode(OctreeNode node)
    {
        Chunk chunk = AcquireChunk();
        Vector3 chunkOrigin = node.Center - Vector3.one * (node.Size * 0.5f);
        Vector3 chunkMax = chunkOrigin + Vector3.one * node.Size;
        Vector3 closestOnNode = Vector3.Max(chunkOrigin, Vector3.Min(_lastUpdateViewerPos, chunkMax));
        float distToViewer = Vector3.Distance(_lastUpdateViewerPos, closestOnNode);

        chunk.transform.SetParent(transform, false);
        chunk.transform.position = chunkOrigin;

        int lod = MapDepthToPowerOfTwoLod(node.Depth, distToViewer);
        chunk.UseSmoothShading = SmoothShadingEnabled;
        chunk.InitializeOctreeChunk(MarchingShader, NoiseGenerator, TerrainMaterial, node.Size, lod, buildMesh: false);
        return chunk;
    }

    void FinalizeSyncBuiltNode(OctreeNode node, Chunk chunk, Vector3 viewerPos, bool allowCollider)
    {
        if (chunk.TriangleCount <= 0)
        {
            Chunk.EmptyChunkState emptyState = chunk.LastEmptyState != Chunk.EmptyChunkState.None
                ? chunk.LastEmptyState
                : Chunk.EmptyChunkState.AllAir;
            ReleaseChunk(chunk);
            node.AssignedChunk = null;
            node.EmptyState = emptyState;
        }
        else
        {
            node.AssignedChunk = chunk;
            RecordNodeMeshBounds(node, chunk);
            if (AutoBakeNearbyColliders && !allowCollider && GetDistanceToNode(node, viewerPos) <= ColliderActiveDistance)
            {
                EnqueueCollider(chunk);
            }
        }
    }

    void BuildChunkForNodeSync(OctreeNode node, bool updateCollider)
    {
        Chunk chunk = PrepareChunkForNode(node);
        Vector3 chunkOrigin = chunk.transform.position;
        Vector3 chunkMax = chunkOrigin + Vector3.one * node.Size;
        CollectIntersectingStrokes(chunkOrigin, chunkMax, _reusableIntersectingStrokes);
        chunk.RebuildWithStrokes(_reusableIntersectingStrokes, updateCollider);
        FinalizeSyncBuiltNode(node, chunk, _lastUpdateViewerPos, updateCollider);
    }

    static bool IsNodeBuiltOrCulled(OctreeNode node)
    {
        if ((!node.IsActiveLeaf && node.Children == null) || node.IsEmptyConfirmed || node.AssignedChunk != null)
        {
            return true;
        }
        if (node.Children != null)
        {
            for (int i = 0; i < 8; i++)
            {
                if (!IsNodeBuiltOrCulled(node.Children[i]))
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    void TryReleaseCoveredParentChunk(OctreeNode parent)
    {
        while (parent != null)
        {
            if (parent.AssignedChunk != null && IsNodeBuiltOrCulled(parent))
            {
                if (!HasActiveAncestorChunk(parent.Parent))
                {
                    SetSubtreeRenderersEnabled(parent, true);
                }
                ReleaseChunk(parent.AssignedChunk);
                parent.AssignedChunk = null;
            }
            parent = parent.Parent;
        }
    }

    static bool HasActiveAncestorChunk(OctreeNode ancestor)
    {
        while (ancestor != null)
        {
            if (ancestor.AssignedChunk != null)
            {
                return true;
            }
            ancestor = ancestor.Parent;
        }
        return false;
    }

    static void SetChunkRendererEnabled(Chunk chunk, bool enabled)
    {
        if (chunk != null && chunk.TryGetComponent<MeshRenderer>(out MeshRenderer mr))
        {
            mr.enabled = enabled;
        }
    }

    static void SetSubtreeRenderersEnabled(OctreeNode parent, bool enabled)
    {
        if (parent == null || parent.Children == null)
        {
            return;
        }

        for (int i = 0; i < 8; i++)
        {
            OctreeNode child = parent.Children[i];
            if (child == null)
            {
                continue;
            }

            if (child.AssignedChunk != null)
            {
                SetChunkRendererEnabled(child.AssignedChunk, enabled);
            }
            else if (child.Children != null)
            {
                SetSubtreeRenderersEnabled(child, enabled);
            }
        }
    }

    void CollectSubtreeChunks(OctreeNode parent)
    {
        if (parent == null || parent.Children == null)
        {
            return;
        }

        for (int i = 0; i < 8; i++)
        {
            OctreeNode child = parent.Children[i];
            if (child == null)
            {
                continue;
            }

            if (child.IsEmptyConfirmed)
            {
                emptyLeafCount++;
            }

            if (child.AssignedChunk != null)
            {
                _activeChunks.Add(child.AssignedChunk);
                activeChunkCount++;
            }

            if (child.Children != null)
            {
                CollectSubtreeChunks(child);
            }
        }
    }

    void UpdateNode(OctreeNode node, Vector3 viewerPos, Vector3 viewerForward, bool synchronousBuild)
    {
        // Stage 1: Spherical Shell Culling
        if (!IntersectsSurfaceShell(node))
        {
            ReleaseSubtree(node);
            node.Children = null;
            culledShellCount++;
            return;
        }

        float half = node.Size * 0.5f;
        Vector3 min = node.Center - Vector3.one * half;
        Vector3 max = node.Center + Vector3.one * half;
        Vector3 closestOnNode = Vector3.Max(min, Vector3.Min(viewerPos, max));
        float dist = Vector3.Distance(viewerPos, closestOnNode);

        // Stage 2: Planetary Horizon Culling & Altitude-Adaptive Render Distance Cap
        if (IsBeyondPlanetHorizon(node, dist) || (node.Depth >= MinShellDepth && dist > _cachedEffectiveRenderDist))
        {
            ReleaseSubtree(node);
            node.Children = null;
            culledHorizonCount++;
            return;
        }

        // Stage 3: 6-Plane Camera Frustum Culling & Unloading (preserving 360-degree safety bubble)
        bool outsideFrustum = EnableFrustumCulling && _hasFrustumPlanes &&
            !GeometryUtility.TestPlanesAABB(_frustumPlanes, new Bounds(node.Center, Vector3.one * (node.Size * 1.65f)));

        if (outsideFrustum && UnloadOutOfFrustumChunks && dist > FrustumUnloadDistance && node.Depth >= MinShellDepth)
        {
            ReleaseSubtree(node);
            node.Children = null;
            culledFrustumCount++;
            return;
        }

        // Distance-based LOD split with velocity look-ahead, hysteresis, and IVT empty-air/solid-bedrock pruning
        Vector3 closestToFocus = Vector3.Max(min, Vector3.Min(_lodFocusPos, max));
        float distFocus = Vector3.Distance(_lodFocusPos, closestToFocus);
        float lodDist = Mathf.Min(dist, distFocus);

        float splitDist = GetSplitDistanceForNode(node);
        float effectiveSplitDist = node.WasSubdivided ? splitDist * 1.18f : splitDist;
        bool allowSubdivide = lodDist <= effectiveSplitDist;

        if (outsideFrustum && dist > ImmediateBubbleRadius)
        {
            allowSubdivide = false;
        }

        if (allowSubdivide && node.Depth >= MaxDepth - 1 && lodDist > MaxDetailDistance)
        {
            allowSubdivide = false;
        }

        if (allowSubdivide && node.Depth >= MinShellDepth && node.IsEmptyConfirmed)
        {
            float cellSize = node.Size / Mathf.Max(1, GridMetrics.PointsPerChunk(MapDepthToPowerOfTwoLod(node.Depth, dist)) - 1);
            bool keepAllAir = node.EmptyState == Chunk.EmptyChunkState.AllAir && cellSize <= 65.0f;
            bool keepAllSolid = node.EmptyState == Chunk.EmptyChunkState.AllSolid &&
                (NoiseGenerator == null || !NoiseGenerator.EnableCaves || cellSize <= 65.0f);
            if ((keepAllAir || keepAllSolid) && !HasIntersectingStroke(min, max))
            {
                allowSubdivide = false;
            }
        }

        // Progressive surface refinement gate: require Depth >= ProgressiveGateDepth to build and confirm
        // surface geometry before subdividing into 8 finer children (eliminates close-up empty-node spikes).
        if (allowSubdivide && !synchronousBuild && node.Depth >= ProgressiveGateDepth)
        {
            bool hasSurfaceEvidence = node.AssignedChunk != null || node.Children != null || node.HasMeshBounds;
            if (!hasSurfaceEvidence && !HasIntersectingStroke(min, max))
            {
                allowSubdivide = false;
            }
        }

        bool shouldSubdivide = (node.Depth < MinShellDepth) ||
                               (node.Depth < MaxDepth && allowSubdivide);

        if (shouldSubdivide)
        {
            node.IsActiveLeaf = false;
            node.IsQueuedForBuild = false;
            node.EmptyState = Chunk.EmptyChunkState.None;
            node.WasSubdivided = true;

            if (node.PendingChunk != null)
            {
                ReleaseChunk(node.PendingChunk);
                _inFlightNodes.Remove(node);
                node.PendingChunk = null;
            }

            if (synchronousBuild && node.AssignedChunk != null)
            {
                ReleaseChunk(node.AssignedChunk);
                node.AssignedChunk = null;
            }

            if (node.Children == null)
            {
                CreateChildren(node);
            }

            for (int i = 0; i < 8; i++)
            {
                UpdateNode(node.Children[i], viewerPos, viewerForward, synchronousBuild);
            }

            if (node.AssignedChunk != null)
            {
                if (IsNodeBuiltOrCulled(node))
                {
                    if (!HasActiveAncestorChunk(node.Parent))
                    {
                        SetSubtreeRenderersEnabled(node, true);
                    }
                    ReleaseChunk(node.AssignedChunk);
                    node.AssignedChunk = null;
                }
                else
                {
                    _activeChunks.Add(node.AssignedChunk);
                    activeChunkCount++;
                }
            }
        }
        else
        {
            node.WasSubdivided = false;

            if (node.Children != null && (node.AssignedChunk != null || node.IsEmptyConfirmed || synchronousBuild))
            {
                ReleaseSubtree(node, preserveAssignedChunk: true);
                node.Children = null;
            }

            node.IsActiveLeaf = true;

            if (node.AssignedChunk == null && !node.IsEmptyConfirmed && node.PendingChunk == null)
            {
                if (synchronousBuild)
                {
                    BuildChunkForNodeSync(node, updateCollider: AutoBakeNearbyColliders && dist <= ColliderActiveDistance);
                }
                else if (!node.IsQueuedForBuild)
                {
                    node.IsQueuedForBuild = true;
                    _pendingBuildList.Add(node);
                }
            }

            if (node.IsEmptyConfirmed)
            {
                emptyLeafCount++;
            }

            if (node.AssignedChunk != null)
            {
                _activeChunks.Add(node.AssignedChunk);
                activeChunkCount++;
            }
            else if (node.Children != null && !node.IsEmptyConfirmed)
            {
                CollectSubtreeChunks(node);
            }
        }
    }

    static void CreateChildren(OctreeNode parent)
    {
        parent.Children = new OctreeNode[8];
        float childSize = parent.Size * 0.5f;
        float quarterSize = parent.Size * 0.25f;

        int index = 0;
        for (int x = -1; x <= 1; x += 2)
        {
            for (int y = -1; y <= 1; y += 2)
            {
                for (int z = -1; z <= 1; z += 2)
                {
                    Vector3 offset = new Vector3(x * quarterSize, y * quarterSize, z * quarterSize);
                    parent.Children[index++] = new OctreeNode(parent.Center + offset, childSize, parent.Depth + 1, parent);
                }
            }
        }
    }

    void ReleaseSubtree(OctreeNode node, bool preserveAssignedChunk = false)
    {
        if (!preserveAssignedChunk)
        {
            node.IsActiveLeaf = false;
            node.IsQueuedForBuild = false;
            node.EmptyState = Chunk.EmptyChunkState.None;
            node.WasSubdivided = false;
            node.HasMeshBounds = false;

            if (node.PendingChunk != null)
            {
                ReleaseChunk(node.PendingChunk);
                _inFlightNodes.Remove(node);
                node.PendingChunk = null;
            }

            if (node.AssignedChunk != null)
            {
                ReleaseChunk(node.AssignedChunk);
                node.AssignedChunk = null;
            }
        }

        if (node.Children == null)
        {
            return;
        }

        for (int i = 0; i < 8; i++)
        {
            ReleaseSubtree(node.Children[i], preserveAssignedChunk: false);
        }
    }

    Chunk AcquireChunk()
    {
        while (_chunkPool.Count > 0)
        {
            Chunk pooled = _chunkPool.Pop();
            if (pooled != null)
            {
                pooled.gameObject.SetActive(true);
                SetChunkRendererEnabled(pooled, true);
                return pooled;
            }
        }

        if (ChunkPrefab != null)
        {
            Chunk instance = Instantiate(ChunkPrefab, transform);
            instance.gameObject.SetActive(true);
            SetChunkRendererEnabled(instance, true);
            return instance;
        }

        GameObject go = new GameObject($"OctreeChunk_{_spawnCounter++}");
        go.transform.SetParent(transform, false);
        go.AddComponent<MeshFilter>();
        go.AddComponent<MeshRenderer>();
        go.AddComponent<MeshCollider>();
        return go.AddComponent<Chunk>();
    }

    void ReleaseChunk(Chunk chunk)
    {
        if (chunk == null)
        {
            return;
        }

        chunk.RemoveBoundaryNormals(_sharedBoundaryNormals);
        chunk.ClearChunkData();
        chunk.gameObject.SetActive(false);
        _chunkPool.Push(chunk);
    }
}
