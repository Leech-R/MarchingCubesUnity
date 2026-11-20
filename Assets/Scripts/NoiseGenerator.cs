using UnityEngine;

public class NoiseGenerator : MonoBehaviour
{
    public ComputeShader NoiseShader;

    [SerializeField] float noiseScale = 1f;
    [SerializeField] float amplitude = 520f;
    [SerializeField] float frequency = 0.00028f;
    [SerializeField] int octaves = 4;
    [SerializeField, Range(0f, 1f)] float groundPercent = 0.2f;

    [SerializeField] float lacunarity = 2.0f;
    [SerializeField] float persistence = 0.5f;
    [SerializeField] bool usePerlinNoise = false;

    [Header("Spherical Planet Settings")]
    [SerializeField] bool useSphericalPlanet = true;
    [SerializeField] Vector3 planetCenter = Vector3.zero;
    [SerializeField] float planetRadius = 30000f;
    [SerializeField] float isoLevel = 0.5f;

    [Header("Subsurface Caverns")]
    [SerializeField] bool enableCaves = false;
    [SerializeField] float caveDepth = 220f;
    [SerializeField] float caveFrequency = 0.0055f;
    [SerializeField, Range(0.05f, 0.30f)] float caveThreshold = 0.15f;

    public float Amplitude => amplitude;
    public float CaveDepth => enableCaves ? caveDepth : 0f;

    public bool EnableCaves
    {
        get => enableCaves;
        set => enableCaves = value;
    }

    public bool UsePerlinNoise
    {
        get => usePerlinNoise;
        set => usePerlinNoise = value;
    }

    public bool UseSphericalPlanet
    {
        get => useSphericalPlanet;
        set => useSphericalPlanet = value;
    }

    public Vector3 PlanetCenter
    {
        get => planetCenter;
        set => planetCenter = value;
    }

    public float PlanetRadius
    {
        get => planetRadius;
        set => planetRadius = value;
    }

    public float IsoLevel
    {
        get => isoLevel;
        set => isoLevel = value;
    }

    /// <summary>
    /// Dispatches the 3D fractal noise compute kernel directly into an existing VRAM buffer.
    /// </summary>
    public void PopulateNoiseBuffer(ComputeBuffer weightsBuffer, int lod, Vector3 chunkOffset, float chunkScale)
    {
        int kernel = NoiseShader.FindKernel("GenerateNoise");

        NoiseShader.SetBuffer(kernel, "_Weights", weightsBuffer);

        NoiseShader.SetInt("_ChunkSize", GridMetrics.PointsPerChunk(lod));
        NoiseShader.SetFloat("_NoiseScale", noiseScale);
        NoiseShader.SetFloat("_Amplitude", amplitude);
        NoiseShader.SetFloat("_Frequency", frequency);
        NoiseShader.SetInt("_Octaves", octaves);
        NoiseShader.SetFloat("_GroundPercent", groundPercent);
        NoiseShader.SetFloat("_Scale", chunkScale);
        NoiseShader.SetInt("_GroundLevel", GridMetrics.GroundLevel);
        NoiseShader.SetVector("_ChunkOffset", chunkOffset);

        float safeLacunarity = lacunarity > 0.01f ? lacunarity : 2.0f;
        float safePersistence = persistence > 0.01f ? persistence : 0.5f;
        float safeIsoLevel = isoLevel != 0f ? isoLevel : 0.5f;

        NoiseShader.SetFloat("_Lacunarity", safeLacunarity);
        NoiseShader.SetFloat("_Persistence", safePersistence);
        NoiseShader.SetInt("_UsePerlinNoise", usePerlinNoise ? 1 : 0);
        NoiseShader.SetInt("_UseSphericalPlanet", useSphericalPlanet ? 1 : 0);
        NoiseShader.SetVector("_PlanetCenter", planetCenter);
        NoiseShader.SetFloat("_PlanetRadius", planetRadius);
        NoiseShader.SetFloat("_IsoLevel", safeIsoLevel);

        int pointsPerChunk = GridMetrics.PointsPerChunk(lod);
        float cellSize = chunkScale / Mathf.Max(1, pointsPerChunk - 1);
        NoiseShader.SetInt("_EnableCaves", enableCaves ? 1 : 0);
        NoiseShader.SetFloat("_CaveDepth", caveDepth);
        NoiseShader.SetFloat("_CaveFrequency", caveFrequency);
        NoiseShader.SetFloat("_CaveThreshold", caveThreshold);
        NoiseShader.SetFloat("_CellSize", cellSize);

        Shader.SetGlobalVector("_GlobalPlanetCenter", planetCenter);
        Shader.SetGlobalFloat("_GlobalPlanetRadius", planetRadius);
        Shader.SetGlobalFloat("_GlobalPlanetAmplitude", amplitude);
        Shader.SetGlobalFloat("_GlobalUseSphericalPlanet", useSphericalPlanet ? 1f : 0f);
        Shader.SetGlobalFloat("_GlobalCaveDepth", CaveDepth);

        int groups = GridMetrics.ThreadGroups(lod);
        NoiseShader.Dispatch(kernel, groups, groups, groups);
    }
}
