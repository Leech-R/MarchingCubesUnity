public static class GridMetrics
{
    public const int NumThreads = 8;

    public const int Scale = 32;

    public const int GroundLevel = Scale / 2;

    // Power-of-two voxel cell counts (Cells = Points - 1: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80)
    public static int[] LODs = {
        9,
        17,
        25,
        33,
        41,
        49,
        57,
        65,
        73,
        81
    };

    public static int LastLod = LODs.Length - 1;

    public static int PointsPerChunk(int lod)
    {
        return LODs[lod];
    }

    public static int ThreadGroups(int lod)
    {
        return (LODs[lod] + NumThreads - 1) / NumThreads;
    }
}
