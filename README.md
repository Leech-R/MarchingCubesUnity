# GPU-Accelerated 3D Procedural Terrain & Caves (Marching Cubes in Unity)

A real-time procedural voxel terrain and cave generation system built in **Unity (URP)** using **C#** and **HLSL Compute Shaders**. This project implements the **Marching Cubes algorithm** entirely on the GPU to polygonize 3D scalar density fields generated with **FastNoiseLite (3D Ridged OpenSimplex2 fractal noise)**, complete with **dynamic Level of Detail (LOD)** and **real-time interactive terraforming**.

---

## Key Features

- **GPU-Accelerated Marching Cubes (`HLSL` Compute Shaders):** Evaluates 3D voxel grids in parallel using `8x8x8` thread groups (`[numthreads(8, 8, 8)]`), outputting triangles directly into a GPU `AppendStructuredBuffer` without CPU bottlenecks.
- **3D Procedural Cave & Terrain Noise:** Generates volumetric scalar fields on the GPU using **FastNoiseLite** (`OpenSimplex2` with `Ridged` multi-fractal octaves) to create complex 3D overhangs, tunnels, and cave networks that standard 2D heightmaps cannot represent.
- **Real-Time Raycast Terraforming:** Supports interactive sculpting (adding and excavating terrain) at runtime via camera raycasting and a dedicated `UpdateWeights` compute kernel.
- **Dynamic Level of Detail (LOD):** Decouples the underlying high-resolution density field (`80³` points per chunk) from the polygonization resolution, supporting 10 selectable LOD tiers (`8³` up to `80³`) on the fly.

---

## How It Works (Architecture & Pipeline)

The system is divided into a **3-stage GPU-to-CPU pipeline** orchestrated by C# components and executed on the GPU via HLSL Compute Shaders:

```
+-----------------------------------------------------------------------+
| 1. 3D Noise Generation (GPU: NoiseCompute.compute)                    |
|    - Evaluates 3D OpenSimplex2 Ridged Fractal Noise at each grid pt   |
|    - Stores scalar density values in a 1D flattened _Weights array    |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 2. Isosurface Extraction (GPU: MarchingCubesCompute.compute)          |
|    - Samples 8 corners of each voxel cell in parallel                 |
|    - Computes an 8-bit cubeIndex (0..255) against _IsoLevel (0.5)     |
|    - Looks up intersected edges in triTable[256][16]                  |
|    - Linearly interpolates vertex positions along edges               |
|    - Appends triangles to AppendStructuredBuffer<Triangle>            |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 3. Mesh Assembly & Physics (CPU: Chunk.cs)                            |
|    - Reads back triangle count via ComputeBuffer.CopyCount            |
|    - Constructs Unity Mesh, recalculates normals, updates Collider    |
+-----------------------------------------------------------------------+
```

### 1. Volumetric Scalar Field (3D Density Grid)
Unlike traditional terrain systems that map `(x, z) -> height` (which can only represent a single surface height per column), this project represents space as a 3D grid of scalar density values ("weights").
- **3D-to-1D Indexing:** Located in `MetricsCompute.compute`, 3D coordinates $(x, y, z)$ within a chunk of size $N$ (`_ChunkSize`) are flattened into a contiguous 1D buffer index:
  $$\text{Index}(x, y, z) = x + N \cdot (y + N \cdot z)$$
- **Noise Evaluation:** `NoiseCompute.compute` runs on the GPU to populate the grid using 3D Ridged Fractal OpenSimplex2 noise (`FNL_NOISE_OPENSIMPLEX2`, `FNL_FRACTAL_RIDGED`), parameterized by frequency, amplitude, octaves, and scale.

### 2. The Marching Cubes Algorithm
Implemented in `MarchingCubesCompute.compute` (`March` kernel):
1. **Voxel Sampling:** Each GPU thread processes one voxel cube by reading the density weights at its **8 corner vertices** (`cubeValues[0..7]`).
2. **Bitmask Classification (`cubeIndex`):** Each corner's weight is compared against the surface threshold (`_IsoLevel = 0.5`). If `cubeValues[i] < _IsoLevel`, the $i$-th bit of `cubeIndex` is set:
   ```hlsl
   int cubeIndex = 0;
   if (cubeValues[0] < _IsoLevel) cubeIndex |= 1;   // 2^0
   if (cubeValues[1] < _IsoLevel) cubeIndex |= 2;   // 2^1
   // ... up to corner 7 (2^7 = 128)
   ```
   Since each of the 8 corners is either inside or outside the surface, there are $2^8 = 256$ possible configurations.
3. **Lookup Table (`MarchingTable.hlsl`):** Using `cubeIndex`, the shader indexes into `triTable[256][16]`, which lists the cube edges (0–11) intersected by the surface, grouped into up to 5 triangles per voxel.
4. **Linear Interpolation (`interp`):** Instead of placing vertices rigidly at edge midpoints (which produces blocky stairs), the exact intersection point $\mathbf{P}$ along an edge between corner $\mathbf{P}_1$ (weight $V_1$) and corner $\mathbf{P}_2$ (weight $V_2$) is linearly interpolated:
   $$\mathbf{P} = \mathbf{P}_1 + \frac{\text{IsoLevel} - V_1}{V_2 - V_1} (\mathbf{P}_2 - \mathbf{P}_1)$$
5. **Append Buffer Output:** Each constructed `Triangle` (3 `float3` vertices scaled to world units) is pushed atomically to `AppendStructuredBuffer<Triangle> _Triangles`.

### 3. Level of Detail (LOD) Scaling
Defined in `GridMetrics.cs`, the chunk stores its master density data at the highest resolution (`LastLod` = 80 points per axis, or $80^3 = 512,000$ sample points). When polygonizing at a lower `LOD` setting (`LODs = { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80 }`), `Chunk.cs` computes a scale factor:
$$\text{LodScaleFactor} = \frac{\text{PointsPerChunk}(\text{LastLod})}{\text{PointsPerChunk}(\text{LOD})}$$
The `March` kernel steps through the underlying density grid by `unit = 1 * _LodScaleFactor`, allowing real-time mesh simplification without regenerating or losing high-resolution terrain edits.

### 4. Real-Time Interactive Terraforming
`TerraformingCamera.cs` casts a physics ray from the camera through the mouse cursor against the chunk's `MeshCollider`:
- **Left Click (`Input.GetMouseButton(0)`):** Adds density (`+1.0`), building terrain.
- **Right Click (`Input.GetMouseButton(1)`):** Subtracts density (`-1.0`), carving caves/tunnels.
- When a hit is detected, `Chunk.EditWeights()` dispatches the `UpdateWeights` compute kernel (`MarchingCubesCompute.compute`), which checks every grid point's world-space distance to `_HitPosition` in parallel and modifies weights within `_BrushSize`, then immediately rebuilds the mesh and collider.

---

## Project Structure

```text
Assets/
├── Scenes/
│   └── SampleScene.unity                  # Main demo scene with Chunk, NoiseGenerator, and Camera
└── Scripts/
    ├── Chunk.cs                           # Manages ComputeBuffers, dispatches March/UpdateWeights, builds Mesh
    ├── GridMetrics.cs                     # Constants for thread group sizes (8), world scale (32), and LOD tiers
    ├── NoiseGenerator.cs                  # Dispatches NoiseCompute shader to generate 3D scalar weights
    ├── TerraformingCamera.cs              # Raycast controller + Gizmo visualizer for adding/removing terrain
    ├── SmoothShadingDemo.cs               # Reference implementation for vertex deduplication & smooth normals
    └── Compute/
        ├── MarchingCubesCompute.compute   # Core GPU kernels: 'March' (polygonization) & 'UpdateWeights'
        ├── NoiseCompute.compute           # GPU kernel: 'GenerateNoise' using 3D FastNoiseLite
        └── Includes/
            ├── FastNoiseLite.compute      # HLSL port of FastNoiseLite procedural noise library
            ├── MarchingTable.hlsl         # Lookup tables: edgeConnections[12][2], cornerOffsets[8], triTable[256][16]
            └── MetricsCompute.compute     # Shared GPU constants and 3D-to-1D indexFromCoord() helper
```

---

## Getting Started & Controls

### Prerequisites
- **Unity Editor:** `6000.0.22f1` (Unity 6) or compatible Unity 2022.3+ LTS with Universal Render Pipeline (URP) and Compute Shader support (DirectX 11/12, Vulkan, or Metal).

### Running the Project
1. Clone or download this repository and open the root folder in **Unity Hub**.
2. Open `Assets/Scenes/SampleScene.unity`.
3. Press **Play** in the Unity Editor.

### Interactive Controls & Inspector Parameters
- **Left Mouse Button (Hold):** Sculpt / add terrain at the cursor raycast position.
- **Right Mouse Button (Hold):** Excavate / dig caves at the cursor raycast position.
- **`Chunk` Inspector:**
  - `LOD` slider (`0` to `9`): Adjusts mesh resolution in real time (`8³` to `80³` grid subdivisions).
- **`NoiseGenerator` Inspector:**
  - `Noise Scale`, `Amplitude`, `Frequency`, `Octaves`, `Ground Percent`: Customize cave density and fractal detail.
- **`TerraformingCamera` Inspector:**
  - `Brush Size`: Radius of the spherical terraforming brush (visualized as a yellow wire sphere Gizmo in Scene view).

---

## Credits & References
- Inspired by the Marching Cubes guide by [PolyCoding](https://polycoding.net/marching-cubes/).
- Original Marching Cubes paper: Lorensen, W. E., & Cline, H. E. (1987). *Marching cubes: A high resolution 3D surface construction algorithm*. ACM SIGGRAPH Computer Graphics.
- Noise library: [FastNoiseLite](https://github.com/Auburn/FastNoiseLite) (HLSL Compute implementation).
