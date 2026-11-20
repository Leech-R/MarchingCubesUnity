# 60 km GPU Procedural Planet & Voxel Terraforming Engine (Unity 6 / URP)

A real-time procedural spherical planet and volumetric terrain engine built in **Unity 6 (`6000.0.22f1`, Universal Render Pipeline)** using **C#**, **HLSL Compute Shaders**, and custom **URP Shaders**. The project streams a **60 km diameter planet** (`R = 30,000 m`) in real time by combining **GPU Marching Cubes**, **Astroneer-style 3-stage octree chunk streaming**, **non-blocking `AsyncGPUReadback`**, **zero-PhysX mesh raycasting**, and **stylized low-poly biome & atmosphere shaders**.

---

## Core Features

- **60 km Spherical Procedural Terrain (`NoiseCompute.compute`, `NoiseGenerator.cs`):**
  - Evaluates a continuous 3D scalar density field over a 30 km-radius sphere (`RootSize = 65,536 m`), combining domain-warped 3D **OpenSimplex2** (or **Perlin**) continents, 2-octave LOD-invariant rolling hills, tectonic ridged mountain chains (`amplitude = 520 m`, `frequency = 0.00028`), continuous shorelines transitioning to a flat spherical sea surface (`-1.6 m` depth cap), restrained 3D cliff overhangs, and optional deep buried subsurface caverns.
- **Astroneer-Style 3-Stage Partial Loading & Progressive Octree LOD (`PlanetOctree.cs`):**
  - Dynamically streams only the visible patch of the planet (`SurfaceRenderDistance = 10 km`, `MaxDepth = 8`, `16³` voxel cells per leaf chunk) using:
    1. **Stage 1 — Spherical Shell Culling:** Skips octants strictly inside solid core or empty space (`_cachedRMin` to `_cachedRMax`) via closest/farthest AABB-to-center distance checks.
    2. **Stage 2 — Planetary Horizon Culling:** Culls chunks occluded behind the planet's curvature (`_cachedCamHorizonAngle`).
    3. **Stage 3 — 6-Plane Camera Frustum Culling & Unloading:** Unloads out-of-view chunks beyond a 360° safety bubble (`ImmediateBubbleRadius = 320 m`, `FrustumUnloadDistance = 900 m`).
  - Uses velocity look-ahead, 3-tier depth-aware split distances, `ProgressiveGateDepth = 6` to prevent speculative subdivision of empty blocks near the ground, and a `Stack<Chunk>` object pool.
- **2-Stage Non-Blocking Async GPU Readback & IVT Empty Pruning (`Chunk.cs`, `MarchingCubesCompute.compute`):**
  - Dispatches compute generation and triangle extraction across a **16-slot `AsyncGPUReadback` pipeline**.
  - **Stage 1** reads back only triangle counts plus Intermediate Value Theorem (IVT) sign flags (`AllAir` vs `AllSolid`) so completely empty chunks are pruned immediately without transferring triangle buffers; **Stage 2** uploads surface geometry within a per-frame CPU budget (`FrameBuildBudgetMs = 1.5 ms`).
- **Stylized Flat-Faceted & Toggleable Smooth-Welded Modes + Watertight Skirts (`Chunk.cs`):**
  - Defaults to crisp flat-shaded low-poly facets (zero vertex-welding or boundary-normal overhead) and can toggle live (`[F]`) to $O(1)$ quantized spatial-hash vertex welding with cross-chunk normal stitching.
  - Generates downward LOD-seam skirts (`1.50 * cellSize`) in both shading modes so chunk boundaries and LOD transitions remain watertight.
- **Zero-PhysX Interactive Terraforming & Batched GPU Stroke Replay (`TerraformingCamera.cs`, `PlanetOctree.cs`, `Chunk.cs`):**
  - Uses a bounds-sorted CPU Möller–Trumbore raycast (`RaycastTerrain` / `RaycastMesh`) directly against pre-skirt surface triangles instead of baking PhysX `MeshCollider`s while flying or aiming.
  - Coalesces strokes and indexes them with an $O(1)$ global bounding-box early-out, clamped bucket spans, and direct list fallback (`HasIntersectingStroke` / `CollectIntersectingStrokes`), caches recently sculpted chunks in an **8-slot LRU VRAM weight buffer pool** for $O(1)$ incremental sculpting (`UpdateWeights`), and replays strokes onto newly streamed chunks via batched GPU dispatches (`UpdateWeightsBatch`).
- **Stylized URP Terrain, Water & Atmospheric Skybox (`ProceduralTerrain.shader`, `SpaceAtmosphereSkybox.shader`):**
  - Shades terrain in 3 fast 3D procedural noise lookups: emerald/teal grasslands, cohesive terracotta valley soil patches, sandy beaches, glacial slate mountain rock, snow caps, turquoise-to-deep-blue spherical ocean normals with analytical waves, and subtle cyan aerial perspective.
  - Renders an orbital-to-surface skybox transitioning smoothly from a procedural starfield and solar corona in space to a vivid cyan atmosphere with drifting procedural puffy clouds and a banded lavender distant moon.

---

## Architecture & Data Pipeline

```text
+-------------------------------------------------------------------------+
| 1. 3-Stage Octree Culling & Progressive LOD (CPU: PlanetOctree.cs)      |
|    - Shell, Planetary Horizon & Frustum culling + velocity look-ahead   |
|    - ProgressiveGateDepth confirms surface at Depth 6 before splitting  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 2. LOD-Invariant Planetary Density Field (GPU: NoiseCompute.compute)    |
|    - Warped continents, 2-octave hills, tectonic ridged mountains       |
|    - Flattened spherical ocean floor & optional deep subsurface caverns |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 3. Marching Cubes & IVT Metadata (GPU: MarchingCubesCompute.compute)    |
|    - Extracts surface triangles into AppendStructuredBuffer<Triangle>   |
|    - Records AllAir / AllSolid IVT sign flag when triangle count == 0   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 4. 2-Stage Async Readback, Skirts & Raycast Prep (CPU: Chunk.cs)        |
|    - Stage 1: count + IVT early-out; Stage 2: CPU-budgeted mesh build   |
|    - Snapshots pre-skirt surface for raycasting & appends seam skirts   |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 5. Stylized Rendering & Skybox (GPU: URP Shaders)                       |
|    - ProceduralTerrain.shader: biomes, smooth water normals, cyan haze  |
|    - SpaceAtmosphereSkybox.shader: stars, sky dome, clouds & banded moon|
+-------------------------------------------------------------------------+
```

---

## Key Engineering Challenges & Optimizations Solved

1. **Eliminating Sync GPU Stalls During Large-World Exploration:** Replaced blocking `ComputeBuffer.GetData()` calls with a 16-slot two-stage `AsyncGPUReadback` pipeline and budgeted CPU mesh finalization (`1.5 ms/frame`).
2. **Preventing Exponential Empty-Chunk Splits & LOD Cracks:** Added IVT empty-state tracking (`AllAir`/`AllSolid`), `ProgressiveGateDepth`, strictly LOD-invariant noise math (removing `_CellSize`-dependent octaves), and `1.50 * cellSize` watertight skirts.
3. **Removing Close-Up Flight & Fragment Shader Bottlenecks:** Cut fragment shader noise cost from multi-octave FBM loops (~144 hashes/pixel) down to 3 single `ValueNoise3D` lookups plus ocean-gated analytical waves, capped close-up depth at `MaxDepth = 8`, defaulted to flat low-poly shading, and disabled background collider cooking (`AutoBakeNearbyColliders = false`).
4. **Fixing Post-Sculpt Movement Lag & Dig/Fill Freezes:** Fixed an $O(N^3)$ bucket-iteration bottleneck across huge octree nodes by adding an $O(1)$ global stroke AABB gate and bucket-span cap, skipped impossible `AllAir`/`AllSolid` brush rebuilds, and replaced Physics raycasts/collider baking with direct CPU Möller–Trumbore mesh raycasting.
5. **Fixing Water Crater Bowls & Coastline Artifacts:** Replaced the deep underwater elevation drop with a flat spherical water level (`-1.6 m` cap), removed a 15 m terrain discontinuity at `contNorm == 0`, and blended water shading completely to smooth radial planet normals (`upDir`).

---

## Project Structure

```text
Assets/
├── GroundMat.mat                              # Stylized terrain material preset
├── SpaceSkyboxMat.mat                         # Space-to-atmosphere skybox material preset
├── ProceduralTerrain.shader                   # URP terrain, ocean & horizon-haze shader
├── SpaceAtmosphereSkybox.shader               # Space starfield, cyan sky, clouds & moon shader
├── Scenes/
│   └── SampleScene.unity                      # Pre-configured 60 km planet scene
└── Scripts/
    ├── Chunk.cs                               # 16-slot async readback, skirts, CPU raycast, 8-slot LRU cache
    ├── PlanetOctree.cs                        # 3-stage culling, progressive LOD, stroke spatial indexing
    ├── GridMetrics.cs                         # Thread-group sizing (8³) and LOD resolution table
    ├── NoiseGenerator.cs                      # Planetary & cavern uniforms + compute dispatch
    ├── TerraformingCamera.cs                  # Smoothed mouse flight controller, HUD, sculpt input
    └── Compute/
        ├── MarchingCubesCompute.compute       # March (triangles + IVT), UpdateWeights, UpdateWeightsBatch
        ├── NoiseCompute.compute               # Spherical terrain, hills, mountains, overhangs & caves
        └── Includes/
            ├── FastNoiseLite.compute          # 3D OpenSimplex2 / Perlin noise primitives
            ├── MarchingTable.hlsl             # 256-case Marching Cubes edge & triangle LUTs
            └── MetricsCompute.compute         # Shared grid dimensions & indexing helpers
```

---

## Getting Started & Controls

### Prerequisites
- **Unity Editor:** `6000.0.22f1` (Unity 6) with Universal Render Pipeline (URP) and Compute Shader support.

### Running the Project
1. Open the repository folder in **Unity Hub** and open `Assets/Scenes/SampleScene.unity`.
2. Press **Play** in the Unity Editor (the scene comes pre-wired with `PlanetOctree`, `NoiseGenerator`, `Chunk`, `TerraformingCamera`, `GroundMat`, and `SpaceSkyboxMat`).

### Controls
| Input | Action |
| :--- | :--- |
| **`W / A / S / D`** | Fly forward / strafe / backward |
| **`E` / `Space`**, **`Q` / `Ctrl`** | Ascend / descend (scales smoothly with altitude) |
| **`Left Shift`** | Flight speed boost (`3.0x`) |
| **`Mouse` / `Arrow Keys`** | Steer camera (with altitude-aware planet horizon auto-alignment) |
| **`Tab` / `C` / `Middle Click`** | Lock cursor for flight steering (`Esc` to unlock; `Alt` or hold MMB for free look) |
| **`Scroll Wheel`** | Adjust flight throttle (`0.2x`–`6.0x`; hold `Left Alt + Scroll` to resize brush) |
| **`Left Mouse` (Hold)** | Add volumetric terrain (fill/build) |
| **`Right Mouse` (Hold)** | Carve volumetric terrain (dig) |
| **`[` / `]`** | Decrease / increase brush radius (`4 m`–`300 m`) |
| **`-` / `=`** | Decrease / increase surface render distance (`2 km`–`20 km`) |
| **`F`** | Toggle Flat Faceted vs Smooth Welded shading |
| **`G`** | Toggle Octree chunk bounds wireframe |
| **`K`** | Toggle Horizon & Frustum partial culling |
| **`L`** | Freeze / resume Octree streaming |
| **`N`** | Switch noise generator (`OpenSimplex2` / `Perlin`) |
| **`R`** | Clear all sculpting edits and rebuild planet |
| **`H`** | Show / hide live telemetry HUD & crosshair |
