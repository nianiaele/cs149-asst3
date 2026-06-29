# Plan: CS149 Part 3 CUDA Tiled Circle Renderer

**Goal**: Replace the incorrect circle-parallel CUDA renderer with a correct, faster tile-owned renderer that preserves per-pixel circle order and avoids image update races.
**Architecture**: Use one CUDA block per 16x16 image tile, one CUDA thread per pixel, process circles in 256-circle batches, compact tile-overlapping circles in shared memory with the provided exclusive scan, shade pixels from the compacted list in original circle order, and write each pixel once.
**Tech Stack**: CUDA C++ in `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`, shared-memory scan from `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/exclusiveScan.cu_inl`, circle-box tests from `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/circleBoxTest.cu_inl`, validation via `make`, `./render -r cuda -c <scene>`, and `./checker.py`.

## Current Status

Steps 1, 2, 3, 4, 5, and 6 are complete in `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`:

- The old circle-parallel render path was replaced with a pixel-owned baseline kernel.
- The new helper keeps the pixel color in a local register accumulator instead of repeatedly reading/writing global image memory.
- `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/CudaRenderer::render()` now launches a 16x16 pixel grid.
- `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k` passed correctness for Step 1.
- Measured Step 1 render time in the verification run: `rgb` about 0.1632 ms, `rand10k` about 18.1058 ms.
- Step 2 added `TILE_WIDTH`, `TILE_HEIGHT`, `SCAN_BLOCK_DIM`, and included `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/exclusiveScan.cu_inl` plus `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/circleBoxTest.cu_inl`.
- `make && ./render -r cuda -c rgb` passed correctness after Step 2.
- Step 3 converted the baseline kernel skeleton to explicit tile coordinates, linear scan thread index, normalized tile bounds, and `CudaRenderer::render()` now launches using `TILE_WIDTH` and `TILE_HEIGHT`.
- `make && ./render -r cuda -c rgb` passed correctness after Step 3.
- Step 4 added per-batch tile-circle flags, shared-memory exclusive scan, compacted per-tile circle IDs, and shading from the compacted list.
- `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k` passed correctness after Step 4.
- Measured Step 4 render time in the verification run: `rgb` about 0.1579 ms, `rand10k` about 0.9530 ms.
- Step 5 verified that pixels are shaded only from the compacted per-tile circle list in preserved input order.
- `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k && ./render -r cuda -c rand100k` passed correctness after Step 5.
- Measured Step 5 render time in the verification run: `rgb` about 0.2039 ms, `rand10k` about 1.1095 ms, `rand100k` about 7.6893 ms.
- Step 6 cached compacted circle positions and radii in shared memory before pixel shading, and added a radius-aware local shading helper.
- `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k && ./render -r cuda -c rand100k` passed correctness after Step 6.
- Measured Step 6 render time in the verification run: `rgb` about 0.1716 ms, `rand10k` about 0.9900 ms, `rand100k` about 7.8072 ms.

This baseline is correct but expected to be too slow on large scenes because each pixel still scans every circle.

## Execution Status

| Step | Status | Evidence |
|------|--------|----------|
| Step 1: Establish a correct pixel-owned baseline | Done | `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k` passed correctness. |
| Step 2: Add tile and scan constants | Done | Added tile/scan constants and includes; `make && ./render -r cuda -c rgb` passed correctness. |
| Step 3: Convert baseline kernel into a tiled render kernel skeleton | Done | Added explicit tile mapping, linear thread index, tile bounds, and tile constant launch; `make && ./render -r cuda -c rgb` passed correctness. |
| Step 4: Add per-batch tile-circle flags and shared-memory scan | Done | Added shared flags, scan, compacted circle IDs, and compacted-list shading; `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k` passed correctness. |
| Step 5: Shade pixels using the compacted circle list | Done | Verified compacted-list shading in preserved order; `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k && ./render -r cuda -c rand100k` passed correctness. |
| Step 6: Cache compacted circle data in shared memory | Done | Cached compacted circle positions/radii in shared memory; `make && ./render -r cuda -c rgb && ./render -r cuda -c rand10k && ./render -r cuda -c rand100k` passed correctness. |
| Step 7: Split snow and non-snow shading paths if needed | Pending | Not started. |
| Step 8: Tune tile and culling choices | Pending | Not started. |
| Step 9: End-to-end verification | Pending | Not started. |

## Task Dependencies

| Group | Steps | Can Parallelize |
|-------|-------|-----------------|
| 1 | Step 1 | No |
| 2 | Steps 2-5 | No, all touch the render kernel |
| 3 | Steps 6-8 | No, tune after correctness |
| 4 | Step 9 | No, final verification |

## Step 1: Establish a correct pixel-owned baseline

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 1a. Expected failing behavior before the change

The starter renderer is incorrect because one thread renders one circle and multiple circle threads can update the same pixel out of order.

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
./render -r cuda -c rgb
```

Expected before fix: correctness mismatch.

### 1b. Implementation

Replace the circle-owned render kernel with a pixel-owned baseline:

- one CUDA thread owns one output pixel
- each thread loops circles from `0` to `numCircles - 1`
- each thread keeps its pixel color in a local `float4`
- each thread blends circles in input order
- each thread writes its final pixel once

Add or keep a local shading helper that:

- takes `circleIndex`, normalized pixel center, circle position, and a reference to local pixel color
- performs point-in-circle test
- computes snow or normal circle color exactly like the old `shadePixel`
- updates the local pixel accumulator only
- does not read or write global image memory

### 1c. Verification

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
./render -r cuda -c rand10k
```

Expected: correctness passes. Runtime may be slow for larger scenes.

## Step 2: Add tile and scan constants

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 2a. Implementation

Near the top of the file, before including `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/exclusiveScan.cu_inl`, define:

```text
TILE_WIDTH = 16
TILE_HEIGHT = 16
SCAN_BLOCK_DIM = 256
```

Then include:

```text
/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/exclusiveScan.cu_inl
/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/circleBoxTest.cu_inl
```

Important constraints:

- `SCAN_BLOCK_DIM` must equal the number of threads in the block.
- `SCAN_BLOCK_DIM` must be a power of two.
- With 16x16 tiles, block size is exactly 256 threads.

### 2b. Verification

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
```

Expected: build succeeds and baseline correctness still passes.

## Step 3: Convert baseline kernel into a tiled render kernel skeleton

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 3a. Implementation

Change the render kernel mapping from generic pixel blocks to explicit image tiles:

- `blockIdx.x`, `blockIdx.y` identify the tile.
- `threadIdx.x`, `threadIdx.y` identify the pixel inside the tile.
- linear thread index must be computed as:

```text
linearThreadIndex = threadIdx.y * blockDim.x + threadIdx.x
```

This linear index is required by the provided shared-memory scan.

Inside the kernel:

1. Compute `pixelX` and `pixelY`.
2. Guard invalid pixels at image boundaries.
3. Compute normalized pixel center.
4. Compute tile bounds in normalized coordinates:
   - left
   - right
   - bottom
   - top
5. Keep the local pixel color initialization and final single write.

### 3b. Verification

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
```

Expected: correctness still passes.

## Step 4: Add per-batch tile-circle flags and shared-memory scan

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 4a. Implementation

Inside the tiled render kernel, allocate shared memory arrays:

```text
flag[256]
scanOutput[256]
scanScratch[512]
compactedCircleIds[256]
numCompacted
```

Replace the full per-pixel circle loop with a batch loop:

```text
for batchStart in 0..numCircles step 256
```

For each batch:

1. `circleId = batchStart + linearThreadIndex`.
2. Each thread tests one circle against this tile using `circleInBox`.
3. Store result into `flag[linearThreadIndex]`.
4. Synchronize all threads.
5. Call `sharedMemExclusiveScan` from `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/exclusiveScan.cu_inl`.
6. Synchronize all threads.
7. Compute `numCompacted = scanOutput[255] + flag[255]`.
8. If this thread's flag is 1, write `circleId` to `compactedCircleIds[scanOutput[linearThreadIndex]]`.
9. Synchronize all threads.

All 256 threads must participate in each scan. It is one logical block-wide scan per tile per circle batch.

### 4b. Verification

Initially, after building the compacted list, still shade from the compacted list. Validate:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
./render -r cuda -c rand10k
```

Expected: correctness passes.

## Step 5: Shade pixels using the compacted circle list

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 5a. Implementation

For each circle batch, after compaction:

```text
for i in 0..numCompacted - 1:
    circleId = compactedCircleIds[i]
    load circle position
    run local shade helper for this pixel
```

The loop over `i` must be sequential and increasing. Do not parallelize this inner shading loop across circles for one pixel, because alpha blending is order-dependent.

This preserves order because:

- batches are processed in increasing circle index order
- scan compaction preserves increasing circle index order within each batch

### 5b. Verification

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
./render -r cuda -c rand10k
./render -r cuda -c rand100k
```

Expected: correctness passes. Runtime should improve compared with the pixel-owned baseline.

## Step 6: Cache compacted circle data in shared memory

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 6a. Implementation

Once the compacted ID version is correct, reduce repeated global memory loads by caching candidate circle data in shared memory:

```text
compactedPosition[256]
compactedRadius[256]
```

Optional for non-snow scenes:

```text
compactedColor[256]
```

Use threads with `linearThreadIndex < numCompacted` to load compacted circle data once into shared memory. Then synchronize before pixel shading.

During pixel shading, each pixel thread reads candidate circle data from shared memory instead of reloading position/radius from global memory for every pixel.

### 6b. Verification

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
./render -r cuda -c rand10k
./render -r cuda -c rand100k
```

Expected: correctness still passes and render time improves or remains similar.

## Step 7: Split snow and non-snow shading paths if needed

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 7a. Implementation

The local shading helper currently checks scene type inside the inner loop. If profiling shows this matters, split render into two helpers or two kernels:

- non-snow shading: fixed alpha `.5f`, use circle color
- snow shading: lookup snow color ramp and compute falloff alpha

In `CudaRenderer::render()`, select the appropriate kernel based on `sceneName`.

### 7b. Verification

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./render -r cuda -c rgb
./render -r cuda -c snowsingle
```

Expected: correctness passes for both snow and non-snow paths.

## Step 8: Tune tile and culling choices

**File**: `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`

### 8a. Experiments

Try these options one at a time:

1. Keep 16x16 tile and exact `circleInBox`.
2. Try `circleInBoxConservative` if exact test overhead dominates.
3. Try tile shape 32x8 while keeping 256 threads.
4. Try tile shape 8x32 while keeping 256 threads.

Do not change multiple variables at once.

### 8b. Verification

For each variant, run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make
./checker.py
```

Record the score table and choose the fastest correct variant.

## Step 9: End-to-end verification

**Files**:

- `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/cudaRenderer.cu`
- `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/exclusiveScan.cu_inl`
- `/data/repos/fbsource/users/lm/lmxl/cs149-asst3/render/circleBoxTest.cu_inl`

### 9a. Final validation

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
make clean
make
./checker.py
```

Expected:

- all scenes pass correctness
- render scores improve over the pixel-owned baseline

### 9b. Manual spot checks

Run:

```bash
cd /data/repos/fbsource/users/lm/lmxl/cs149-asst3/render
./render -r cuda -c rgb
./render -r cuda -c pattern
./render -r cuda -c snowsingle
./render -r cuda -c biglittle
```

Expected: all correctness checks pass.

## Final Notes

- Avoid allocating a global `numTiles * numCircles` map. It can be too large for `rand1M` and `micro2M`.
- Candidate circle lists should be temporary and per block in shared memory.
- The final render kernel should not need atomics.
- The final render kernel should write each pixel exactly once.
- The core correctness invariant is per-pixel circle order, not global ordering across unrelated pixels.
