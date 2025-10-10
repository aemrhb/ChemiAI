# Auto-Slicing Feature for Large Mesh Inference

This feature automatically slices large meshes into smaller chunks before inference, making it possible to process huge meshes that would otherwise be too large for the model.

## Overview

When working with very large mesh files (millions of faces), processing the entire mesh at once can be:
- Memory intensive
- Slow to process
- Prone to out-of-memory errors

The auto-slicing feature solves this by:
1. Automatically dividing large meshes into smaller, manageable pieces
2. Processing each slice independently during inference
3. Removing unused vertices from each slice to optimize memory usage

## How It Works

1. **Before Inference**: Large mesh is automatically sliced into smaller parts
2. **Sliced meshes are saved** near the original mesh location for reuse
3. **Inference runs** on the smaller sliced meshes instead of the original
4. **Results are generated** for each slice

## Usage

### 1. Enable Auto-Slicing in Your Config

Add the `auto_slice` section to your inference configuration YAML file:

```yaml
auto_slice:
  enabled: true                    # Turn on auto-slicing
  mode: 'grid'                     # Slicing mode: 'grid' or 'adaptive'
  grid_divisions: [3, 3, 1]        # For grid mode: [x, y, z] divisions
  reuse_existing_slices: true      # Reuse slices if they already exist
```

### 2. Run Inference

```bash
python Infer.py --config config_inference_auto_slice_example.yaml
```

## Configuration Options

### Basic Settings

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable/disable auto-slicing | `false` |
| `mode` | Slicing mode: `'grid'` or `'adaptive'` | `'grid'` |
| `reuse_existing_slices` | Reuse existing slices instead of regenerating | `true` |
| `output_dir` | Where to save sliced meshes | `../sliced_meshes` |

### Grid Mode Settings

Used when `mode: 'grid'`:

| Option | Description | Example |
|--------|-------------|---------|
| `grid_divisions` | Number of divisions in [X, Y, Z] | `[3, 3, 1]` for 9 slices |

**Examples:**
- `[2, 2, 1]` → 4 slices (2×2×1 grid)
- `[3, 3, 1]` → 9 slices (3×3×1 grid)
- `[4, 4, 2]` → 32 slices (4×4×2 grid)
- `[5, 5, 1]` → 25 slices (5×5×1 grid)

### Adaptive Mode Settings

Used when `mode: 'adaptive'`:

| Option | Description | Default |
|--------|-------------|---------|
| `target_faces_per_box` | Target number of faces per slice | `5000` |
| `min_boxes` | Minimum number of slices | `4` |
| `max_boxes` | Maximum number of slices | `50` |

**Example:**
```yaml
auto_slice:
  enabled: true
  mode: 'adaptive'
  target_faces_per_box: 10000  # Larger slices
  min_boxes: 4
  max_boxes: 30
```

## Examples

### Example 1: Small Mesh (Fast Processing)

```yaml
auto_slice:
  enabled: true
  mode: 'grid'
  grid_divisions: [2, 2, 1]  # 4 slices
```

### Example 2: Medium Mesh (Balanced)

```yaml
auto_slice:
  enabled: true
  mode: 'grid'
  grid_divisions: [3, 3, 1]  # 9 slices
```

### Example 3: Large Mesh (Fine Granularity)

```yaml
auto_slice:
  enabled: true
  mode: 'grid'
  grid_divisions: [5, 5, 2]  # 50 slices
```

### Example 4: Adaptive Based on Complexity

```yaml
auto_slice:
  enabled: true
  mode: 'adaptive'
  target_faces_per_box: 5000
  min_boxes: 4
  max_boxes: 50
```

## Output Structure

When auto-slicing is enabled, the following happens:

```
Original mesh location:
  /path/to/meshes/
    └── LaNau-v3-DIADRASIS.ply (original large mesh)

Sliced meshes location (default):
  /path/to/sliced_meshes/
    ├── LaNau-v3-DIADRASIS_slice_000.ply
    ├── LaNau-v3-DIADRASIS_slice_001.ply
    ├── LaNau-v3-DIADRASIS_slice_002.ply
    └── ... (more slices)

Inference results:
  /path/to/output/predictions/
    └── predicted_meshes/
        ├── LaNau-v3-DIADRASIS_slice_000_predicted.ply
        ├── LaNau-v3-DIADRASIS_slice_001_predicted.ply
        └── ... (predictions for each slice)
```

## Benefits

1. **Memory Efficient**: Each slice contains only the vertices it uses
2. **Scalable**: Process meshes of any size
3. **Reusable**: Slices are saved and can be reused for multiple inference runs
4. **Automatic**: No manual bounding box specification needed
5. **Flexible**: Choose between grid-based or adaptive slicing

## Tips

1. **Start with grid mode** - Easier to configure and understand
2. **Adjust grid divisions** based on your mesh size and available memory
3. **Enable `reuse_existing_slices`** to speed up subsequent runs
4. **Use adaptive mode** for meshes with varying density
5. **Monitor memory usage** - if you get OOM errors, increase the number of divisions

## Troubleshooting

**Problem**: Out of memory during inference
- **Solution**: Increase grid divisions (e.g., from `[3,3,1]` to `[4,4,1]`)

**Problem**: Slicing is too slow
- **Solution**: Enable `reuse_existing_slices: true` to avoid re-slicing

**Problem**: Too many small files
- **Solution**: Reduce grid divisions or increase `target_faces_per_box` in adaptive mode

**Problem**: Slices have different sizes
- **Solution**: This is normal - areas with more geometry will have more faces

## Technical Details

- **Vertex Removal**: Unused vertices are automatically removed from each slice
- **No Overlap**: Slices have zero overlap by default (configurable in code)
- **Face Assignment**: Faces are assigned to slices based on whether all vertices are within the bounding box
- **File Format**: Sliced meshes maintain the same PLY format as the original

