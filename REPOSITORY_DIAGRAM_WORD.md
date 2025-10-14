# CHemiAI Repository Architecture Diagram

**A PyTorch-based deep learning system for mesh analysis with texture-geometry integration and automatic mesh slicing**

---

## 1. Complete Directory Structure

```
CHemiAI/
|
+-- Project Configuration
|   +-- requirements.txt                          # Python dependencies
|   +-- config_inference_auto_slice_example.yaml  # Auto-slice inference config
|   +-- config_inference_with_slicing.yaml        # Standard slicing config
|   +-- AUTO_SLICE_README.md                      # Auto-slicing documentation
|
+-- Entry Points
|   +-- train.py                                  # Main training script
|   |   +-- EMA (Exponential Moving Average)
|   |   +-- Mixed Precision Training (AMP)
|   |   +-- Checkpoint Management
|   |   +-- Metrics: IoU, F1, Accuracy
|   |
|   +-- Infer.py                                  # Inference & auto-slicing
|       +-- Auto-slicing (grid/adaptive modes)
|       +-- Batch prediction
|       +-- Result aggregation
|
+-- Core Models
|   +-- model_G_2.py                              # Transformer-based nomeformer
|   |   +-- MultiHeadAttention                    # Self-attention with positional encoding
|   |   +-- TransformerBlock                      # Attention + FFN with residual
|   |   +-- VertexEmbedding                       # Embed vertices (XYZ + normals)
|   |   +-- FaceEmbedding                         # Aggregate vertices to faces
|   |   +-- nomeformer                            # Main transformer architecture
|   |
|   +-- integrated_texture_geometry_model.py      # Texture + Geometry fusion
|       +-- TexturePatchEmbedding                 # Process RGB pixel sequences
|       +-- IntegratedTextureGeometryModel        # Multi-modal fusion
|       |   +-- Geometry Branch (vertices/faces)
|       |   +-- Texture Branch (RGB pixels)
|       |   +-- Cross-Modal Attention
|       +-- IntegratedDownstreamClassifier        # Classification head
|
+-- Data Processing
|   +-- mesh_dataset_2.py                         # Geometry dataset
|   |   +-- MeshDataset                           # Load .ply meshes
|   |   +-- MeshAugmentation                      # Rotation, scaling, jittering
|   |   +-- compute_cluster_centroids             # Spatial clustering
|   |   +-- reorder_clusters_by_proximity         # MST-based ordering
|   |   +-- custom_collate_fn                     # Batch collation
|   |
|   +-- mesh_texture_dataset.py                   # Texture dataset
|       +-- MeshTextureDataset                    # Load meshes + texture pixels
|       +-- texture_custom_collate_fn             # Texture batch collation
|       +-- Inherits augmentation from mesh_dataset_2
|
+-- Utilities & Tools
|   +-- tools/
|       +-- __init__.py
|       +-- downst.py                             # Downstream classifier
|       |   +-- DownstreamClassifier              # MLP classifier head
|       |
|       +-- helper.py                             # Training helpers
|       |   +-- init_opt()                        # Optimizer initialization
|       |
|       +-- check_point.py                        # Model persistence
|       |   +-- save_checkpoint()                 # Save model state
|       |   +-- load_checkpoint()                 # Load model state
|       |
|       +-- auto_mesh_slicer.py                   # Mesh slicing system
|           +-- parse_ply_file()                  # Read PLY format
|           +-- generate_automatic_bounding_boxes() # Grid-based slicing
|           +-- generate_adaptive_bounding_boxes() # Density-based slicing
|           +-- slice_mesh_with_bounding_boxes()  # Extract mesh slices
|           +-- write_ply_file()                  # Write PLY format
|
+-- Source Components
|   +-- src/
|       +-- train.py                              # Training loop utilities
|       +-- transforms.py                         # Data transformations
|       +-- utils/
|           +-- distributed.py                    # Multi-GPU training
|           +-- logging.py                        # Training logs
|           +-- schedulers.py                     # LR schedulers
|           +-- tensors.py                        # Tensor utilities
|
+-- Loss Functions
|   +-- loss.py
|       +-- MaskedCrossEntropyLoss                # Handles variable-length sequences
|
+-- Cache & Build
    +-- __pycache__/                              # Python bytecode cache
```

---

## 2. Module Dependencies and Relationships

### 2.1 Entry Points

**TRAINING PIPELINE (train.py)**
- Training Pipeline with EMA Support
- Mixed Precision Training
- Checkpoint Management
- Connects to:
  * mesh_dataset_2.py (loads geometry data)
  * mesh_texture_dataset.py (loads texture data)
  * model_G_2.py (initializes models)
  * integrated_texture_geometry_model.py (initializes integrated models)
  * tools/downst.py (downstream classifier)
  * loss.py (computes loss)
  * tools/helper.py (optimizer initialization)
  * tools/check_point.py (saves/loads checkpoints)
  * src/utils/ (various utilities)

**INFERENCE PIPELINE (Infer.py)**
- Inference Pipeline with Auto-Slicing
- Batch Processing
- Metrics Computation
- Connects to:
  * tools/auto_mesh_slicer.py (preprocesses large meshes)
  * mesh_dataset_2.py (loads geometry data)
  * mesh_texture_dataset.py (loads texture data)
  * model_G_2.py (runs inference)
  * integrated_texture_geometry_model.py (runs inference)
  * tools/downst.py (downstream classifier)
  * tools/check_point.py (loads checkpoints)

### 2.2 Data Loading Layer

**Geometry Dataset (mesh_dataset_2.py)**
- Loads .ply mesh files
- Provides MeshDataset class
- Provides MeshAugmentation class
- Clustering and spatial ordering
- Custom collation functions

**Texture Dataset (mesh_texture_dataset.py)**
- Loads texture pixel data
- Provides MeshTextureDataset class
- Inherits from mesh_dataset_2.py
- Custom texture collation functions

### 2.3 Model Architecture Layer

**Core Model (model_G_2.py)**
- nomeformer: Main transformer architecture
- MultiHeadAttention: Self-attention mechanism
- TransformerBlock: Attention + Feed-forward
- Used by: train.py, Infer.py, integrated_texture_geometry_model.py

**Integrated Model (integrated_texture_geometry_model.py)**
- Extends model_G_2.py
- TexturePatchEmbedding: Process RGB pixels
- IntegratedTextureGeometryModel: Fusion model
- Cross-modal attention between geometry and texture

**Downstream Classifier (tools/downst.py)**
- Classification head
- Uses features from model_G_2.py
- MLP layers for predictions

### 2.4 Training Components

**Loss Function (loss.py)**
- MaskedCrossEntropyLoss
- Handles variable-length sequences
- Used by train.py

**Helper Utilities (tools/helper.py)**
- Optimizer initialization
- Training setup functions

**Checkpoint Manager (tools/check_point.py)**
- Save model states
- Load model states
- Used by train.py and Infer.py

### 2.5 Inference Components

**Auto Mesh Slicer (tools/auto_mesh_slicer.py)**
- Automatic mesh slicing
- Grid-based mode
- Adaptive density-based mode
- Bounding box generation
- Used by Infer.py

---

## 3. Complete Data Flow Pipeline

### 3.1 Input Phase

**INPUT DATA**
1. Mesh Files (.ply format)
   - Vertices (XYZ coordinates)
   - Faces (vertex indices)
   - Labels (per-face classifications)

2. Texture Data
   - RGB pixel sequences
   - Mapped to mesh faces

3. Configuration Files (YAML)
   - Model parameters
   - Training settings
   - Auto-slice configuration

### 3.2 Preprocessing Phase

**PREPROCESSING STEPS**

Step 1: Auto-Slicing (Optional)
- Check if auto-slice is enabled
- If enabled and mode is "grid":
  * Generate fixed spatial divisions (e.g., 3x3x1)
  * Create 9 bounding boxes
- If enabled and mode is "adaptive":
  * Calculate face density
  * Generate variable-sized boxes
  * Target specific faces per box (e.g., 5000)
- If disabled: proceed to augmentation

Step 2: Data Augmentation
- Random rotation
- Random scaling
- Vertex jittering
- Noise addition

Step 3: Spatial Clustering
- Apply KMeans clustering
- Compute cluster centroids
- Reorder clusters using MST
- Organize faces spatially

### 3.3 Model Processing Phase

**MODEL ARCHITECTURE PROCESSING**

GEOMETRY BRANCH:
1. Vertex Embedding
   - Input: [Batch, Clusters, Faces, Vertices, 6]
   - 6 channels: XYZ + Normals (Nx, Ny, Nz)
   - Output: [Batch, Clusters, Faces, Vertices, embed_dim]

2. Face-Level Attention
   - Aggregate vertices per face
   - Multi-head self-attention
   - Output: [Batch, Clusters, Faces, embed_dim]

3. Cluster-Level Attention
   - Process faces within each cluster
   - Local spatial relationships
   - Output: [Batch, Clusters, Faces, embed_dim]

4. Global Attention
   - Cross-cluster relationships
   - Learn global structure
   - Output: [Batch, Clusters, Faces, embed_dim]

TEXTURE BRANCH (if using integrated model):
1. Pixel Projection
   - Input: [Batch, Clusters, Faces, Pixels, 3]
   - 3 channels: RGB
   - Output: [Batch, Clusters, Faces, Pixels, embed_dim]

2. Pixel-Level Attention
   - Local attention within pixel sequences
   - Summarize to CLS token
   - Output: [Batch, Clusters, Faces, embed_dim]

FUSION (if using integrated model):
3. Cross-Modal Fusion
   - Geometry-to-Texture attention
   - Texture-to-Geometry attention
   - Feature concatenation
   - Output: [Batch, Clusters, Faces, 2*embed_dim]

4. Projection
   - Reduce back to embed_dim
   - Output: [Batch, Clusters, Faces, embed_dim]

CLASSIFICATION:
5. Downstream Classifier
   - MLP layers
   - Per-face classification
   - Output: [Batch, Clusters, Faces, num_classes]

### 3.4 Output Phase

**OUTPUT RESULTS**

1. Predictions
   - Face labels
   - Per-class probabilities
   - Confidence scores

2. Metrics
   - IoU (Intersection over Union)
   - F1 Score
   - Accuracy
   - Confusion Matrix

3. Visualizations
   - Labeled meshes
   - Saved as .ply files
   - Color-coded by class

4. Checkpoints (during training)
   - Model state saved as .pth files
   - Optimizer state
   - Training statistics

---

## 4. Model Architecture Details

### 4.1 Geometry-Only Model (nomeformer)

**ARCHITECTURE FLOW**

```
Input: Mesh with N faces
|
V
[LAYER 1: Vertex Embedding]
- Input: [B, C, F, V, 6] where 6 = XYZ + Normals
- Process: Linear projection to embed_dim
- Output: [B, C, F, V, embed_dim]
|
V
[LAYER 2: Face-Level Transformer]
- Multi-Head Self-Attention
- Aggregate V vertices into 1 face representation
- Uses CLS token or average pooling
- Output: [B, C, F, embed_dim]
|
V
[LAYER 3: Cluster-Level Transformer]
- Process F faces within each cluster
- Learn local spatial relationships
- Multiple transformer blocks
- Output: [B, C, F, embed_dim]
|
V
[LAYER 4: Global Transformer]
- Cross-cluster attention
- Learn global mesh structure
- Multiple transformer blocks
- Output: [B, C, F, embed_dim]
|
V
[LAYER 5: Downstream Classifier]
- MLP layers (Linear + ReLU + Dropout)
- Per-face classification
- Output: [B, C, F, num_classes]
```

**KEY COMPONENTS**

- B = Batch size
- C = Number of clusters
- F = Number of faces per cluster
- V = Number of vertices per face (typically 3 for triangular meshes)
- embed_dim = Embedding dimension (e.g., 128)
- num_classes = Number of classification categories

### 4.2 Integrated Texture-Geometry Model

**TWO-BRANCH ARCHITECTURE**

```
Input Mesh
|
+----------------+----------------+
|                                 |
V                                 V
[GEOMETRY BRANCH]                 [TEXTURE BRANCH]
                                  
Vertices (XYZ + Normals)          RGB Pixels per Face
[B, C, F, V, 6]                   [B, C, F, P, 3]
|                                 |
V                                 V
Vertex Embedding                  Pixel Projection
[B, C, F, V, embed_dim]           [B, C, F, P, embed_dim]
|                                 |
V                                 V
Face-Level Attention              Local Attention
(aggregate vertices)              (extract CLS token)
[B, C, F, embed_dim]              [B, C, F, embed_dim]
|                                 |
+----------------+----------------+
                 |
                 V
        [CROSS-MODAL FUSION]
        
        Geometry -> Texture Attention
        Texture -> Geometry Attention
        Concatenate Features
        [B, C, F, 2*embed_dim]
                 |
                 V
        [Projection Layer]
        Reduce to embed_dim
        [B, C, F, embed_dim]
                 |
                 V
        [Cluster Transformer]
        Process within clusters
                 |
                 V
        [Global Transformer]
        Cross-cluster processing
                 |
                 V
        [Integrated Classifier]
        Final predictions
                 |
                 V
        Output: [B, C, F, num_classes]
```

**KEY FEATURES**

- P = Number of pixels per face
- Two parallel processing branches
- Cross-modal attention for feature fusion
- Shared downstream transformers
- End-to-end trainable

---

## 5. Auto-Slicing System Workflow

**STEP-BY-STEP PROCESS**

### Step 1: Input Large Mesh
- Example: 2 million faces
- May cause out-of-memory errors

### Step 2: Check Reuse Setting
- If reuse_existing_slices = True AND slices exist:
  * Load existing slice files
  * Skip to Step 7
- Otherwise:
  * Continue to Step 3

### Step 3: Parse PLY File
- Read vertices and faces
- Extract geometry data
- Prepare for slicing

### Step 4: Choose Slicing Mode

**OPTION A: Grid Mode**
- Fixed spatial divisions
- Example: grid_divisions = [3, 3, 1]
- Creates 3x3x1 = 9 bounding boxes
- Uniform distribution

**OPTION B: Adaptive Mode**
- Density-based slicing
- Parameters:
  * target_faces_per_box = 5000
  * min_boxes = 4
  * max_boxes = 50
- Variable box sizes based on face density
- More boxes in dense regions

### Step 5: Slice Mesh
- Assign each face to a bounding box
- Extract vertices for each slice
- Remove unused vertices
- Optimize memory usage

### Step 6: Write PLY Files
- Save each slice as separate .ply file
- Naming: original_name_slice_000.ply, _slice_001.ply, etc.
- Store in sliced_meshes directory
- Can be reused for future inference

### Step 7: Run Inference
- Load each slice independently
- Process through model
- Generate predictions per slice

### Step 8: Aggregate Results
- Combine predictions from all slices
- Map back to original mesh structure
- Generate final output

### Step 9: Output Results
- Save labeled mesh slices
- Save aggregated predictions
- Generate metrics

**BENEFITS OF AUTO-SLICING**

1. Memory Efficiency: Process large meshes without OOM errors
2. Scalability: Handle meshes of any size
3. Reusability: Save slices for multiple inference runs
4. Flexibility: Choose between grid and adaptive modes
5. Automatic: No manual intervention required

---

## 6. Dependencies and Installation

### 6.1 Required Packages (requirements.txt)

**Deep Learning Framework**
- torch >= 2.0.0 (Core deep learning framework)
- torchvision >= 0.15.0 (Vision utilities)
- tensorboard >= 2.13.0 (Training visualization)

**3D Mesh Processing**
- trimesh >= 3.23.0 (3D mesh manipulation and I/O)
- networkx >= 2.8.0 (Graph operations, required by trimesh)

**Numerical Computing**
- numpy >= 1.24.0 (Array operations)
- scipy >= 1.10.0 (Scientific computing, KDTree, MST)

**Machine Learning**
- scikit-learn >= 1.3.0 (KMeans clustering, metrics)

**Metrics and Evaluation**
- pytorch-ignite >= 0.4.11 (IoU, ConfusionMatrix)
- torchmetrics >= 1.0.0 (F1 Score, Accuracy)

**Configuration and Utilities**
- pyyaml >= 6.0 (YAML configuration parsing)
- tqdm >= 4.65.0 (Progress bars)
- Pillow >= 10.0.0 (Image handling)
- glob2 >= 0.7 (File pattern matching)

### 6.2 Installation Instructions

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Using conda**
```bash
conda install --file requirements.txt
```

**Option 3: Install specific packages**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install trimesh>=3.23.0 numpy>=1.24.0 scipy>=1.10.0
pip install scikit-learn>=1.3.0 pytorch-ignite>=0.4.11
pip install torchmetrics>=1.0.0 pyyaml>=6.0 tqdm>=4.65.0
```

---

## 7. Key Features and Capabilities

### 7.1 Auto-Slicing System

| Feature | Description | Details |
|---------|-------------|---------|
| **Problem Solved** | Large meshes cause OOM errors | Meshes with > 1M faces exceed GPU memory |
| **Solution** | Automatic spatial decomposition | Splits mesh into manageable chunks |
| **Grid Mode** | Fixed spatial divisions | Example: 3x3x1 creates 9 equal-sized regions |
| **Adaptive Mode** | Density-based slicing | Variable box sizes based on face count |
| **Reusability** | Save and reuse slices | Slices saved for multiple inference runs |
| **Vertex Cleanup** | Remove unused vertices | Reduces memory footprint per slice |
| **Configuration** | YAML-based setup | Easy to enable and configure |

### 7.2 Texture-Geometry Integration

| Component | Function | Benefit |
|-----------|----------|---------|
| **Dual Processing** | Separate branches for geometry and texture | Specialized processing for each modality |
| **Pixel Embeddings** | Convert RGB sequences to embeddings | Rich texture representation |
| **Cross-Modal Attention** | Geometry-Texture feature fusion | Learn joint representations |
| **CLS Token Pooling** | Summarize pixel sequences | Fixed-size representation per face |
| **End-to-End** | Joint training | Optimized for combined features |

### 7.3 Training Infrastructure

| Feature | Benefit | Implementation |
|---------|---------|----------------|
| **EMA** | Training stability | Exponential Moving Average of parameters |
| **Mixed Precision** | Faster training | Automatic Mixed Precision (AMP) |
| **Checkpointing** | Resume capability | Save/load model states |
| **Multi-GPU** | Distributed training | DistributedDataParallel support |
| **Augmentation** | Robustness | Rotation, scaling, jittering |
| **Gradient Scaling** | Numerical stability | GradScaler for mixed precision |

### 7.4 Evaluation Metrics

| Metric | Purpose | Use Case |
|--------|---------|----------|
| **IoU** | Intersection over Union | Segmentation quality |
| **F1 Score** | Harmonic mean of precision/recall | Balanced classification metric |
| **Accuracy** | Overall correctness | General performance |
| **Confusion Matrix** | Per-class analysis | Detailed error analysis |

---

## 8. Quick Start Guide

### 8.1 Installation

**Step 1: Navigate to project directory**
```bash
cd CHemiAI
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

### 8.2 Training

**Geometry-Only Model**
```bash
python train.py --config config_train_geometry.yaml
```

**Integrated Texture-Geometry Model**
```bash
python train.py --config config_train_integrated.yaml
```

### 8.3 Inference

**Standard Inference**
```bash
python Infer.py --config config_inference.yaml
```

**Inference with Auto-Slicing (for large meshes)**
```bash
python Infer.py --config config_inference_auto_slice_example.yaml
```

---

## 9. Configuration Examples

### 9.1 Training Configuration (YAML)

```yaml
model:
  type: 'integrated'              # Options: 'geometry_only', 'integrated'
  embedding_dim: 128
  num_heads: 4
  num_layers: 6
  dropout: 0.1
  num_classes: 10

training:
  batch_size: 8
  learning_rate: 0.001
  epochs: 100
  use_ema: true
  ema_decay: 0.9999
  mixed_precision: true

data:
  mesh_dir: '/path/to/meshes'
  texture_dir: '/path/to/textures'
  num_workers: 4
  max_faces_per_cluster: 512
  num_clusters: 16

optimizer:
  type: 'AdamW'
  weight_decay: 0.01

scheduler:
  type: 'CosineAnnealingLR'
  T_max: 100
```

### 9.2 Inference Configuration with Auto-Slicing (YAML)

```yaml
inference:
  checkpoint: 'checkpoints/best_model.pth'
  output_dir: 'predictions/'
  batch_size: 4
  save_visualizations: true

auto_slice:
  enabled: true
  mode: 'grid'                    # Options: 'grid', 'adaptive'
  grid_divisions: [3, 3, 1]       # Creates 9 slices
  reuse_existing_slices: true
  output_dir: 'sliced_meshes/'

# Optional: Adaptive mode parameters (if mode='adaptive')
# auto_slice:
#   enabled: true
#   mode: 'adaptive'
#   target_faces_per_box: 5000
#   min_boxes: 4
#   max_boxes: 50
#   reuse_existing_slices: true
#   output_dir: 'sliced_meshes/'
```

### 9.3 Grid Divisions Examples

| Configuration | Slices Created | Use Case |
|---------------|----------------|----------|
| [2, 2, 1] | 4 slices | Small meshes, fast processing |
| [3, 3, 1] | 9 slices | Medium meshes, balanced |
| [4, 4, 1] | 16 slices | Large meshes |
| [5, 5, 1] | 25 slices | Very large meshes |
| [4, 4, 2] | 32 slices | 3D dense meshes |
| [5, 5, 2] | 50 slices | Maximum granularity |

---

## 10. Performance Considerations

### 10.1 Memory Optimization Techniques

| Technique | Benefit | When to Use |
|-----------|---------|-------------|
| **Auto-Slicing** | Reduce memory per forward pass | Meshes > 100K faces |
| **Vertex Removal** | Eliminate unused vertices | All sliced meshes |
| **Gradient Checkpointing** | Trade compute for memory | Very deep models |
| **Mixed Precision** | ~50% memory reduction | All modern GPUs (>= V100) |
| **Batch Size Tuning** | Fit more in memory | Always optimize |

### 10.2 Speed Optimization Techniques

| Technique | Speedup | Implementation |
|-----------|---------|----------------|
| **Batch Processing** | 2-4x | Process multiple meshes simultaneously |
| **Reuse Slices** | 5-10x on re-run | Set reuse_existing_slices: true |
| **Multi-GPU** | Linear with GPUs | Use DistributedDataParallel |
| **AMP** | 1.5-2x | Enable mixed_precision: true |
| **DataLoader Workers** | 2-3x | Set num_workers > 0 |

### 10.3 Recommended Hardware

**Minimum Requirements**
- GPU: NVIDIA GTX 1080 Ti (11GB VRAM)
- RAM: 16GB
- Storage: 50GB free space

**Recommended Configuration**
- GPU: NVIDIA RTX 3090 or A100 (24GB+ VRAM)
- RAM: 32GB
- Storage: 100GB SSD

**Large-Scale Training**
- GPU: Multiple NVIDIA A100 (40GB or 80GB)
- RAM: 64GB+
- Storage: 500GB+ NVMe SSD

---

## 11. Data Format Specifications

### 11.1 Input Mesh Format (.ply)

**PLY File Structure**
```
ply
format ascii 1.0
comment Created by CHemiAI
element vertex N
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face M
property list uchar int vertex_indices
property int label
end_header
[vertex data]
[face data]
```

**Vertex Data** (N rows)
- x, y, z: 3D coordinates
- nx, ny, nz: Normal vectors

**Face Data** (M rows)
- vertex_indices: List of vertex IDs (typically 3 for triangles)
- label: Classification label for the face

### 11.2 Texture Data Format (JSON)

```json
{
  "mesh_name": "example_mesh.ply",
  "faces": [
    {
      "face_id": 0,
      "pixels": [
        [255, 128, 64],
        [240, 120, 60],
        ...
      ],
      "num_pixels": 256
    },
    {
      "face_id": 1,
      "pixels": [...],
      "num_pixels": 256
    }
  ]
}
```

**Field Descriptions**
- face_id: Face index in mesh
- pixels: List of [R, G, B] values
- num_pixels: Number of pixels sampled for this face

---

## 12. Project Statistics

### 12.1 Codebase Metrics

| Metric | Count |
|--------|-------|
| Total Python Files | 16 |
| Core Model Files | 2 |
| Dataset Classes | 2 |
| Utility Modules | 7 |
| Configuration Files | 2 |
| Documentation Files | 3 |
| Total Lines of Code | ~5,000+ |

### 12.2 Model Statistics

| Component | Parameters |
|-----------|------------|
| Vertex Embedding | ~50K |
| Multi-Head Attention (per layer) | ~200K |
| Transformer Block (per layer) | ~400K |
| Total (6 layers, embed_dim=128) | ~2.5M |
| Downstream Classifier | ~100K |
| **Total Model Size** | **~2.6M parameters** |

---

## 13. Technology Stack Summary

### 13.1 Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Deep Learning | PyTorch | 2.0+ | Model architecture & training |
| 3D Processing | Trimesh | 3.23+ | Mesh manipulation & I/O |
| Numerical | NumPy | 1.24+ | Array operations |
| Scientific | SciPy | 1.10+ | Spatial algorithms (KDTree, MST) |
| Clustering | scikit-learn | 1.3+ | KMeans spatial partitioning |
| Metrics | PyTorch-Ignite | 0.4.11+ | IoU, ConfusionMatrix |
| Metrics | TorchMetrics | 1.0+ | F1 Score, Accuracy |
| Visualization | TensorBoard | 2.13+ | Training monitoring |
| Configuration | PyYAML | 6.0+ | Experiment configuration |
| Utilities | tqdm | 4.65+ | Progress bars |

### 13.2 Architecture Patterns

**Design Patterns Used**
- Hierarchical Attention (Face -> Cluster -> Global)
- Cross-Modal Fusion (Geometry + Texture)
- Transformer Architecture
- Residual Connections
- Layer Normalization

**Best Practices Implemented**
- Mixed Precision Training
- Exponential Moving Average
- Gradient Clipping
- Learning Rate Scheduling
- Checkpoint Management
- Distributed Training Support

---

## 14. Architecture Highlights

### 14.1 Novel Components

**1. Hierarchical Attention Mechanism**
- Face-level: Aggregate vertices to face representation
- Cluster-level: Process faces within spatial clusters
- Global-level: Learn cross-cluster relationships
- Benefit: Efficient processing of large meshes with spatial coherence

**2. Cross-Modal Fusion**
- Separate processing branches for geometry and texture
- Bidirectional cross-attention
- Late fusion strategy
- Benefit: Leverage both geometric and visual information

**3. Auto-Slicing System**
- Grid-based spatial decomposition
- Adaptive density-based slicing
- Automatic vertex cleanup
- Benefit: Handle arbitrarily large meshes without memory issues

**4. Spatial Clustering with MST**
- KMeans for initial clustering
- Minimum Spanning Tree for ordering
- Ensures spatial coherence in clusters
- Benefit: Better attention patterns and performance

### 14.2 Research Features

**Training Innovations**
- Exponential Moving Average (EMA) for parameter stability
- Mixed Precision Training for efficiency
- Masked loss for variable-length sequences
- Gradient scaling for numerical stability

**Attention Mechanisms**
- Relative positional encoding option
- CLS token and average pooling options
- Multi-scale attention (face, cluster, global)
- Cross-modal attention for fusion

---

## 15. Troubleshooting Guide

### 15.1 Common Issues and Solutions

**Issue: Out of Memory (OOM) Error**
- Solution 1: Enable auto-slicing with smaller grid divisions
- Solution 2: Reduce batch size
- Solution 3: Use mixed precision training
- Solution 4: Reduce embedding dimension or number of layers

**Issue: Slow Training**
- Solution 1: Enable mixed precision (mixed_precision: true)
- Solution 2: Increase num_workers in DataLoader
- Solution 3: Use multiple GPUs with DistributedDataParallel
- Solution 4: Reduce number of clusters or faces per cluster

**Issue: Poor Model Performance**
- Solution 1: Increase training epochs
- Solution 2: Add more data augmentation
- Solution 3: Tune learning rate
- Solution 4: Use EMA (use_ema: true)
- Solution 5: Check data quality and labels

**Issue: Auto-Slicing Too Slow**
- Solution 1: Enable reuse_existing_slices: true
- Solution 2: Increase target_faces_per_box (adaptive mode)
- Solution 3: Reduce grid divisions (grid mode)

**Issue: Model Not Loading**
- Solution 1: Check checkpoint path in config
- Solution 2: Ensure model architecture matches checkpoint
- Solution 3: Check CUDA availability and device compatibility

---

## 16. Future Extensions

### 16.1 Potential Enhancements

**Model Improvements**
- Add graph neural network layers
- Implement attention mechanisms from recent papers
- Add multi-scale feature fusion
- Explore different positional encodings

**Data Processing**
- Support for more mesh formats (OBJ, STL, OFF)
- Real-time texture extraction from images
- Advanced augmentation techniques
- Dynamic clustering based on mesh complexity

**Inference Optimization**
- Model quantization for faster inference
- ONNX export for deployment
- TensorRT optimization
- Batch inference optimization

**Usability**
- Web interface for inference
- Docker container for easy deployment
- Pre-trained model zoo
- Visualization tools

---

## 17. Additional Resources

### 17.1 Documentation Files

- **AUTO_SLICE_README.md**: Detailed auto-slicing documentation
- **config_inference_auto_slice_example.yaml**: Example configuration
- **requirements.txt**: Complete dependency list

### 17.2 Key File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| train.py | Training pipeline | EMA, mixed precision, checkpointing |
| Infer.py | Inference pipeline | Auto-slicing, batch prediction |
| model_G_2.py | Core model | nomeformer, MultiHeadAttention |
| integrated_texture_geometry_model.py | Fusion model | Cross-modal attention |
| mesh_dataset_2.py | Data loading | MeshDataset, augmentation |
| tools/auto_mesh_slicer.py | Mesh slicing | Grid/adaptive slicing |

---

## 18. Contact and Support

**Project Information**
- Repository: CHemiAI
- Purpose: Mesh analysis with texture-geometry integration
- Framework: PyTorch
- License: (Specify your license)

**Getting Help**
- Check documentation files in the repository
- Review configuration examples
- Consult troubleshooting guide (Section 15)
- Check GitHub issues (if applicable)

---

**Document Information**
- Generated: October 12, 2025
- Version: 1.0
- Format: Microsoft Word Compatible
- Repository: CHemiAI - Mesh Analysis Framework

---

END OF DOCUMENT

