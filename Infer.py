import argparse
import os
import yaml
import torch
import numpy as np
import trimesh
import glob
from torch.utils.data import DataLoader
from ignite.metrics import IoU, ConfusionMatrix
from torchmetrics import F1Score, Accuracy

from mesh_dataset_2 import MeshDataset, custom_collate_fn
from mesh_texture_dataset import MeshTextureDataset, texture_custom_collate_fn
from model_G_2 import nomeformer
from tools.downst import DownstreamClassifier
from integrated_texture_geometry_model import IntegratedTextureGeometryModel, IntegratedDownstreamClassifier
from tools.auto_mesh_slicer import (
    parse_ply_file, 
    generate_automatic_bounding_boxes,
    generate_adaptive_bounding_boxes,
    slice_mesh_with_bounding_boxes,
    write_ply_file
)


def slice_large_mesh(mesh_path, output_dir, grid_divisions=(3, 3, 1), slicing_mode='grid', 
                     target_faces_per_box=5000, min_boxes=4, max_boxes=50):
    """
    Slice a large mesh into smaller parts for efficient inference.
    
    Args:
        mesh_path: Path to the original large mesh
        output_dir: Directory to save sliced meshes
        grid_divisions: Tuple of (x, y, z) divisions for grid-based slicing
        slicing_mode: 'grid' or 'adaptive'
        target_faces_per_box: Target faces per box for adaptive mode
        min_boxes: Minimum boxes for adaptive mode
        max_boxes: Maximum boxes for adaptive mode
        
    Returns:
        List of paths to sliced mesh files
    """
    print(f"\n{'='*60}")
    print(f"Slicing large mesh: {os.path.basename(mesh_path)}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the mesh
    print("Loading mesh...")
    vertices, faces = parse_ply_file(mesh_path)
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")
    
    # Generate bounding boxes
    if slicing_mode == 'adaptive':
        print(f"\nGenerating adaptive bounding boxes (target: {target_faces_per_box} faces/box)...")
        bounding_boxes = generate_adaptive_bounding_boxes(
            vertices, faces, target_faces_per_box, min_boxes, max_boxes
        )
    else:  # grid mode
        print(f"\nGenerating grid-based bounding boxes {grid_divisions}...")
        bounding_boxes = generate_automatic_bounding_boxes(
            vertices, grid_divisions=grid_divisions, overlap=0.0
        )
    
    print(f"Generated {len(bounding_boxes)} bounding boxes")
    
    # Slice the mesh
    print("\nSlicing mesh...")
    split_meshes = slice_mesh_with_bounding_boxes(vertices, faces, bounding_boxes)
    print(f"Created {len(split_meshes)} mesh slices")
    
    # Write sliced meshes
    print("\nWriting sliced meshes...")
    sliced_mesh_paths = []
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    
    for i, (slice_vertices, slice_faces, bbox) in enumerate(split_meshes):
        output_path = os.path.join(output_dir, f"{base_name}_slice_{i:03d}.ply")
        write_ply_file(output_path, slice_vertices, slice_faces)
        sliced_mesh_paths.append(output_path)
        
        reduction_percent = 100.0 * (1 - len(slice_vertices) / len(vertices))
        print(f"  Slice {i:03d}: {len(slice_faces)} faces, {len(slice_vertices)} vertices "
              f"(reduced by {reduction_percent:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"Slicing complete! Created {len(sliced_mesh_paths)} slices")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return sliced_mesh_paths


def process_mesh_directory_with_slicing(mesh_dir, slice_output_dir, auto_slice_config):
    """
    Process all meshes in a directory, slicing if needed.
    
    Args:
        mesh_dir: Directory containing original meshes
        slice_output_dir: Directory to save sliced meshes
        auto_slice_config: Configuration dict for slicing
        
    Returns:
        Directory path containing meshes to use for inference
    """
    # Check if we should slice the meshes
    should_slice = auto_slice_config.get('enabled', False)
    
    if not should_slice:
        print("Auto-slicing disabled. Using original meshes.")
        return mesh_dir
    
    # Check if sliced meshes already exist (check for .ply files specifically)
    existing_slices = glob.glob(os.path.join(slice_output_dir, "*.ply")) if os.path.exists(slice_output_dir) else []
    if len(existing_slices) > 0:
        reuse_existing = auto_slice_config.get('reuse_existing_slices', True)
        if reuse_existing:
            print(f"\nSliced meshes already exist at {slice_output_dir}")
            print("Reusing existing slices. Set 'reuse_existing_slices: false' to regenerate.")
            return slice_output_dir
        else:
            print(f"\nRemoving existing slices in {slice_output_dir}...")
            for f in existing_slices:
                os.remove(f)
    
    # Get all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))
    
    if not mesh_files:
        print(f"Warning: No PLY files found in {mesh_dir}")
        return mesh_dir
    
    print(f"\nFound {len(mesh_files)} mesh(es) to slice")
    
    # Get slicing parameters
    slicing_mode = auto_slice_config.get('mode', 'grid')
    grid_divisions = tuple(auto_slice_config.get('grid_divisions', [3, 3, 1]))
    target_faces_per_box = auto_slice_config.get('target_faces_per_box', 5000)
    min_boxes = auto_slice_config.get('min_boxes', 1)  # Changed to 1 to allow no slicing for small meshes
    max_boxes = auto_slice_config.get('max_boxes', 50)
    
    # Slice each mesh
    all_sliced_paths = []
    for mesh_path in mesh_files:
        try:
            sliced_paths = slice_large_mesh(
                mesh_path=mesh_path,
                output_dir=slice_output_dir,
                grid_divisions=grid_divisions,
                slicing_mode=slicing_mode,
                target_faces_per_box=target_faces_per_box,
                min_boxes=min_boxes,
                max_boxes=max_boxes
            )
            all_sliced_paths.extend(sliced_paths)
        except Exception as e:
            print(f"Error slicing {mesh_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTotal sliced meshes created: {len(all_sliced_paths)}")
    
    return slice_output_dir if all_sliced_paths else mesh_dir


def _write_ascii_ply_with_labels(file_path: str, vertices_xyz: np.ndarray, faces_idx: np.ndarray, 
                                   vertex_labels: np.ndarray, face_labels: np.ndarray) -> None:
    """Write an ASCII PLY with per-vertex and per-face classification labels.

    Args:
        file_path: Output PLY file path
        vertices_xyz: (N, 3) float32/float64 positions
        faces_idx: (M, 3) int vertex indices (triangles)
        vertex_labels: (N,) int vertex classification labels
        face_labels: (M,) int face classification labels
    """
    num_vertices = int(vertices_xyz.shape[0])
    num_faces = int(faces_idx.shape[0])

    with open(file_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int label\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property int label\n")
        f.write("end_header\n")

        # Vertices
        for i in range(num_vertices):
            x, y, z = vertices_xyz[i]
            label = int(vertex_labels[i])
            f.write(f"{float(x)} {float(y)} {float(z)} {label}\n")

        # Faces (assume triangles)
        for j in range(num_faces):
            v0, v1, v2 = faces_idx[j]
            label = int(face_labels[j])
            f.write(f"3 {int(v0)} {int(v1)} {int(v2)} {label}\n")


def _propagate_labels_to_duplicate_faces(mesh: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    """Copy a representative label to all duplicates so duplicate faces share identical labels.
    
    Args:
        mesh: Trimesh object
        labels: (M,) array of face labels
        
    Returns:
        Updated labels array with duplicates propagated
    """
    faces = mesh.faces
    if faces.size == 0:
        return labels
    faces_norm = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(faces_norm, axis=0, return_inverse=True, return_counts=True)
    if not np.any(counts > 1):
        return labels
    # Group indices by group id
    group_to_indices = {}
    for face_index, group_id in enumerate(inverse):
        gid = int(group_id)
        if gid not in group_to_indices:
            group_to_indices[gid] = []
        group_to_indices[gid].append(int(face_index))
    for indices in group_to_indices.values():
        if len(indices) <= 1:
            continue
        ref = indices[0]
        labels[indices] = labels[ref]
    return labels


def save_mesh_with_predictions(mesh_path: str, predictions: np.ndarray, output_path: str, 
                                dataset=None, mesh_idx: int = None) -> None:
    """Save a mesh with predicted face labels.
    
    Args:
        mesh_path: Path to the original mesh file
        predictions: Array of predicted face labels
        output_path: Path to save the labeled mesh
        dataset: Optional dataset to get face ordering information
        mesh_idx: Optional mesh index in the dataset
    """
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        num_faces = mesh.faces.shape[0]
        num_vertices = mesh.vertices.shape[0]
        
        # Initialize face labels
        face_labels = np.zeros(num_faces, dtype=np.int32)
        
        # Assign predictions to faces
        # Note: predictions should match the face ordering from the dataset
        pred_len = min(len(predictions), num_faces)
        face_labels[:pred_len] = predictions[:pred_len].astype(np.int32)
        
        # Propagate labels to duplicate faces
        face_labels = _propagate_labels_to_duplicate_faces(mesh, face_labels)
        
        # Derive vertex labels by majority voting from incident faces
        vertex_labels = np.zeros(num_vertices, dtype=np.int32)
        vertex_label_counts = {}  # vertex_idx -> {label: count}
        
        for face_idx, (v0, v1, v2) in enumerate(mesh.faces):
            face_label = int(face_labels[face_idx])
            for v in [v0, v1, v2]:
                if v not in vertex_label_counts:
                    vertex_label_counts[v] = {}
                vertex_label_counts[v][face_label] = vertex_label_counts[v].get(face_label, 0) + 1
        
        # Assign vertex labels based on majority vote
        for v_idx, label_counts in vertex_label_counts.items():
            vertex_labels[v_idx] = max(label_counts.items(), key=lambda x: x[1])[0]
        
        # Save mesh with labels
        vertices_xyz = mesh.vertices.astype(np.float64)
        faces_idx = mesh.faces.astype(np.int64)
        _write_ascii_ply_with_labels(output_path, vertices_xyz, faces_idx, vertex_labels, face_labels)
        
        # Print statistics
        unique_labels, counts = np.unique(face_labels, return_counts=True)
        print(f"  Face label distribution: {dict(zip(unique_labels, counts))}")
        
    except Exception as e:
        print(f"Error saving mesh {mesh_path}: {e}")


def build_model_from_config(config, device, use_texture=False):
    feature_dim = config['model']['feature_dim']
    embedding_dim = config['model']['embedding_dim']
    num_heads = config['model']['num_heads']
    num_attention_blocks = config['model']['num_attention_blocks']
    N_class = config['model']['n_classes']
    dropout = config['model'].get('dropout', 0.1)
    use_hierarchical = config['model'].get('use_hierarchical', False)
    fourier = config['model'].get('fourier', False)
    relative_positional_encoding = config['model'].get('relative_positional_encoding', False)

    if use_texture:
        # Create integrated texture-geometry model
        texture_embed_dim = config['model'].get('texture_embed_dim', 64)
        fusion_method = config['model'].get('fusion_method', 'gated')
        max_texture_pixels = config['model'].get('max_texture_pixels', 128)
        
        encoder = IntegratedTextureGeometryModel(
            geometry_feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            texture_embed_dim=texture_embed_dim,
            num_heads=num_heads,
            num_attention_blocks=num_attention_blocks,
            dropout=dropout,
            summary_mode='cls',
            use_hierarchical=use_hierarchical,
            fourier=fourier,
            relative_positional_encoding=relative_positional_encoding,
            fusion_method=fusion_method,
            max_texture_pixels=max_texture_pixels
        )
        model = IntegratedDownstreamClassifier(
            integrated_encoder=encoder,
            num_classes=N_class,
            embedding_dim=embedding_dim,
            dropout=dropout,
            freeze_encoder_layers=0,
            fusion_method=fusion_method
        ).to(device)
        print("Created integrated texture-geometry model for inference")
    else:
        # Create geometry-only model
        encoder = nomeformer(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_attention_blocks=num_attention_blocks,
            dropout=dropout,
            summary_mode='cls',
            use_hierarchical=use_hierarchical,
            num_hierarchical_stages=1,
            fourier=fourier,
            relative_positional_encoding=relative_positional_encoding,
        )
        model = DownstreamClassifier(encoder, N_class, embedding_dim, dropout, True, 0).to(device)
        print("Created geometry-only model for inference")
    
    return model


def predict(model, data_loader, num_classes, device, use_texture=False, output_dir=None, 
             save_meshes=True, test_mesh_dir=None, dataset=None):
    """
    Run inference without labels - just get predictions.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the test data
        num_classes: Number of classes
        device: Device to run on
        use_texture: Whether using texture features
        output_dir: Optional directory to save predictions
        save_meshes: Whether to save meshes with predicted labels
        test_mesh_dir: Directory containing the original mesh files
        dataset: Dataset object to get mesh file information
        
    Returns:
        all_predictions: List of predictions for each mesh
    """
    model.eval()
    all_predictions = []
    mesh_predictions = {}  # mesh_file -> list of predictions
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if use_texture:
                # Texture dataset returns: (geometry_features, labels, texture_sequences, masks, texture_masks)
                geometry_features, labels, texture_sequences, masks, texture_masks = data
                geometry_features = geometry_features.to(device)
                texture_sequences = texture_sequences.to(device)
                masks = masks.to(device)
                texture_masks = texture_masks.to(device)
                
                logits = model(geometry_features, texture_sequences, masks, texture_masks)
            else:
                # Geometry-only dataset returns: (batch, labels, masks)
                batch, labels, masks = data
                batch = batch.to(device)
                masks = masks.to(device)
                
                logits = model(batch, masks)
            
            # Get predictions
            pred = logits.reshape(-1, num_classes)
            mask_flat = masks.view(-1)
            valid_mask = mask_flat == 1
            
            # Only keep predictions for valid (non-padded) faces
            pred_valid = pred[valid_mask]
            pred_classes = pred_valid.argmax(dim=1)
            pred_np = pred_classes.cpu().numpy()
            
            all_predictions.append(pred_np)
            
            # Track predictions per mesh if saving meshes
            if save_meshes and dataset is not None:
                # Get the mesh file for this batch
                # The dataset index corresponds to cluster batches, need to map back to mesh
                mesh_idx = batch_idx // (dataset.n_clusters // dataset.clusters_per_batch) if hasattr(dataset, 'n_clusters') else batch_idx
                if mesh_idx < len(dataset.mesh_files):
                    mesh_file = dataset.mesh_files[mesh_idx]
                    if mesh_file not in mesh_predictions:
                        mesh_predictions[mesh_file] = []
                    mesh_predictions[mesh_file].append(pred_np)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    print(f"Prediction complete! Processed {len(all_predictions)} batches.")
    
    # Save meshes with predicted labels
    if save_meshes and test_mesh_dir is not None and mesh_predictions:
        mesh_output_dir = os.path.join(output_dir, "predicted_meshes") if output_dir else "predicted_meshes"
        os.makedirs(mesh_output_dir, exist_ok=True)
        
        print(f"\nSaving {len(mesh_predictions)} meshes with predicted labels...")
        for mesh_file, pred_list in mesh_predictions.items():
            # Concatenate all predictions for this mesh
            all_preds = np.concatenate(pred_list) if len(pred_list) > 1 else pred_list[0]
            
            # Construct paths
            mesh_path = os.path.join(test_mesh_dir, mesh_file)
            base_name = os.path.splitext(mesh_file)[0]
            output_path = os.path.join(mesh_output_dir, f"{base_name}_predicted.ply")
            
            # Save mesh with predictions
            print(f"Saving {mesh_file}...")
            save_mesh_with_predictions(mesh_path, all_preds, output_path)
        
        print(f"Predicted meshes saved to {mesh_output_dir}")
    
    return all_predictions


def evaluate(model, data_loader, num_classes, ignore_index, device, use_texture=False):
    """
    Run evaluation with labels - compute metrics.
    """
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes)
    if ignore_index is not None:
        miou_metric = IoU(cm=cm, ignore_index=ignore_index)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='none', ignore_index=ignore_index).to(device)
        acc_metric = Accuracy(task='multiclass', num_classes=num_classes, average='none', ignore_index=ignore_index).to(device)
    else:
        miou_metric = IoU(cm=cm)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='none').to(device)
        acc_metric = Accuracy(task='multiclass', num_classes=num_classes, average='none').to(device)

    f1_metric.reset()
    acc_metric.reset()

    with torch.no_grad():
        for data in data_loader:
            if use_texture:
                # Texture dataset returns: (geometry_features, labels, texture_sequences, masks, texture_masks)
                geometry_features, labels, texture_sequences, masks, texture_masks = data
                geometry_features = geometry_features.to(device)
                texture_sequences = texture_sequences.to(device)
                labels = labels.to(device) +1
                masks = masks.to(device)
                texture_masks = texture_masks.to(device)
                
                logits = model(geometry_features, texture_sequences, masks, texture_masks)
            else:
                # Geometry-only dataset returns: (batch, labels, masks)
                batch, labels, masks = data
                batch = batch.to(device)
                labels = labels.to(device) + 1
                masks = masks.to(device)
                
                logits = model(batch, masks)
            
            pred = logits.reshape(-1, num_classes)
            target = labels.reshape(-1).long()
            mask = masks.view(-1)
            valid_mask = mask == 1
            pred = pred[valid_mask]
            target = target[valid_mask]
            
            # ConfusionMatrix expects logits
            cm.update((pred, target))
            
            # F1Score and Accuracy expect class indices, not logits
            pred_classes = pred.argmax(dim=1)
            f1_metric.update(pred_classes, target)
            acc_metric.update(pred_classes, target)

    f1_scores = f1_metric.compute()
    accuracy = acc_metric.compute()
    if ignore_index is not None and ignore_index < num_classes:
        class_mask = torch.arange(num_classes, device=f1_scores.device) != ignore_index
        mean_f1 = f1_scores[class_mask].mean().item()
        mean_acc = accuracy[class_mask].mean().item()
    else:
        mean_f1 = f1_scores.mean().item()
        mean_acc = accuracy.mean().item()
    miou = miou_metric.compute()

    return mean_f1, mean_acc, miou, f1_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained downstream model on a test directory.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config containing evaluation paths')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    n_clusters = config['model']['n_clusters']
    clusters_per_batch = config['model'].get('clusters_per_batch', 300)
    PE = config['model']['use_pe']
    batch_size = config['model'].get('batch_size', 1)
    ignore_index = config['model'].get('ignore_index', None)
    num_classes = config['model']['n_classes']
    include_normals = config['model'].get('include_normals', True)
    additional_geometrical_features = config['model'].get('additional_geometrical_features', False)
    
    # Texture-specific configuration
    use_texture = config['model'].get('use_texture', False)
    texture_patch_size = config['model'].get('texture_patch_size', 16)
    max_texture_pixels = config['model'].get('max_texture_pixels', 128)
    
    # Read evaluation paths from config
    test_mesh_dir = config['paths']['test_mesh_dir']
    test_label_dir = config['paths'].get('test_label_dir', None)
    test_json_dir = config['paths'].get('test_json_dir', None)
    test_texture_dir = config['paths'].get('test_texture_dir', None)
    checkpoint_path = config['paths']['checkpoint_path']
    output_dir = config['paths'].get('output_dir', None)
    
    # Auto-slicing configuration
    auto_slice_config = config.get('auto_slice', {})
    auto_slice_enabled = auto_slice_config.get('enabled', False)
    
    # Process mesh slicing if enabled
    original_mesh_dir = test_mesh_dir  # Keep reference to original
    if auto_slice_enabled:
        # Determine output directory for sliced meshes
        slice_output_dir = auto_slice_config.get('output_dir', None)
        if slice_output_dir is None:
            # Default: create 'sliced_meshes' subdirectory next to original mesh directory
            parent_dir = os.path.dirname(test_mesh_dir.rstrip('/\\'))
            slice_output_dir = os.path.join(parent_dir, 'sliced_meshes')
        
        # Clear existing sliced meshes at the beginning of each inference run
        if os.path.exists(slice_output_dir) and len(os.listdir(slice_output_dir)) > 0:
            print("\n" + "="*60)
            print("Clearing existing sliced meshes before inference...")
            print("="*60)
            sliced_files = glob.glob(os.path.join(slice_output_dir, "*.ply"))
            for f in sliced_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")
            print(f"Removed {len(sliced_files)} sliced mesh files from {slice_output_dir}")
            print("="*60 + "\n")
        
        # Process meshes with slicing
        test_mesh_dir = process_mesh_directory_with_slicing(
            mesh_dir=test_mesh_dir,
            slice_output_dir=slice_output_dir,
            auto_slice_config=auto_slice_config
        )
        
        print(f"\nUsing mesh directory for inference: {test_mesh_dir}")
    else:
        print("\nAuto-slicing is disabled. Using original meshes.")
    
    # Check if we should save meshes with predictions
    save_meshes = config['paths'].get('save_meshes', True)
    
    # Check if we have labels (evaluation mode) or not (prediction mode)
    has_labels = config['paths'].get('has_labels', True)
    if test_label_dir is None:
        has_labels = False
        print("No label directory provided - running in prediction mode")
    
    if has_labels:
        print("Running in evaluation mode (with labels)")
    else:
        print("Running in prediction mode (without labels)")
        if save_meshes:
            print("Will save predicted meshes with labels")

    # Create dataset and dataloader based on whether texture is used
    if use_texture and test_texture_dir is not None:
        print(f"Using MeshTextureDataset with texture features from {test_texture_dir}")
        dataset = MeshTextureDataset(
            mesh_dir=test_mesh_dir,
            label_dir=test_label_dir if has_labels else None,
            texture_dir=test_texture_dir,
            n_clusters=n_clusters,
            clusters_per_batch=clusters_per_batch,
            PE=PE,
            json_dir=test_json_dir,
            augmentation=None,
            transform=None,
            include_normals=include_normals,
            additional_geometrical_features=additional_geometrical_features,
            texture_patch_size=texture_patch_size,
            max_texture_pixels=max_texture_pixels,
            pe_bbox_normalized=True,
            require_labels=has_labels
        )
        # Clear cached meshes at the beginning of inference
        print("\n" + "="*60)
        print("Clearing cached meshes before inference...")
        print("="*60)
        dataset.clear_cache()
        print("="*60 + "\n")
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=texture_custom_collate_fn)
    else:
        print(f"Using MeshDataset with geometry features only from {test_mesh_dir}")
        dataset = MeshDataset(
            mesh_dir=test_mesh_dir,
            label_dir=test_label_dir if has_labels else None,
            n_clusters=n_clusters,
            clusters_per_batch=clusters_per_batch,
            PE=PE,
            json_dir=test_json_dir,
            augmentation=None,
            transform=None,
            include_normals=include_normals,
            additional_geometrical_features=additional_geometrical_features,
            require_labels=has_labels
        )
        # Clear cached meshes at the beginning of inference
        print("\n" + "="*60)
        print("Clearing cached meshes before inference...")
        print("="*60)
        dataset.clear_cache()
        print("="*60 + "\n")
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Build model with texture support if needed
    model = build_model_from_config(config, device, use_texture=use_texture and test_texture_dir is not None)
    
    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    current_state = model.state_dict()
    checkpoint_state = ckpt['model_state_dict']
    matched = {k: v for k, v in checkpoint_state.items() if k in current_state and current_state[k].shape == v.shape}
    current_state.update(matched)
    model.load_state_dict(current_state, strict=False)
    
    # Report classifier restoration explicitly
    classifier_keys = [k for k in matched.keys() if k.startswith('classifier.')]
    print(f"Restored {len(classifier_keys)} classifier params; matched {len(matched)} / {len(current_state)} total params.")
    model = model.to(device)

    # Run inference or evaluation based on mode
    if has_labels:
        # Evaluation mode - compute metrics
        mean_f1, mean_acc, miou, f1_scores = evaluate(
            model, data_loader, num_classes, ignore_index, device, 
            use_texture=use_texture and test_texture_dir is not None
        )
        print(f"Evaluation Results:\n  Mean F1: {mean_f1:.4f}\n  Mean Accuracy: {mean_acc:.4f}\n  mIoU: {miou}")
        print(f"Per-class F1: {f1_scores}")
    else:
        # Prediction mode - just get predictions
        predictions = predict(
            model, data_loader, num_classes, device,
            use_texture=use_texture and test_texture_dir is not None,
            output_dir=output_dir,
            save_meshes=save_meshes,
            test_mesh_dir=test_mesh_dir,
            dataset=dataset
        )
        print(f"Generated predictions for {len(predictions)} batches")
        if output_dir:
            print(f"Predictions saved to: {output_dir}")


if __name__ == '__main__':
    main()


