"""
Train DINOv2 Approach 3 Mixup Plus: Final Enhanced Version

This script combines multiple advanced techniques for maximum unpaired class accuracy:
  1. Cross-Entropy loss (class-balanced)
  2. Triplet Margin loss
  3. Domain mixup (leaf ↔ field)
  4. Consistency loss for mixup features
  5. Cosine classifier head (normalized features + weights)

Outputs to outputs_approach3_mixup_plus/ directory.
Fully compatible with plant_gui.py inference pipeline.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train_approach_2 import (
    compute_subset_accuracy,
    load_class_id_list,
    stratified_split,
    topk_accuracy,
)


class CosineClassifier(nn.Module):
    """
    Cosine similarity-based classifier with learnable scale.
    
    Computes: logits = scale * cos(θ) = scale * (W·x) / (||W|| * ||x||)
    """
    
    def __init__(self, in_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) feature tensor
        
        Returns:
            logits: (B, C) scaled cosine similarity
        """
        # Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)  # (B, D)
        w_norm = F.normalize(self.weight, p=2, dim=1)  # (C, D)
        
        # Compute scaled cosine similarity
        logits = self.scale * torch.matmul(x_norm, w_norm.t())  # (B, C)
        return logits


class MLPWithCosineHead(nn.Module):
    """
    MLP feature extractor + Cosine classifier head.
    
    Architecture:
      - Input (D) → Linear(512) → ReLU → Linear(256) → ReLU
      - Features (256) → CosineClassifier → Logits (num_classes)
    """
    
    def __init__(self, input_dim: int, num_classes: int, cosine_scale: float = 20.0):
        super().__init__()
        
        # Feature extractor backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        
        # Cosine classifier head
        self.head = CosineClassifier(256, num_classes, scale=cosine_scale)
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, D) input features
            return_features: If True, return both logits and intermediate features
        
        Returns:
            logits: (B, num_classes)
            features: (B, 256) if return_features=True
        """
        features = self.backbone(x)  # (B, 256)
        logits = self.head(features)  # (B, num_classes)
        
        if return_features:
            return logits, features
        return logits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINOv2 Approach 3 Mixup Plus (Final Enhanced Version)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--train-features",
        type=str,
        default="output_approach_2/features_train.npz",
        help="Path to training features NPZ file (relative to data-root)",
    )
    parser.add_argument(
        "--test-features",
        type=str,
        default="output_approach_2/features_test.npz",
        help="Path to test features NPZ file (relative to data-root)",
    )
    parser.add_argument(
        "--label-mapping",
        type=str,
        default="output_approach_2/label_mapping.json",
        help="Path to label mapping JSON file (relative to data-root)",
    )
    parser.add_argument(
        "--class-with-pairs",
        type=str,
        default="list/class_with_pairs.txt",
        help="Path to class with pairs list file (relative to data-root)",
    )
    parser.add_argument(
        "--class-without-pairs",
        type=str,
        default="list/class_without_pairs.txt",
        help="Path to class without pairs list file (relative to data-root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_approach_3_model_3",
        help="Output directory for trained model and metrics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--lambda-triplet",
        type=float,
        default=0.2,
        help="Weight for triplet loss (use 0.0 to disable)",
    )
    parser.add_argument(
        "--lambda-consistency",
        type=float,
        default=0.2,
        help="Weight for consistency loss on mixup features",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin for TripletMarginLoss",
    )
    parser.add_argument(
        "--enable-mixup",
        action="store_true",
        default=True,
        help="Enable domain mixup in feature space",
    )
    parser.add_argument(
        "--no-mixup",
        action="store_false",
        dest="enable_mixup",
        help="Disable domain mixup",
    )
    parser.add_argument(
        "--mixup-alpha-min",
        type=float,
        default=0.3,
        help="Minimum alpha value for mixup",
    )
    parser.add_argument(
        "--mixup-alpha-max",
        type=float,
        default=0.7,
        help="Maximum alpha value for mixup",
    )
    parser.add_argument(
        "--unpaired-class-weight",
        type=float,
        default=3.0,
        help="Class weight multiplier for unpaired classes in CE loss",
    )
    parser.add_argument(
        "--cosine-scale",
        type=float,
        default=20.0,
        help="Scale factor for cosine classifier logits",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unpack_checkpoint(raw_obj):
    """
    Handle checkpoints that either store a bare state_dict or wrap it with metadata.
    
    Returns:
        state_dict, metadata_dict
    """
    if isinstance(raw_obj, dict) and ("state_dict" in raw_obj or "model_state" in raw_obj):
        state_dict = raw_obj.get("state_dict") or raw_obj.get("model_state")
        metadata = {k: v for k, v in raw_obj.items() if k not in {"state_dict", "model_state"}}
    else:
        state_dict = raw_obj
        metadata = {}
    return state_dict, metadata


def save_classifier_checkpoint(
    model: nn.Module,
    path: Path,
    input_dim: int,
    num_classes: int,
    cosine_scale: float,
) -> None:
    """Save classifier weights along with minimal metadata for GUI loading."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": "mlp_cosine_head",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "cosine_scale": cosine_scale,
        "version": "approach3_mixup_plus_v1",
    }
    torch.save(checkpoint, path)


def build_class_weights(
    num_classes: int,
    unpaired_class_ids: set,
    unpaired_weight: float,
) -> torch.Tensor:
    """
    Build class weights for balanced cross-entropy loss.
    
    Args:
        num_classes: Total number of classes
        unpaired_class_ids: Set of unpaired class IDs
        unpaired_weight: Weight multiplier for unpaired classes
    
    Returns:
        weights: (num_classes,) tensor of class weights
    """
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cid in unpaired_class_ids:
        if cid < num_classes:
            weights[cid] = unpaired_weight
    return weights


def mine_triplets_in_batch(
    hidden: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine triplets within a batch for Triplet Margin Loss.
    
    Returns:
        anchors, positives, negatives: (T, H) tensors or empty tensors
    """
    device = hidden.device
    unique_labels = torch.unique(labels)
    
    anchors_list = []
    positives_list = []
    negatives_list = []
    
    for label in unique_labels:
        pos_mask = labels == label
        neg_mask = labels != label
        
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        
        if len(pos_indices) < 2 or len(neg_indices) < 1:
            continue
        
        perm = torch.randperm(len(pos_indices))
        anchor_idx = pos_indices[perm[0]]
        positive_idx = pos_indices[perm[1]]
        neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))[0]]
        
        anchors_list.append(hidden[anchor_idx])
        positives_list.append(hidden[positive_idx])
        negatives_list.append(hidden[neg_idx])
    
    if not anchors_list:
        return (
            torch.empty((0, hidden.size(1)), device=device),
            torch.empty((0, hidden.size(1)), device=device),
            torch.empty((0, hidden.size(1)), device=device),
        )
    
    return torch.stack(anchors_list), torch.stack(positives_list), torch.stack(negatives_list)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    triplet_criterion: nn.Module,
    lambda_triplet: float,
    lambda_consistency: float,
    device: torch.device,
    enable_mixup: bool = False,
    mixup_alpha_min: float = 0.3,
    mixup_alpha_max: float = 0.7,
) -> Tuple[float, float, float, float]:
    """
    Train for one epoch with all loss components.
    
    Returns:
        avg_ce_loss, avg_triplet_loss, avg_consistency_loss, avg_total_loss
    """
    model.train()
    
    total_ce_loss = 0.0
    total_triplet_loss = 0.0
    total_consistency_loss = 0.0
    total_combined_loss = 0.0
    num_batches = 0
    
    for batch_data in tqdm(train_loader, desc="Training", leave=False):
        if len(batch_data) == 3:
            features, labels, domains = batch_data
            features = features.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
        else:
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
            domains = None
        
        optimizer.zero_grad()
        
        # Initialize consistency loss
        consistency_loss = torch.tensor(0.0, device=device)
        
        # Apply domain mixup if enabled
        if enable_mixup and domains is not None:
            leaf_mask = domains == 0
            field_mask = domains == 1
            
            leaf_idx = torch.where(leaf_mask)[0]
            field_idx = torch.where(field_mask)[0]
            
            if len(leaf_idx) > 0 and len(field_idx) > 0:
                # Perform mixup
                n_mix = min(len(leaf_idx), len(field_idx))
                
                leaf_perm = torch.randperm(len(leaf_idx))[:n_mix]
                field_perm = torch.randperm(len(field_idx))[:n_mix]
                
                leaf_feats = features[leaf_idx[leaf_perm]]
                field_feats = features[field_idx[field_perm]]
                leaf_labels = labels[leaf_idx[leaf_perm]]
                
                # Sample mixing coefficient
                alpha = torch.rand(n_mix, 1, device=device)
                alpha = alpha * (mixup_alpha_max - mixup_alpha_min) + mixup_alpha_min
                
                # Create mixed input features
                mixed_feats = alpha * leaf_feats + (1 - alpha) * field_feats
                
                # Forward pass for consistency loss
                if lambda_consistency > 0:
                    _, feat_leaf = model(leaf_feats, return_features=True)
                    _, feat_field = model(field_feats, return_features=True)
                    _, feat_mixed = model(mixed_feats, return_features=True)
                    
                    # Target: linear interpolation of embeddings
                    feat_target = alpha * feat_leaf + (1 - alpha) * feat_field
                    
                    # MSE loss between mixed embedding and target
                    consistency_loss = F.mse_loss(feat_mixed, feat_target)
                
                # Augment batch with mixed samples
                features = torch.cat([features, mixed_feats], dim=0)
                labels = torch.cat([labels, leaf_labels], dim=0)
        
        # Forward pass
        logits, hidden = model(features, return_features=True)
        
        # Cross-entropy loss
        ce_loss = ce_criterion(logits, labels)
        
        # Triplet loss
        if lambda_triplet > 0:
            anchors, positives, negatives = mine_triplets_in_batch(hidden, labels)
            if anchors.size(0) > 0:
                triplet_loss = triplet_criterion(anchors, positives, negatives)
            else:
                triplet_loss = torch.tensor(0.0, device=device)
        else:
            triplet_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = ce_loss + lambda_triplet * triplet_loss + lambda_consistency * consistency_loss
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_ce_loss += ce_loss.item()
        total_triplet_loss += triplet_loss.item()
        total_consistency_loss += consistency_loss.item()
        total_combined_loss += total_loss.item()
        num_batches += 1
    
    return (
        total_ce_loss / num_batches,
        total_triplet_loss / num_batches,
        total_consistency_loss / num_batches,
        total_combined_loss / num_batches,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_logits: bool = False,
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate model on a dataset.
    
    Returns:
        top1_acc, top5_acc, all_logits, all_targets
    """
    model.eval()
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
            if len(batch_data) == 3:
                features, labels, _ = batch_data
            else:
                features, labels = batch_data
            
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features, return_features=False)
            
            all_logits.append(logits)
            all_targets.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    top1_acc, top5_acc = topk_accuracy(all_logits, all_targets, topk=(1, 5))
    
    if return_logits:
        return top1_acc, top5_acc, all_logits, all_targets
    else:
        return top1_acc, top5_acc, None, None


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_features_path = data_root / args.train_features
    test_features_path = data_root / args.test_features
    label_mapping_path = data_root / args.label_mapping
    
    print("=" * 80)
    print("DINOv2 Approach 3 Mixup Plus (Final Enhanced Version)")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Train features: {train_features_path}")
    print(f"Test features: {test_features_path}")
    print(f"Output directory: {output_dir}")
    print(f"\nHyperparameters:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Lambda (triplet): {args.lambda_triplet}")
    print(f"  - Lambda (consistency): {args.lambda_consistency}")
    print(f"  - Margin: {args.margin}")
    print(f"  - Enable mixup: {args.enable_mixup}")
    if args.enable_mixup:
        print(f"  - Mixup alpha range: [{args.mixup_alpha_min}, {args.mixup_alpha_max}]")
    print(f"  - Unpaired class weight: {args.unpaired_class_weight}")
    print(f"  - Cosine scale: {args.cosine_scale}")
    print(f"  - Validation ratio: {args.val_ratio}")
    print(f"  - Random seed: {args.seed}")
    print("=" * 80)
    
    # Load training features
    print("\n[1/6] Loading training features...")
    train_npz = np.load(train_features_path)
    X_all = train_npz["X_train"]
    y_all = train_npz["y_train"]
    domain_all = train_npz.get("domain_train", None)
    
    print(f"  Loaded {X_all.shape[0]} training samples with {X_all.shape[1]} dimensions")
    if domain_all is not None:
        print(f"  Domain info: {(domain_all == 0).sum()} leaf, {(domain_all == 1).sum()} field")
    else:
        print("  WARNING: No domain info. Mixup disabled.")
        args.enable_mixup = False
    
    # Load test features
    print("\n[2/6] Loading test features...")
    test_npz = np.load(test_features_path)
    X_test = test_npz["X_test"]
    y_test = test_npz["y_test"]
    domain_test = test_npz.get("domain_test", None)
    print(f"  Loaded {X_test.shape[0]} test samples")
    
    # Load label mapping and class lists
    print("\n[3/6] Loading label mapping and class lists...")
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    
    pair_class_ids_raw = load_class_id_list(data_root / args.class_with_pairs)
    unpair_class_ids_raw = load_class_id_list(data_root / args.class_without_pairs)
    
    pair_class_ids = {label_mapping[label] for label in pair_class_ids_raw if label in label_mapping}
    unpair_class_ids = {label_mapping[label] for label in unpair_class_ids_raw if label in label_mapping}
    
    print(f"  Paired classes: {len(pair_class_ids)}")
    print(f"  Unpaired classes: {len(unpair_class_ids)}")
    
    # Perform stratified split
    print(f"\n[4/6] Performing stratified train/val split (val_ratio={args.val_ratio})...")
    train_indices, val_indices = stratified_split(y_all, val_ratio=args.val_ratio, seed=args.seed)
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    
    if domain_all is not None:
        domain_train = domain_all[train_indices]
        domain_val = domain_all[val_indices]
    else:
        domain_train = None
        domain_val = None
    
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    
    # Create DataLoaders
    print("\n[5/6] Creating DataLoaders...")
    
    if domain_train is not None:
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
            torch.from_numpy(domain_train).long(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
            torch.from_numpy(domain_val).long(),
        )
    else:
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
        )
    
    if domain_test is not None:
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).long(),
            torch.from_numpy(domain_test).long(),
        )
    else:
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).long(),
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Build model
    print("\n[6/6] Building MLP with Cosine Classifier...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    input_dim = X_all.shape[1]
    num_classes = int(y_all.max()) + 1
    print(f"  Input dim: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    
    model = MLPWithCosineHead(input_dim, num_classes, cosine_scale=args.cosine_scale).to(device)
    
    # Build class-balanced weights
    class_weights = build_class_weights(num_classes, unpair_class_ids, args.unpaired_class_weight)
    print(f"  Class-balanced CE: unpaired classes weighted {args.unpaired_class_weight}x")
    
    # Define loss functions and optimizer
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    triplet_criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 80)
    
    best_val_top1 = 0.0
    best_epoch = 0
    best_model_path = output_dir / "best_mlp_classifier.pt"
    
    for epoch in range(1, args.epochs + 1):
        avg_ce, avg_triplet, avg_consistency, avg_total = train_epoch(
            model,
            train_loader,
            optimizer,
            ce_criterion,
            triplet_criterion,
            args.lambda_triplet,
            args.lambda_consistency,
            device,
            enable_mixup=args.enable_mixup,
            mixup_alpha_min=args.mixup_alpha_min,
            mixup_alpha_max=args.mixup_alpha_max,
        )
        
        val_top1, val_top5, _, _ = evaluate_model(model, val_loader, device)
        
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"CE: {avg_ce:.4f} | Trip: {avg_triplet:.4f} | Cons: {avg_consistency:.4f} | "
            f"Total: {avg_total:.4f} | Val Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}%"
        )
        
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            save_classifier_checkpoint(
                model,
                best_model_path,
                input_dim=input_dim,
                num_classes=num_classes,
                cosine_scale=args.cosine_scale,
            )
            print(f"  ✓ New best model saved! (Val Top-1: {val_top1:.2f}%)")
    
    print("=" * 80)
    print(f"\nTraining complete! Best val Top-1: {best_val_top1:.2f}% at epoch {best_epoch}")
    print(f"Best model saved to: {best_model_path}")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluating best model on test set...")
    print("=" * 80)
    
    raw_checkpoint = torch.load(best_model_path, map_location=device)
    state_dict, checkpoint_meta = _unpack_checkpoint(raw_checkpoint)
    model.head.scale = checkpoint_meta.get("cosine_scale", args.cosine_scale)
    model.load_state_dict(state_dict)
    
    # Validation metrics for the best checkpoint
    val_top1_best, val_top5_best, _, _ = evaluate_model(model, val_loader, device)
    
    test_top1, test_top5, test_logits, test_targets = evaluate_model(
        model, test_loader, device, return_logits=True
    )
    
    pair_top1, pair_top5 = compute_subset_accuracy(test_logits, test_targets, pair_class_ids, topk=(1, 5))
    unpair_top1, unpair_top5 = compute_subset_accuracy(test_logits, test_targets, unpair_class_ids, topk=(1, 5))
    
    print()
    print(f"Validation Top-1 accuracy: {val_top1_best:.2f}%")
    print(f"Validation Top-5 accuracy: {val_top5_best:.2f}%")
    print(f"Test Overall Top-1 accuracy: {test_top1:.2f}%")
    print(f"Test Overall Top-5 accuracy: {test_top5:.2f}%")
    if not math.isnan(pair_top1):
        print(f"Test Pair Top-1 accuracy: {pair_top1:.2f}%")
        print(f"Test Pair Top-5 accuracy: {pair_top5:.2f}%")
    else:
        print("Test Pair accuracy: not available (no pair-class samples in test set).")
    if not math.isnan(unpair_top1):
        print(f"Test Unpaired Top-1 accuracy: {unpair_top1:.2f}%")
        print(f"Test Unpaired Top-5 accuracy: {unpair_top5:.2f}%")
    else:
        print("Test Unpaired accuracy: not available (no unpaired-class samples in test set).")
    
    # Save metrics
    metrics = {
        "overall_top1": test_top1,
        "overall_top5": test_top5,
        "paired_top1": pair_top1,
        "paired_top5": pair_top5,
        "unpaired_top1": unpair_top1,
        "unpaired_top5": unpair_top5,
        "best_val_top1": best_val_top1,
        "best_epoch": best_epoch,
        "lambda_triplet": args.lambda_triplet,
        "lambda_consistency": args.lambda_consistency,
        "margin": args.margin,
        "enable_mixup": args.enable_mixup,
        "mixup_alpha_min": args.mixup_alpha_min,
        "mixup_alpha_max": args.mixup_alpha_max,
        "unpaired_class_weight": args.unpaired_class_weight,
        "cosine_scale": args.cosine_scale,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "val_top1": val_top1_best,
        "val_top5": val_top5_best,
        "checkpoint_path": str(best_model_path),
        "arch": checkpoint_meta.get("arch", "mlp_cosine_head"),
        "input_dim": input_dim,
        "num_classes": num_classes,
    }
    
    metrics_path = output_dir / "metrics_approach3_mixup_plus.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    print("=" * 80)
    print("✓ Approach 3 Mixup Plus training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
