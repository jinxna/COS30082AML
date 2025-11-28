# train_dinov2_triplet_prototype_eval_final_fixed_with_bar.py
# Fixed to match Approach 3 (DINOv2 feature extractor + MLP classifier finetune + test eval) with tqdm
import argparse
import io
import json
import math
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# -----------------------
# CONFIG / DEFAULTS
# -----------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed: DINOv2 + MLP finetune (replicates Approach 3)."
    )
    parser.add_argument("--data-root", default=r"C:\Users\scana\Downloads\COS30082AML-main\COS30082AML-main", type=str)
    parser.add_argument("--train-list", default="list/train.txt", type=str)
    parser.add_argument("--test-list", default="list/test.txt", type=str)
    parser.add_argument("--groundtruth", default="list/groundtruth.txt", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--head-lr", default=1e-3, type=float)
    parser.add_argument("--backbone-lr", default=1e-5, type=float)
    parser.add_argument("--output-dir", default="output_approach_3_model_2", type=str)
    parser.add_argument(
        "--checkpoint-path",
        default="model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz",
        type=str,
    )
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dinov2-arch", default="dinov2_vitb14_reg", type=str)
    parser.add_argument("--class-with-pairs", default="list/class_with_pairs.txt", type=str)
    parser.add_argument("--class-without-pairs", default="list/class_without_pairs.txt", type=str)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_default_transform(image_size: int = 518) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

# -----------------------
# DATASET
# -----------------------
class PlantDataset(Dataset):
    def __init__(self, data_root: str, list_file: str, with_labels: bool = True, transform=None) -> None:
        self.data_root = Path(data_root)
        self.list_file = Path(list_file)
        self.with_labels = with_labels
        self.transform = transform or build_default_transform()
        self.samples: List[Tuple[str, Optional[int]]] = []
        self._load_list()

    def _load_list(self) -> None:
        if not self.list_file.exists():
            raise FileNotFoundError(f"List file not found: {self.list_file}")
        with self.list_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if self.with_labels:
                    if len(parts) != 2:
                        raise ValueError(f"Expected '<path> <label>' per line, got: {line}")
                    rel_path, label = parts
                    self.samples.append((rel_path, int(label)))
                else:
                    rel_path = parts[0]
                    self.samples.append((rel_path, None))
        if not self.samples:
            raise ValueError(f"No samples found in {self.list_file}")

    @property
    def labels(self) -> List[int]:
        if not self.with_labels:
            raise AttributeError("Dataset does not provide labels.")
        return [label for _, label in self.samples if label is not None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        img_path = self.data_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.with_labels:
            return image, label
        return image, rel_path

def stratified_split(labels: Sequence[int], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for label, idxs in label_to_indices.items():
        idxs_copy = idxs[:]
        rng.shuffle(idxs_copy)
        if len(idxs_copy) == 1:
            train_indices.extend(idxs_copy)
            continue
        val_count = max(1, int(round(len(idxs_copy) * val_ratio)))
        if val_count >= len(idxs_copy):
            val_count = len(idxs_copy) - 1
        val_indices.extend(idxs_copy[:val_count])
        train_indices.extend(idxs_copy[val_count:])
    if not train_indices or not val_indices:
        raise RuntimeError("Failed to create non-empty train/val splits. Adjust val_ratio or dataset size.")
    return sorted(train_indices), sorted(val_indices)

def load_class_id_list(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    class_ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                class_ids.add(int(line.split()[0]))
            except ValueError:
                continue
    return class_ids

def build_label_mapping(label_arrays: Sequence[np.ndarray]) -> Dict[int, int]:
    combined: List[np.ndarray] = []
    for arr in label_arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        combined.append(arr.reshape(-1))
    if not combined:
        raise ValueError("No labels provided to build label mapping.")
    unique_labels = np.unique(np.concatenate(combined, axis=0))
    return {int(label): idx for idx, label in enumerate(unique_labels)}


def remap_labels(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(label)] for label in labels], dtype=np.int64)


# -----------------------
# CHECKPOINT LOADING UTIL (supports .tar.gz with internal .pt)
# -----------------------
def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    # try direct torch.load
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        pass

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with tarfile.open(checkpoint_path, "r:*") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        preferred_suffixes = {".pt", ".pth", ".bin", ".pth.tar", ".pt.tar"}
        selected_member = None
        for member in members:
            name = member.name.lower()
            suffix = Path(member.name).suffix.lower()
            if any(name.endswith(ps) for ps in preferred_suffixes) or suffix in preferred_suffixes or "state_dict" in Path(member.name).stem:
                selected_member = member
                break
        if selected_member is None and members:
            selected_member = members[0]
        if selected_member is None:
            raise RuntimeError(f"No suitable file found inside archive {checkpoint_path}")
        extracted = archive.extractfile(selected_member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract {selected_member.name} from archive {checkpoint_path}")
        buffer = io.BytesIO(extracted.read())
    return torch.load(buffer, map_location="cpu")


# -----------------------
# DINOv2 feature extractor wrapper (same semantics used in Approach 3)
# -----------------------
class DinoFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "embed_dim", None)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = None
        if hasattr(self.backbone, "forward_features"):
            outputs = self.backbone.forward_features(images)
        else:
            outputs = self.backbone(images)
        if isinstance(outputs, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooled_output"):
                if key in outputs:
                    feats = outputs[key]
                    break
            else:
                feats = next(iter(outputs.values()))
        else:
            feats = outputs
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if feats.ndim == 3:
            feats = feats[:, 0]
        return feats

def _align_pos_embed(state_dict: Dict[str, torch.Tensor], backbone: nn.Module) -> Dict[str, torch.Tensor]:
    if "pos_embed" not in state_dict or not hasattr(backbone, "pos_embed"):
        return state_dict
    src = state_dict["pos_embed"]
    dst = backbone.pos_embed
    if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
        return state_dict
    if src.shape == dst.shape:
        return state_dict
    if src.ndim != 3 or dst.ndim != 3:
        return state_dict
    num_src = src.shape[1]
    num_dst = dst.shape[1]
    if num_src == num_dst - 1:
        cls_token = state_dict.get("cls_token")
        if not isinstance(cls_token, torch.Tensor):
            cls_token = dst[:, :1, :].clone()
        elif cls_token.ndim == 2:
            cls_token = cls_token.unsqueeze(0)
        cls_token = cls_token[:, :1, :]
        patched = torch.cat([cls_token, src], dim=1)
        state_dict["pos_embed"] = patched
        print("[Info] Adjusted pos_embed by prepending CLS token to match target shape.")
    elif num_src == num_dst + 1:
        patched = src[:, 1:, :]
        state_dict["pos_embed"] = patched
        print("[Info] Adjusted pos_embed by removing extra token to match target shape.")
    else:
        print(f"[Warning] Unable to align pos_embed automatically (src tokens={num_src}, dst tokens={num_dst}).")
    return state_dict

def build_dinov2_backbone(checkpoint_path: str, device: torch.device, arch: str) -> DinoFeatureExtractor:
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
    except Exception as err:
        raise RuntimeError("Failed to instantiate DINOv2 backbone via torch.hub. Ensure Dinov2 is available.") from err

    checkpoint = load_checkpoint_state(Path(checkpoint_path))
    state_dict = checkpoint
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break

    state_dict = _align_pos_embed(state_dict, backbone)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading checkpoint (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading checkpoint (showing first 5): {unexpected[:5]}")

    feature_extractor = DinoFeatureExtractor(backbone).to(device)
    return feature_extractor

# -----------------------
# CLASSIFIER MLP + TRAIN / EVAL (continued)
# -----------------------
def build_mlp_classifier(input_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (1, 5)) -> List[float]:
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / targets.size(0)).item()
        results.append(acc)
    return results

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, return_logits: bool = False):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    top1, top5 = topk_accuracy(logits_cat.to(device), targets_cat.to(device), topk=(1, 5))
    if return_logits:
        return top1, top5, logits_cat, targets_cat
    return top1, top5

class RemappedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, label_mapping: Dict[int, int]) -> None:
        self.base = base_dataset
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, orig_label = self.base[idx]
        new_label = self.label_mapping[int(orig_label)]
        return image, new_label

class TestWithLabelsDataset(Dataset):
    def __init__(self, base_dataset: PlantDataset, groundtruth: Dict[str, int], label_mapping: Dict[int, int]) -> None:
        if base_dataset.with_labels:
            raise ValueError("base_dataset must be created with with_labels=False")
        self.base = base_dataset
        self.indices: List[int] = []
        self.labels: List[int] = []
        for idx, (rel_path, _) in enumerate(self.base.samples):
            image_name = Path(rel_path).name
            if image_name not in groundtruth:
                continue
            orig_label = groundtruth[image_name]
            if orig_label not in label_mapping:
                continue
            self.indices.append(idx)
            self.labels.append(label_mapping[orig_label])
        if not self.indices:
            raise RuntimeError("No test samples had labels present in the label mapping.")
        print(f"TestWithLabelsDataset: using {len(self.indices)} test samples with groundtruth labels.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        image, _ = self.base[base_idx]
        label = self.labels[idx]
        return image, label

def load_groundtruth(groundtruth_path: Path) -> Dict[str, int]:
    if not groundtruth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {groundtruth_path}")
    mapping: Dict[str, int] = {}
    with groundtruth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            image_name, label = parts
            mapping[Path(image_name).name] = int(label)
    if not mapping:
        raise ValueError(f"No entries found in ground truth file: {groundtruth_path}")
    return mapping

def train_finetune(model: nn.Module, backbone: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, epochs: int, head_lr: float, backbone_lr: float, output_dir: Path):
    criterion = nn.CrossEntropyLoss()
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters()]

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
    )

    best_acc = 0.0
    best_state = None
    best_path = output_dir / "best_dinov2_finetune_last2.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        # Use tqdm for training batches
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        val_top1, val_top5 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%")
        if val_top1 > best_acc:
            best_acc = val_top1
            best_state = model.state_dict()
            torch.save(best_state, best_path)
            print(f"  -> New best model saved to {best_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -----------------------
# MAIN
# -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_default_transform()
    train_dataset = PlantDataset(data_root=str(data_root), list_file=str(data_root / args.train_list), with_labels=True, transform=transform)
    train_indices, val_indices = stratified_split(train_dataset.labels, val_ratio=args.val_ratio, seed=args.seed)

    pin_memory = torch.cuda.is_available()
    train_subset_raw = Subset(train_dataset, train_indices)
    val_subset_raw = Subset(train_dataset, val_indices)

    raw_train_labels_full = np.array(train_dataset.labels, dtype=np.int64)
    raw_train_labels = raw_train_labels_full[train_indices]
    raw_val_labels = raw_train_labels_full[val_indices]
    label_mapping = build_label_mapping([raw_train_labels, raw_val_labels])

    mapping_path = output_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump({str(int(k)): int(v) for k, v in label_mapping.items()}, handle, indent=2)
    print(f"Saved label mapping to {mapping_path}")

    train_subset = RemappedDataset(train_subset_raw, label_mapping)
    val_subset = RemappedDataset(val_subset_raw, label_mapping)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # Build DINOv2 backbone & feature extractor
    feature_extractor = build_dinov2_backbone(str(data_root / args.checkpoint_path), device, args.dinov2_arch)
    backbone = feature_extractor.backbone

    # Freeze all params first
    for p in backbone.parameters():
        p.requires_grad = False

    # Unfreeze last 2 transformer blocks (if available)
    if hasattr(backbone, "blocks"):
        for block in backbone.blocks[-2:]:
            for p in block.parameters():
                p.requires_grad = True
        print("Unfroze last 2 DINOv2 blocks for fine-tuning.")
    else:
        print("Warning: backbone has no 'blocks' attribute; cannot unfreeze last 2 blocks.")

    # Build classifier & full model (DINO features -> MLP -> logits)
    if feature_extractor.embed_dim is None:
        raise RuntimeError("DINOv2 backbone missing embed_dim")
    input_dim = feature_extractor.embed_dim
    num_classes = len(label_mapping)
    classifier = build_mlp_classifier(input_dim, num_classes).to(device)

    # Wrap into a module that accepts images, extracts features, returns logits
    class DinoWithMLP(nn.Module):
        def __init__(self, feature_extractor: DinoFeatureExtractor, classifier: nn.Module):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.classifier = classifier
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.feature_extractor(x)
            logits = self.classifier(feats)
            return logits

    model = DinoWithMLP(feature_extractor, classifier).to(device)

    # Train (fine-tune)
    model = train_finetune(model=model, backbone=backbone, train_loader=train_loader, val_loader=val_loader,
                           device=device, epochs=args.epochs, head_lr=args.head_lr, backbone_lr=args.backbone_lr, output_dir=output_dir)

    # Load best & evaluate on test
    best_path = output_dir / "best_dinov2_finetune_last2.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from {best_path}")
    else:
        print("Best model not found; using current weights.")

    # Build test dataset (with groundtruth->remapped mapping)
    groundtruth = load_groundtruth(data_root / args.groundtruth)
    test_base = PlantDataset(data_root=str(data_root), list_file=str(data_root / args.test_list), with_labels=False, transform=transform)
    test_dataset = TestWithLabelsDataset(test_base, groundtruth, label_mapping)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # Pair/unpair mapping
    pair_class_ids_raw = load_class_id_list(data_root / args.class_with_pairs)
    unpair_class_ids_raw = load_class_id_list(data_root / args.class_without_pairs)
    pair_class_ids = {label_mapping[label] for label in pair_class_ids_raw if label in label_mapping}
    unpair_class_ids = {label_mapping[label] for label in unpair_class_ids_raw if label in label_mapping}
    if not pair_class_ids:
        print("[Warning] No overlapping classes found for --class-with-pairs.")
    if not unpair_class_ids:
        print("[Warning] No overlapping classes found for --class-without-pairs.")

    test_top1, test_top5, test_logits, test_targets = evaluate(model, test_loader, device, return_logits=True)

    def compute_subset_accuracy(logits: torch.Tensor, targets: torch.Tensor, subset_classes: Set[int]):
        if not subset_classes:
            return float("nan"), float("nan")
        mask = torch.tensor([t.item() in subset_classes for t in targets])
        if mask.sum() == 0:
            return float("nan"), float("nan")
        subset_logits = logits[mask]
        subset_targets = targets[mask]
        return topk_accuracy(subset_logits.to(device), subset_targets.to(device), topk=(1, 5))

    pair_top1, pair_top5 = compute_subset_accuracy(test_logits, test_targets, pair_class_ids)
    unpair_top1, unpair_top5 = compute_subset_accuracy(test_logits, test_targets, unpair_class_ids)

    print(f"\nTest Overall Top-1 accuracy: {test_top1:.2f}%")
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

    # Save final model
    final_path = output_dir / "final_dinov2_finetune_last2.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    main()
