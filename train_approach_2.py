import argparse
import io
import json
import math
import os
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline DINOv2 embedding extraction + MLP classifier for cross-domain plant ID."
    )
    parser.add_argument("--data-root", default=".", type=str, help="Project root directory.")
    parser.add_argument("--train-list", default="list/train.txt", type=str, help="Train list file.")
    parser.add_argument("--test-list", default="list/test.txt", type=str, help="Test list file.")
    parser.add_argument("--groundtruth", default="list/groundtruth.txt", type=str, help="Test GT file.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size for feature extraction.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs for MLP training.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for MLP training.")
    parser.add_argument("--output-dir", default="output_approach_2", type=str, help="Directory for outputs.")
    parser.add_argument(
        "--checkpoint-path",
        default="model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz",
        type=str,
        help="Path to plant-pretrained DINOv2 checkpoint.",
    )
    parser.add_argument("--val-ratio", default=0.2, type=float, help="Fraction of train data used for validation.")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers for embedding extraction.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--dinov2-arch",
        default="dinov2_vitb14_reg",
        type=str,
        help="TorchHub DINOv2 architecture identifier (e.g., dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg).",
    )
    parser.add_argument(
        "--class-with-pairs",
        default="list/class_with_pairs.txt",
        type=str,
        help="File listing class IDs that have herbarium-field image pairs (one ID per line).",
    )
    parser.add_argument(
        "--class-without-pairs",
        default="list/class_without_pairs.txt",
        type=str,
        help="File listing class IDs without herbarium-field image pairs (one ID per line).",
    )
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


class PlantDataset(Dataset):
    def __init__(self, data_root: str, list_file: str, with_labels: bool = True, transform=None) -> None:
        self.data_root = Path(data_root)
        self.list_file = Path(list_file)
        self.with_labels = with_labels
        self.transform = transform or build_default_transform()
        self.samples: List[Tuple[str, Optional[int]]] = []
        self._load_list()
    
    def _get_domain_from_path(self, rel_path: str) -> int:
        """Extract domain from file path: 0 = herbarium, 1 = field/photo."""
        path_lower = rel_path.lower()
        if "herbarium" in path_lower:
            return 0
        else:
            return 1  # photo/field or test images are considered field

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
        domain = self._get_domain_from_path(rel_path)
        if self.with_labels:
            return image, label, domain
        return image, rel_path, domain


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
        raise FileNotFoundError(f"Class list file not found: {path}")
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
    if not class_ids:
        raise ValueError(f"No class IDs parsed from {path}")
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


def compute_subset_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, valid_class_ids: Set[int], topk: Sequence[int] = (1, 5)
) -> Tuple[float, float]:
    if not valid_class_ids:
        return float("nan"), float("nan")
    mask = torch.tensor(
        [int(t.item()) in valid_class_ids for t in targets],
        device=targets.device,
        dtype=torch.bool,
    )
    if mask.sum().item() == 0:
        return float("nan"), float("nan")
    subset_logits = logits[mask]
    subset_targets = targets[mask]
    return topk_accuracy(subset_logits, subset_targets, topk)


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        pass

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    suffixes = checkpoint_path.suffixes
    if suffixes[-2:] not in [[".tar", ".gz"], [".tar", ".gzip"]] and checkpoint_path.suffix not in {".tgz", ".tar"}:
        raise RuntimeError(f"Unable to load checkpoint as torch.load and no tar archive detected: {checkpoint_path}")

    with tarfile.open(checkpoint_path, "r:*") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        preferred_suffixes = {".pt", ".pth", ".bin", ".pth.tar", ".pt.tar"}
        selected_member = None
        for member in members:
            name = member.name.lower()
            suffix = Path(member.name).suffix.lower()
            if any(
                name.endswith(ps) for ps in preferred_suffixes
            ) or suffix in preferred_suffixes or "state_dict" in Path(member.name).stem:
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
        # torch.load again using the in-memory buffer
    return torch.load(buffer, map_location="cpu")


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
                # Fall back to first value in dict
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
        print(
            f"[Warning] Unable to align pos_embed automatically (source tokens={num_src}, target tokens={num_dst})."
        )
    return state_dict


def build_dinov2_backbone(checkpoint_path: str, device: torch.device, arch: str) -> DinoFeatureExtractor:
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
    except Exception as err:
        raise RuntimeError(
            "Failed to instantiate DINOv2 backbone via torch.hub. "
            "Install the Dinov2 repo or provide an instantiated model manually."
        ) from err

    checkpoint = load_checkpoint_state(Path(checkpoint_path))
    state_dict = checkpoint
    # TODO: adjust key names below to match the actual checkpoint contents if different.
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

    backbone.eval()
    feature_extractor = DinoFeatureExtractor(backbone).to(device)
    return feature_extractor


def extract_embeddings(
    backbone: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract embeddings, labels, and domain information.
    
    Returns:
        embeddings: (N, D) array of features
        labels: (N,) array of labels or image paths
        domains: (N,) array of domain indicators (0=herbarium, 1=field)
    """
    backbone.eval()
    embeddings: List[torch.Tensor] = []
    targets: List = []
    domains: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device, non_blocking=True)
            outputs = backbone(images)
            embeddings.append(outputs.cpu())
            if len(batch) > 1:
                meta = batch[1]
                if torch.is_tensor(meta):
                    targets.append(meta.cpu())
                else:
                    targets.extend(meta)
            if len(batch) > 2:
                domain = batch[2]
                if torch.is_tensor(domain):
                    domains.append(domain.cpu())
                else:
                    domains.extend(domain)

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")
    emb_tensor = torch.cat(embeddings, dim=0)
    
    # Process domains
    domain_array = None
    if domains:
        first_domain = domains[0]
        if torch.is_tensor(first_domain):
            domain_array = torch.cat(domains, dim=0).numpy()
        else:
            domain_array = np.array(domains, dtype=np.int32)

    if not targets:
        return emb_tensor.numpy(), None, domain_array

    first_item = targets[0]
    if torch.is_tensor(first_item):
        target_tensor = torch.cat(targets, dim=0)
        return emb_tensor.numpy(), target_tensor.numpy(), domain_array

    # Assume metadata such as image paths (list of strings)
    return emb_tensor.numpy(), np.array(targets), domain_array


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


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, return_logits: bool = False
) -> Tuple[float, float]:
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            all_logits.append(logits)
            all_targets.append(labels)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    top1, top5 = topk_accuracy(logits_cat, targets_cat, topk=(1, 5))
    if return_logits:
        return top1, top5, logits_cat, targets_cat
    return top1, top5


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: Path,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_state = None
    best_path = output_dir / "best_mlp_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        val_top1, val_top5 = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%"
        )
        if val_top1 > best_acc:
            best_acc = val_top1
            best_state = model.state_dict()
            torch.save(best_state, best_path)
            print(f"  -> New best model saved to {best_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_default_transform()
    train_dataset = PlantDataset(
        data_root=str(data_root),
        list_file=str(data_root / args.train_list),
        with_labels=True,
        transform=transform,
    )
    train_indices, val_indices = stratified_split(train_dataset.labels, val_ratio=args.val_ratio, seed=args.seed)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_dataset = PlantDataset(
        data_root=str(data_root),
        list_file=str(data_root / args.test_list),
        with_labels=False,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    backbone = build_dinov2_backbone(str(data_root / args.checkpoint_path), device, args.dinov2_arch)
    train_embeddings, train_labels, train_domains = extract_embeddings(backbone, train_loader, device)
    val_embeddings, val_labels, val_domains = extract_embeddings(backbone, val_loader, device)
    test_embeddings, test_image_names, test_domains = extract_embeddings(backbone, test_loader, device)

    raw_train_labels = train_labels.astype(np.int64)
    raw_val_labels = val_labels.astype(np.int64)
    label_mapping = build_label_mapping([raw_train_labels, raw_val_labels])
    train_labels = remap_labels(raw_train_labels, label_mapping)
    val_labels = remap_labels(raw_val_labels, label_mapping)
    mapping_path = output_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump({str(int(k)): int(v) for k, v in label_mapping.items()}, handle, indent=2)
    print(f"Saved label mapping to {mapping_path}")

    # Save features in NPZ format as requested
    # Combine train and val for complete training set
    X_train = np.vstack([train_embeddings, val_embeddings]).astype(np.float32)
    y_train = np.concatenate([train_labels, val_labels]).astype(np.int64)
    domain_train = np.concatenate([train_domains, val_domains]).astype(np.int32)
    
    np.savez(
        output_dir / "features_train.npz",
        X_train=X_train,
        y_train=y_train,
        domain_train=domain_train
    )
    print(f"Saved features_train.npz with shapes: X_train={X_train.shape}, y_train={y_train.shape}, domain_train={domain_train.shape}")

    # Also save original separate files for backward compatibility
    np.save(output_dir / "train_embeddings.npy", train_embeddings)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "val_embeddings.npy", val_embeddings)
    np.save(output_dir / "val_labels.npy", val_labels)
    np.save(output_dir / "test_embeddings.npy", test_embeddings)
    np.save(output_dir / "test_image_names.npy", test_image_names)
    print(f"Saved embedding artifacts to {output_dir}")

    pair_class_ids_raw = load_class_id_list(data_root / args.class_with_pairs)
    unpair_class_ids_raw = load_class_id_list(data_root / args.class_without_pairs)
    pair_class_ids = {label_mapping[label] for label in pair_class_ids_raw if label in label_mapping}
    unpair_class_ids = {label_mapping[label] for label in unpair_class_ids_raw if label in label_mapping}
    if not pair_class_ids:
        print("[Warning] No overlapping classes found for --class-with-pairs.")
    if not unpair_class_ids:
        print("[Warning] No overlapping classes found for --class-without-pairs.")

    input_dim = train_embeddings.shape[1]
    num_classes = len(label_mapping)

    train_tensor_dataset = TensorDataset(
        torch.from_numpy(train_embeddings).float(), torch.from_numpy(train_labels).long()
    )
    val_tensor_dataset = TensorDataset(torch.from_numpy(val_embeddings).float(), torch.from_numpy(val_labels).long())

    emb_train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    emb_val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    classifier = build_mlp_classifier(input_dim, num_classes).to(device)
    classifier = train_classifier(
        classifier,
        emb_train_loader,
        emb_val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=output_dir,
    )

    val_top1, val_top5 = evaluate(classifier, emb_val_loader, device)
    print(f"Validation Top-1 accuracy: {val_top1:.2f}%")
    print(f"Validation Top-5 accuracy: {val_top5:.2f}%")

    groundtruth = load_groundtruth(data_root / args.groundtruth)
    image_names = [Path(name).name for name in test_image_names]
    valid_indices = [idx for idx, name in enumerate(image_names) if name in groundtruth]

    if not valid_indices:
        raise RuntimeError("No overlap between test image names and ground truth keys.")

    filtered_embeddings_list: List[np.ndarray] = []
    filtered_labels_list: List[int] = []
    filtered_raw_labels: List[int] = []
    for idx in valid_indices:
        image_name = image_names[idx]
        raw_label = groundtruth[image_name]
        mapped_label = label_mapping.get(raw_label)
        if mapped_label is None:
            continue
        filtered_embeddings_list.append(test_embeddings[idx])
        filtered_labels_list.append(mapped_label)
        filtered_raw_labels.append(raw_label)

    if not filtered_embeddings_list:
        raise RuntimeError("No test samples had labels present in the training label mapping.")

    filtered_embeddings = torch.from_numpy(np.stack(filtered_embeddings_list)).float()
    filtered_labels = torch.tensor(filtered_labels_list, dtype=torch.long)
    
    # Also filter test domains
    filtered_domains_list: List[int] = []
    for idx in valid_indices:
        image_name = image_names[idx]
        raw_label = groundtruth[image_name]
        mapped_label = label_mapping.get(raw_label)
        if mapped_label is None:
            continue
        filtered_domains_list.append(test_domains[idx])
    
    # Save test features in NPZ format
    X_test = np.stack(filtered_embeddings_list).astype(np.float32)
    y_test = np.array(filtered_labels_list, dtype=np.int64)
    domain_test = np.array(filtered_domains_list, dtype=np.int32)
    
    np.savez(
        output_dir / "features_test.npz",
        X_test=X_test,
        y_test=y_test,
        domain_test=domain_test
    )
    print(f"Saved features_test.npz with shapes: X_test={X_test.shape}, y_test={y_test.shape}, domain_test={domain_test.shape}")
    
    test_tensor_dataset = TensorDataset(filtered_embeddings, filtered_labels)
    test_loader_emb = DataLoader(
        test_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_top1, test_top5, test_logits, test_targets = evaluate(
        classifier, test_loader_emb, device, return_logits=True
    )
    pair_top1, pair_top5 = compute_subset_accuracy(test_logits, test_targets, pair_class_ids)
    unpair_top1, unpair_top5 = compute_subset_accuracy(test_logits, test_targets, unpair_class_ids)
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


if __name__ == "__main__":
    main()
