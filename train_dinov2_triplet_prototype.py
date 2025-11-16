#!/usr/bin/env python3
"""
Approach 3 (Option 1) script: train a projection head with triplet loss on top of a frozen
plant-pretrained DINOv2 backbone, build class prototypes from herbarium images, and evaluate
test images with a nearest-prototype classifier.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.serialization  # type: ignore[attr-defined]

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

torch.serialization.add_safe_globals([argparse.Namespace])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train triplet + prototype head on top of DINOv2.")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory for dataset and metadata files.")
    parser.add_argument("--train-list", type=str, default="list/train.txt", help="Path to train.txt relative to data root.")
    parser.add_argument("--test-list", type=str, default="list/test.txt", help="Path to test.txt relative to data root.")
    parser.add_argument(
        "--groundtruth",
        type=str,
        default="list/groundtruth.txt",
        help="Path to groundtruth labels for the test set (relative to data root).",
    )
    parser.add_argument(
        "--class-with-pairs",
        type=str,
        default="list/class_with_pairs.txt",
        help="File containing class IDs that have herbarium + field training images.",
    )
    parser.add_argument(
        "--class-without-pairs",
        type=str,
        default="list/class_without_pairs.txt",
        help="File containing class IDs that only have herbarium images.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz",
        help="Path to the plant-pretrained DINOv2 checkpoint (relative to data root).",
    )
    parser.add_argument(
        "--dinov2-arch",
        type=str,
        default="dinov2_vitb14_reg",
        help="TorchHub identifier for the DINOv2 backbone (e.g., dinov2_vitb14_reg).",
    )
    parser.add_argument("--image-size", type=int, default=518, help="Resize & crop size for the DINOv2 backbone.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for triplet batches.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for prototype/test evaluation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs for the projection head.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the projection head optimizer.")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for the triplet loss.")
    parser.add_argument("--proj-dim", type=int, default=256, help="Output dimension of the projection head.")
    parser.add_argument("--output-dir", type=str, default="outputs_triplet", help="Directory to store artifacts.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_default_transform(image_size: int = 518) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_domain_from_path(path: str) -> str:
    # Adjust this rule if your directory names differ; we assume "herbarium" vs "field" substrings.
    return "herbarium" if "herbarium" in path.lower() else "field"


class PlantDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        list_file: Path,
        with_labels: bool,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.list_file = Path(list_file)
        self.with_labels = with_labels
        self.transform = transform or build_default_transform()
        self.samples: List[Dict[str, Optional[str]]] = []
        self._load_list()

    def _load_list(self) -> None:
        if not self.list_file.exists():
            raise FileNotFoundError(f"List file not found: {self.list_file}")
        with self.list_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                rel_path = parts[0]
                label = None
                if self.with_labels:
                    if len(parts) < 2:
                        raise ValueError(f"Unable to parse class ID from line: {line}")
                    label = int(parts[1])
                full_path = self.data_root / rel_path
                domain = infer_domain_from_path(rel_path)
                self.samples.append(
                    {
                        "path": full_path,
                        "relative_path": rel_path,
                        "label": label,
                        "domain": domain,
                        "image_name": Path(rel_path).name,
                    }
                )
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.list_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        tensor = self.transform(image)
        if self.with_labels:
            return tensor, int(sample["label"]), str(sample["domain"]), str(sample["image_name"])
        return tensor, str(sample["image_name"])


def collate_labeled_batch(
    batch: Sequence[Tuple[torch.Tensor, int, str, str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    images, labels, domains, names = zip(*batch)
    image_tensor = torch.stack(list(images), dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, label_tensor, list(domains), list(names)


def collate_unlabeled_batch(batch: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    images, names = zip(*batch)
    image_tensor = torch.stack(list(images), dim=0)
    return image_tensor, list(names)


def load_class_id_file(path: Path) -> Set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Class ID file not found: {path}")
    class_ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            class_ids.add(int(stripped.split()[0]))
    return class_ids


def load_class_lists(class_with_pairs: Path, class_without_pairs: Path) -> Tuple[Set[int], Set[int]]:
    pair_ids = load_class_id_file(class_with_pairs)
    unpair_ids = load_class_id_file(class_without_pairs)
    return pair_ids, unpair_ids


def load_groundtruth_map(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Groundtruth file not found: {path}")
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            image_name = Path(parts[0]).name  # normalize to match dataset image_name
            mapping[image_name] = int(parts[1])
    if not mapping:
        raise RuntimeError(f"No entries found in groundtruth file: {path}")
    return mapping


class DinoFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "embed_dim", None)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone.forward_features(images) if hasattr(self.backbone, "forward_features") else self.backbone(images)
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


def _align_pos_embed(state_dict: Dict[str, torch.Tensor], backbone: nn.Module) -> Dict[str, torch.Tensor]:
    if "pos_embed" not in state_dict or not hasattr(backbone, "pos_embed"):
        return state_dict
    src = state_dict["pos_embed"]
    dst = backbone.pos_embed
    if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
        return state_dict
    if src.shape == dst.shape or src.ndim != 3 or dst.ndim != 3:
        return state_dict
    num_src = src.shape[1]
    num_dst = dst.shape[1]
    if num_src == num_dst - 1:
        cls_token = state_dict.get("cls_token")
        if not isinstance(cls_token, torch.Tensor):
            cls_token = dst[:, :1, :].clone()
        elif cls_token.ndim == 2:
            cls_token = cls_token.unsqueeze(0)
        state_dict["pos_embed"] = torch.cat([cls_token[:, :1, :], src], dim=1)
        print("[Info] Adjusted pos_embed by prepending CLS token to match target shape.")
    elif num_src == num_dst + 1:
        state_dict["pos_embed"] = src[:, 1:, :]
        print("[Info] Adjusted pos_embed by removing extra token to match target shape.")
    else:
        print(f"[Warning] Unable to align pos_embed automatically (source tokens={num_src}, target tokens={num_dst}).")
    return state_dict


def build_dinov2_backbone(checkpoint_path: str, device: torch.device, arch: str) -> DinoFeatureExtractor:
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
    except Exception as err:  # pragma: no cover - depends on hub availability
        raise RuntimeError("Failed to instantiate DINOv2 backbone via torch.hub.") from err

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

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    feature_extractor = DinoFeatureExtractor(backbone).to(device)
    return feature_extractor


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def build_projection_head(in_dim: int, proj_dim: int) -> ProjectionHead:
    return ProjectionHead(in_dim, proj_dim)


def build_class_domain_indices(
    dataset: PlantDataset,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    class_to_field: Dict[int, List[int]] = defaultdict(list)
    class_to_herbarium: Dict[int, List[int]] = defaultdict(list)
    class_to_all: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        label = sample["label"]
        domain = sample["domain"]
        if label is None:
            continue
        class_to_all[label].append(idx)
        if domain == "herbarium":
            class_to_herbarium[label].append(idx)
        else:
            class_to_field[label].append(idx)
    return class_to_field, class_to_herbarium, class_to_all


class TripletDataset(Dataset):
    def __init__(
        self,
        dataset: PlantDataset,
        class_to_field: Dict[int, List[int]],
        class_to_herbarium: Dict[int, List[int]],
        class_to_all: Dict[int, List[int]],
        paired_class_ids: Set[int],
    ) -> None:
        self.dataset = dataset
        self.class_to_field = class_to_field
        self.class_to_herbarium = class_to_herbarium
        self.class_to_all = class_to_all
        self.valid_classes = [
            cid for cid in sorted(paired_class_ids) if class_to_field.get(cid) and class_to_herbarium.get(cid)
        ]
        if not self.valid_classes:
            raise RuntimeError("No paired classes contain both field and herbarium samples for triplet mining.")

        self.anchor_indices: List[int] = []
        for cid in self.valid_classes:
            self.anchor_indices.extend(class_to_field[cid])

        self.neg_classes_by_class: Dict[int, List[int]] = {}
        all_classes = [cid for cid, indices in class_to_all.items() if indices]
        for cid in self.valid_classes:
            neg_candidates = [other for other in all_classes if other != cid]
            if not neg_candidates:
                raise RuntimeError("Unable to sample negatives; dataset must contain at least two classes.")
            self.neg_classes_by_class[cid] = neg_candidates

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx = self.anchor_indices[index]
        anchor_label = int(self.dataset.samples[anchor_idx]["label"])
        pos_idx = random.choice(self.class_to_herbarium[anchor_label])
        neg_class = random.choice(self.neg_classes_by_class[anchor_label])
        neg_idx = random.choice(self.class_to_all[neg_class])

        anchor_image = self.dataset[anchor_idx][0]
        positive_image = self.dataset[pos_idx][0]
        negative_image = self.dataset[neg_idx][0]
        return anchor_image, positive_image, negative_image


def build_triplet_dataloader(
    train_dataset: PlantDataset,
    paired_class_ids: Set[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    class_to_field, class_to_herbarium, class_to_all = build_class_domain_indices(train_dataset)
    triplet_dataset = TripletDataset(train_dataset, class_to_field, class_to_herbarium, class_to_all, paired_class_ids)
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(triplet_dataset) >= batch_size,
    )
    return dataloader


def train_triplet_head(
    backbone: nn.Module,
    projection_head: nn.Module,
    triplet_loader: DataLoader,
    device: torch.device,
    epochs: int,
    margin: float,
    lr: float,
    output_dir: Path,
) -> Path:
    projection_head.train()
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(projection_head.parameters(), lr=lr)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0
        for images_a, images_p, images_n in triplet_loader:
            images_a = images_a.to(device, non_blocking=True)
            images_p = images_p.to(device, non_blocking=True)
            images_n = images_n.to(device, non_blocking=True)

            with torch.no_grad():
                feat_a = backbone(images_a)
                feat_p = backbone(images_p)
                feat_n = backbone(images_n)

            z_a = projection_head(feat_a)
            z_p = projection_head(feat_p)
            z_n = projection_head(feat_n)

            loss = criterion(z_a, z_p, z_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch:02d}/{epochs} - Triplet loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in projection_head.state_dict().items()}

    best_path = output_dir / "best_projection_head.pt"
    if best_state is not None:
        projection_head.load_state_dict(best_state, strict=True)
        torch.save(best_state, best_path)
        print(f"Saved best projection head to {best_path}")
    else:
        print("[Warning] No training batches executed; projection head was not updated.")
    return best_path


def compute_class_prototypes(
    backbone: nn.Module,
    projection_head: nn.Module,
    train_dataset: PlantDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> Tuple[torch.Tensor, List[int]]:
    projection_head.eval()
    backbone.eval()
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_labeled_batch,
    )
    class_embeddings: Dict[int, List[torch.Tensor]] = defaultdict(list)

    with torch.no_grad():
        for images, labels, domains, _ in loader:
            images = images.to(device, non_blocking=True)
            feats = backbone(images)
            embeds = projection_head(feats)
            for emb, label, domain in zip(embeds.cpu(), labels.tolist(), domains):
                if domain != "herbarium":
                    continue
                class_embeddings[label].append(emb)

    if not class_embeddings:
        raise RuntimeError("No herbarium embeddings found; cannot compute class prototypes.")

    sorted_class_ids = sorted(class_embeddings.keys())
    prototypes = []
    for cid in sorted_class_ids:
        stack = torch.stack(class_embeddings[cid], dim=0)
        prototypes.append(stack.mean(dim=0))
    prototype_tensor = torch.stack(prototypes, dim=0)
    proto_path = output_dir / "class_prototypes.pt"
    torch.save(
        {"prototypes": prototype_tensor, "class_ids": sorted_class_ids, "proj_dim": prototype_tensor.shape[1]},
        proto_path,
    )
    mapping_path = output_dir / "prototype_class_ids.json"
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump({str(idx): int(cid) for idx, cid in enumerate(sorted_class_ids)}, handle, indent=2)
    print(f"Saved class prototypes to {proto_path}")
    print(f"Saved prototype class mapping to {mapping_path}")
    return prototype_tensor.to(device), sorted_class_ids


def evaluate_prototypes(
    backbone: nn.Module,
    projection_head: nn.Module,
    test_loader: DataLoader,
    prototypes: torch.Tensor,
    prototype_class_ids: List[int],
    paired_class_ids: Set[int],
    unpaired_class_ids: Set[int],
    groundtruth_map: Dict[str, int],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    projection_head.eval()
    backbone.eval()
    prototypes = prototypes.to(device)
    topk = min(5, prototypes.shape[0])

    stats = {
        "overall": {"top1_correct": 0, "top5_correct": 0, "total": 0},
        "paired": {"top1_correct": 0, "top5_correct": 0, "total": 0},
        "unpaired": {"top1_correct": 0, "top5_correct": 0, "total": 0},
    }

    def update_stats(bucket: Dict[str, int], top_indices: Sequence[int], true_label: int) -> None:
        bucket["total"] += 1
        if top_indices and top_indices[0] == true_label:
            bucket["top1_correct"] += 1
        if true_label in top_indices[: min(5, len(top_indices))]:
            bucket["top5_correct"] += 1

    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device, non_blocking=True)
            feats = backbone(images)
            embeds = projection_head(feats)
            scores = torch.matmul(embeds, prototypes.t())
            values, indices = torch.topk(scores, k=topk, dim=1, largest=True)

            for sample_idx, name in enumerate(image_names):
                if name not in groundtruth_map:
                    print(f"[Warning] Groundtruth missing for {name}; skipping.")
                    continue
                candidate_indices = indices[sample_idx].tolist()
                pred_classes = [prototype_class_ids[idx] for idx in candidate_indices]
                true_label = groundtruth_map[name]
                update_stats(stats["overall"], pred_classes, true_label)
                if true_label in paired_class_ids:
                    update_stats(stats["paired"], pred_classes, true_label)
                if true_label in unpaired_class_ids:
                    update_stats(stats["unpaired"], pred_classes, true_label)

    metrics: Dict[str, Dict[str, float]] = {}
    for key, bucket in stats.items():
        total = bucket["total"]
        top1 = bucket["top1_correct"] / total if total else 0.0
        top5 = bucket["top5_correct"] / total if total else 0.0
        metrics[key] = {"top1": top1, "top5": top5, "num_samples": total}
        print(f"Test {key.capitalize()} Top-1 accuracy: {top1*100:.2f}% ({bucket['top1_correct']}/{total})")
        print(f"Test {key.capitalize()} Top-5 accuracy: {top5*100:.2f}% ({bucket['top5_correct']}/{total})")
    return metrics


def save_metrics(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> Path:
    metrics_path = output_dir / "metrics_triplet_prototype.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")
    return metrics_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_default_transform(args.image_size)
    train_dataset = PlantDataset(
        data_root=data_root,
        list_file=data_root / args.train_list,
        with_labels=True,
        transform=transform,
    )
    test_dataset = PlantDataset(
        data_root=data_root,
        list_file=data_root / args.test_list,
        with_labels=False,
        transform=transform,
    )

    paired_class_ids, unpaired_class_ids = load_class_lists(
        data_root / args.class_with_pairs, data_root / args.class_without_pairs
    )
    groundtruth_map = load_groundtruth_map(data_root / args.groundtruth)

    triplet_loader = build_triplet_dataloader(
        train_dataset=train_dataset,
        paired_class_ids=paired_class_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    backbone = build_dinov2_backbone(str(data_root / args.checkpoint_path), device, args.dinov2_arch)
    if not hasattr(backbone, "embed_dim") or backbone.embed_dim is None:
        raise AttributeError("DINOv2 backbone does not expose embed_dim; cannot size projection head.")
    projection_head = build_projection_head(backbone.embed_dim, args.proj_dim).to(device)

    train_triplet_head(
        backbone=backbone,
        projection_head=projection_head,
        triplet_loader=triplet_loader,
        device=device,
        epochs=args.epochs,
        margin=args.margin,
        lr=args.lr,
        output_dir=output_dir,
    )

    prototypes, prototype_class_ids = compute_class_prototypes(
        backbone=backbone,
        projection_head=projection_head,
        train_dataset=train_dataset,
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_unlabeled_batch,
    )
    metrics = evaluate_prototypes(
        backbone=backbone,
        projection_head=projection_head,
        test_loader=test_loader,
        prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
        paired_class_ids=paired_class_ids,
        unpaired_class_ids=unpaired_class_ids,
        groundtruth_map=groundtruth_map,
        device=device,
    )
    save_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
