import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

from train_approach_2 import (
    build_default_transform,
    build_dinov2_backbone,
    build_mlp_classifier,
    DinoFeatureExtractor,
)


st.set_page_config(page_title="Cross-Domain Plant ID", page_icon="🌿", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');

.stApp {
    background: radial-gradient(circle at top, #0f172a 10%, #020617 70%);
    color: #f8fafc;
    font-family: 'Space Grotesk', sans-serif;
}

.result-card {
    background: linear-gradient(135deg, rgba(15,118,110,0.85), rgba(14,165,233,0.85));
    border-radius: 26px;
    padding: 28px;
    box-shadow: 0 30px 60px rgba(2,6,23,0.55);
    color: #f8fafc;
    animation: floatCard 8s cubic-bezier(.65,0,.35,1) infinite;
}

@keyframes floatCard {
    0%   { transform: translateY(0px); }
    45%  { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

.prediction-row {
    background: rgba(15,23,42,0.55);
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #e2e8f0;
}

.prediction-row span.rank {
    font-weight: 600;
    color: #38bdf8;
    margin-right: 10px;
}

.thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
}

.thumb-card {
    border-radius: 18px;
    overflow: hidden;
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(148,163,184,0.2);
    box-shadow: 0 20px 45px rgba(2,6,23,0.6);
}

.thumb-card img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    display: block;
}

.thumb-card div {
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #cbd5f5;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

APPROACH1_IMG_SIZE = 224
transform = build_default_transform()


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 20.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(x_norm, w_norm.t())


class MLPWithCosineHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cosine_scale: float = 20.0) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.head = CosineClassifier(256, num_classes, scale=cosine_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def _align_pos_embed(state_dict: Dict[str, torch.Tensor], backbone: nn.Module) -> Dict[str, torch.Tensor]:
    # Adjust position embedding shape mismatch if present.
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
        cls_token = cls_token[:, :1, :]
        patched = torch.cat([cls_token, src], dim=1)
        state_dict = dict(state_dict)
        state_dict["pos_embed"] = patched
    return state_dict


class DinoWithMLP(nn.Module):
    def __init__(self, feature_extractor: DinoFeatureExtractor, classifier: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits


class BackboneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


def _load_classifier_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and ("state_dict" in checkpoint or "model_state" in checkpoint):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state")
        metadata = {k: v for k, v in checkpoint.items() if k not in {"state_dict", "model_state"}}
    else:
        state_dict = checkpoint
        metadata = {}
    return state_dict, metadata


def _build_classifier_from_checkpoint(
    embed_dim: int,
    num_classes: int,
    state_dict,
    metadata,
    device: torch.device,
) -> nn.Module:
    arch = metadata.get("arch")
    if arch is None:
        arch = "mlp_cosine_head" if any(k.startswith("head.") for k in state_dict.keys()) else "mlp"
    cosine_scale = metadata.get("cosine_scale", 20.0)
    if arch == "mlp_cosine_head":
        classifier = MLPWithCosineHead(embed_dim, num_classes, cosine_scale=cosine_scale).to(device)
    else:
        classifier = build_mlp_classifier(embed_dim, num_classes).to(device)
    classifier.load_state_dict(state_dict, strict=False)
    if arch == "mlp_cosine_head" and hasattr(classifier, "head"):
        classifier.head.scale = float(cosine_scale)
    return classifier


def _align_finetune_state(state_dict: Dict[str, torch.Tensor], feature_extractor: DinoFeatureExtractor) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        return state_dict
    key_prefix = "feature_extractor.backbone."
    if any(k.startswith(key_prefix) for k in state_dict.keys()):
        backbone = getattr(feature_extractor, "backbone", feature_extractor)
        sub = {k.replace(key_prefix, ""): v for k, v in state_dict.items() if k.startswith(key_prefix)}
        sub_aligned = _align_pos_embed(sub, backbone)
        updated = dict(state_dict)
        for k, v in sub_aligned.items():
            updated[key_prefix + k] = v
        return updated
    return state_dict


def load_model_approach2(
    data_root: str,
    checkpoint_rel: str,
    classifier_rel: str,
    dinov2_arch: str,
    num_classes: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (Path(data_root) / checkpoint_rel).resolve()
    classifier_path = (Path(data_root) / classifier_rel).resolve()
    backbone = build_dinov2_backbone(str(checkpoint_path), device, dinov2_arch)
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(backbone, "backbone", None)
        embed_dim = getattr(embed_dim, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Unable to determine embedding dimension from backbone.")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    state_dict, metadata = _load_classifier_checkpoint(classifier_path, device)
    classifier = _build_classifier_from_checkpoint(
        embed_dim,
        num_classes,
        state_dict,
        metadata,
        device,
    )
    model = BackboneClassifier(backbone, classifier).to(device)
    model.eval()
    return model, device


def load_model_approach3(
    data_root: str,
    plant_checkpoint_rel: str,
    finetuned_rel: str,
    dinov2_arch: str,
    num_classes: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plant_checkpoint_path = (Path(data_root) / plant_checkpoint_rel).resolve()
    finetuned_path = (Path(data_root) / finetuned_rel).resolve()
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Finetuned checkpoint not found: {finetuned_path}")
    feature_extractor = build_dinov2_backbone(str(plant_checkpoint_path), device, dinov2_arch)
    embed_dim = getattr(feature_extractor, "embed_dim", None)
    if embed_dim is None and hasattr(feature_extractor, "backbone"):
        embed_dim = getattr(feature_extractor.backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Unable to determine embedding dimension from feature extractor.")
    classifier = build_mlp_classifier(embed_dim, num_classes).to(device)
    model = DinoWithMLP(feature_extractor, classifier).to(device)
    state_dict = torch.load(finetuned_path, map_location=device)
    state_dict = _align_finetune_state(state_dict, feature_extractor)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device


@st.cache_data(show_spinner=False)
def load_label_mapping(mapping_path: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping = {int(k): int(v) for k, v in data.items()}
    inverse = {v: k for k, v in mapping.items()}
    return mapping, inverse


@st.cache_data(show_spinner=False)
def load_species_lookup(species_file: str) -> Dict[int, str]:
    path = Path(species_file)
    if not path.exists():
        raise FileNotFoundError(f"Species list not found: {path}")
    lookup: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                class_id, name = line.split(";", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                class_id, name = parts
            try:
                lookup[int(class_id.strip())] = name.strip()
            except ValueError:
                continue
    return lookup


@st.cache_data(show_spinner=False)
def load_herbarium_index(train_list_path: str, data_root: str, max_per_class: int = 12) -> Dict[int, List[str]]:
    path = Path(data_root) / train_list_path
    if not path.exists():
        raise FileNotFoundError(f"Training list not found: {path}")
    root = Path(data_root)
    index: Dict[int, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rel_path, label = parts
            try:
                class_id = int(label)
            except ValueError:
                continue
            if "herbarium" not in rel_path.lower():
                continue
            abs_path = (root / rel_path).resolve()
            if not abs_path.exists():
                continue
            bucket = index.setdefault(class_id, [])
            if len(bucket) < max_per_class:
                bucket.append(str(abs_path))
    return index


@st.cache_resource(show_spinner=True)
def load_models(
    data_root: str,
    checkpoint_rel: str,
    classifier_rel: str,
    dinov2_arch: str,
    num_classes: int,
) -> Tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (Path(data_root) / checkpoint_rel).resolve()
    classifier_path = (Path(data_root) / classifier_rel).resolve()
    backbone = build_dinov2_backbone(str(checkpoint_path), device, dinov2_arch)
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(backbone.backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Unable to determine embedding dimension from backbone.")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    state_dict, metadata = _load_classifier_checkpoint(classifier_path, device)
    classifier = _build_classifier_from_checkpoint(
        embed_dim,
        num_classes,
        state_dict,
        metadata,
        device,
    )
    classifier.eval()
    backbone.eval()
    return backbone, classifier, device


def run_inference_torch(
    image: Image.Image,
    model: nn.Module,
    device,
    inverse_label_map: Dict[int, int],
    species_lookup: Dict[int, str],
    topk: int = 5,
) -> List[Dict]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
    topk = min(topk, probs.shape[0])
    values, indices = torch.topk(probs, k=topk)
    results = []
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        idx = int(idx)
        raw_class = inverse_label_map.get(idx, None)
        display_name = species_lookup.get(raw_class, "Unknown species")
        results.append(
            {
                "rank": rank,
                "mapped_id": idx,
                "class_id": raw_class,
                "species": display_name,
                "score": score,
            }
        )
    return results


def preprocess_approach1(image: Image.Image, image_size: int = APPROACH1_IMG_SIZE):
    if tf is None:
        raise RuntimeError("TensorFlow is not available. Install tensorflow to use Approach 1.")
    arr = np.array(image.convert("RGB"))
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    tensor = tf.image.resize(tensor, [image_size, image_size])
    tensor = tf.keras.applications.resnet50.preprocess_input(tensor)
    return tf.expand_dims(tensor, axis=0)


@st.cache_resource(show_spinner=True)
def load_approach1_model(data_root: str, keras_rel: str):
    if tf is None:
        raise ImportError("TensorFlow is not available. Install tensorflow to use Approach 1.")
    model_path = (Path(data_root) / keras_rel).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")
    # Load without compiling to avoid deserializing legacy compiled losses/optimizers
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def run_inference_approach1(
    image: Image.Image,
    model,
    inverse_label_map: Dict[int, int],
    species_lookup: Dict[int, str],
    topk: int = 5,
) -> List[Dict]:
    if tf is None:
        raise RuntimeError("TensorFlow is not available. Install tensorflow to use Approach 1.")
    tensor = preprocess_approach1(image)
    probs = model.predict(tensor, verbose=0)[0]
    topk = min(topk, probs.shape[0])
    indices = np.argsort(-probs)[:topk]
    results = []
    for rank, idx in enumerate(indices.tolist(), start=1):
        idx_int = int(idx)
        raw_class = inverse_label_map.get(idx_int, None)
        display_name = species_lookup.get(raw_class, "Unknown species")
        results.append(
            {
                "rank": rank,
                "mapped_id": idx_int,
                "class_id": raw_class,
                "species": display_name,
                "score": float(probs[idx_int]),
            }
        )
    return results


def render_predictions(predictions: List[Dict]):
    if not predictions:
        return
    top_pred = predictions[0]
    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:0.95rem;opacity:0.8;">Top prediction</div>
            <div style="font-size:2.2rem;font-weight:600;">{top_pred['species']}</div>
            <div style="font-size:1rem;margin-top:6px;">Class ID: {top_pred['class_id']}</div>
            <div style="font-size:0.9rem;margin-top:14px;">Confidence: {top_pred['score']*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Ranking")
    for pred in predictions:
        st.markdown(
            f"""
            <div class="prediction-row">
                <span class="rank">#{pred['rank']}</span>
                <strong>{pred['species']}</strong>
                <span style="float:right;">{pred['score']*100:.2f}%</span><br/>
                <small>Class ID: {pred['class_id']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_herbarium_gallery(class_id: int, herbarium_index: Dict[int, List[str]], limit: int):
    if class_id is None:
        st.info("No herbarium references available for this prediction.")
        return
    images = herbarium_index.get(class_id, [])
    if not images:
        st.warning("No herbarium sheets found for this class in the train set.")
        return
    st.markdown("### Herbarium references")
    columns = st.columns(min(limit, 3))
    for idx, image_path in enumerate(images[:limit]):
        col = columns[idx % len(columns)]
        with col:
            try:
                st.image(image_path, caption=Path(image_path).name, use_container_width=True)
            except Exception as err:
                st.warning(f"Unable to load image {image_path}: {err}")


def main() -> None:
    st.title("🌿 Cross-Domain Plant Species Identifier")
    st.caption("Upload a field photo to see the predicted species and matching herbarium sheets.")

    with st.sidebar:
        st.header("Configuration")
        model_type = st.selectbox(
            "Model type",
            [
                "Approach 1 (Keras ResNet50)",
                "Approach 2 (DINO embeddings + MLP)",
                "Approach 3 Model 1 (finetuned DINO+MLP)",
                "Approach 3 Model 2 (finetuned DINO+MLP)",
                "Approach 3 Model 3 (finetuned DINO+MLP)",
            ],
            index=2,
        )
        data_root = Path(st.text_input("Data root", ".")).expanduser().resolve()
        default_label = "outputs/label_mapping.json"
        if model_type.startswith("Approach 1"):
            default_label = "output_approach_1/label_mapping.json"
        elif model_type.startswith("Approach 2"):
            default_label = "output_approach_2/label_mapping.json"
        elif "Model 1" in model_type:
            default_label = "output_approach_3_model_1/label_mapping.json"
        elif "Model 2" in model_type:
            default_label = "output_approach_3_model_2/label_mapping.json"
        elif "Model 3" in model_type:
            # Approach 3 Model 3 trains on precomputed embeddings; reuse Approach 2 mapping by default.
            default_label = "output_approach_2/label_mapping.json"
        label_mapping_rel = st.text_input("Label mapping", default_label)
        species_list_rel = st.text_input("Species list", "list/species_list.txt")
        train_list_rel = st.text_input("Train list", "list/train.txt")
        if model_type in (
            "Approach 2 (DINO embeddings + MLP)",
            "Approach 3 Model 3 (finetuned DINO+MLP)",
        ):
            checkpoint_rel = st.text_input(
                "Plant DINO checkpoint",
                "model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz",
            )
            classifier_default = (
                "output_approach_2/best_mlp_classifier.pt"
                if model_type.startswith("Approach 2")
                else "output_approach_3_model_3/best_mlp_classifier.pt"
            )
            classifier_rel = st.text_input("MLP weights", classifier_default)
            dinov2_arch = st.selectbox(
                "Backbone",
                ["dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"],
                index=0,
            )
            device_label = f"{'GPU' if torch.cuda.is_available() else 'CPU'} (PyTorch)"
        elif model_type.startswith("Approach 1"):
            keras_rel = st.text_input("Keras model (.keras)", "output_approach_1/final_model.keras")
            if tf is None or not hasattr(tf, "config"):
                device_label = "TensorFlow not available"
            else:
                tf_gpus = tf.config.list_physical_devices("GPU")
                device_label = f"{'GPU' if tf_gpus else 'CPU'} (TensorFlow)"
        else:
            plant_checkpoint_rel = st.text_input(
                "Plant DINO checkpoint",
                "model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz",
            )
            finetuned_default = "output_approach_3_model_1/best_dinov2_finetune_last2.pt"
            if "Model 2" in model_type:
                finetuned_default = "output_approach_3_model_2/final_dinov2_finetune_last2.pt"
            finetuned_rel = st.text_input("Finetuned weights", finetuned_default)
            dinov2_arch = st.selectbox(
                "Backbone",
                ["dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"],
                index=0,
            )
            device_label = f"{'GPU' if torch.cuda.is_available() else 'CPU'} (PyTorch)"
        herbarium_limit = st.slider("Herbarium references", 2, 6, 3)
        topk = st.slider("Top-k predictions", 3, 8, 5)
        st.markdown(f"**Device:** {device_label}")

    uploaded = st.file_uploader("Upload a field image", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            preview = Image.open(uploaded).convert("RGB")
            st.image(preview, caption="Field image", use_container_width=True)
        except Exception as err:
            st.error(f"Failed to read image: {err}")
            return
    else:
        st.info("Select a plant photo to begin.")
        return

    try:
        label_mapping, inverse_label_map = load_label_mapping(str(data_root / label_mapping_rel))
    except Exception as err:
        st.error(f"Label mapping error: {err}")
        return

    try:
        species_lookup = load_species_lookup(str(data_root / species_list_rel))
    except Exception as err:
        st.error(f"Species list error: {err}")
        return

    try:
        herbarium_index = load_herbarium_index(train_list_rel, str(data_root))
    except Exception as err:
        st.warning(f"Herbarium index warning: {err}")
        herbarium_index = {}

    if model_type.startswith("Approach 1"):
        if tf is None:
            st.error("TensorFlow is not available. Install tensorflow to use Approach 1.")
            return
        try:
            keras_model = load_approach1_model(str(data_root), keras_rel)
        except Exception as err:
            st.error(f"Model loading error: {err}")
            return
        model_classes = None
        try:
            if hasattr(keras_model, "output_shape") and keras_model.output_shape:
                model_classes = keras_model.output_shape[-1]
        except Exception:
            model_classes = None
        if model_classes and model_classes != len(inverse_label_map):
            st.warning(
                f"Label mapping has {len(inverse_label_map)} classes but model outputs {model_classes}; check mapping order."
            )
        with st.spinner("Analyzing plant traits..."):
            predictions = run_inference_approach1(
                preview,
                keras_model,
                inverse_label_map,
                species_lookup,
                topk=topk,
            )
    elif model_type.startswith("Approach 2") or "Model 3" in model_type:
        try:
            model, device = load_model_approach2(
                str(data_root),
                checkpoint_rel,
                classifier_rel,
                dinov2_arch,
                num_classes=len(label_mapping),
            )
        except Exception as err:
            st.error(f"Model loading error: {err}")
            return
        with st.spinner("Analyzing plant traits..."):
            predictions = run_inference_torch(
                preview,
                model,
                device,
                inverse_label_map,
                species_lookup,
                topk=topk,
            )
    else:
        try:
            model, device = load_model_approach3(
                str(data_root),
                plant_checkpoint_rel,
                finetuned_rel,
                dinov2_arch,
                num_classes=len(label_mapping),
            )
        except Exception as err:
            st.error(f"Model loading error: {err}")
            return
        with st.spinner("Analyzing plant traits..."):
            predictions = run_inference_torch(
                preview,
                model,
                device,
                inverse_label_map,
                species_lookup,
                topk=topk,
            )

    render_predictions(predictions)
    if predictions:
        render_herbarium_gallery(predictions[0]["class_id"], herbarium_index, herbarium_limit)


if __name__ == "__main__":
    main()
