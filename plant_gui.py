import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import streamlit as st
import torch

from train_dinov2_baseline import (
    build_default_transform,
    build_dinov2_backbone,
    build_mlp_classifier,
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

transform = build_default_transform()


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
    checkpoint_path = str((Path(data_root) / checkpoint_rel).resolve())
    classifier_path = (Path(data_root) / classifier_rel).resolve()
    backbone = build_dinov2_backbone(checkpoint_path, device, dinov2_arch)
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(backbone.backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Unable to determine embedding dimension from backbone.")
    classifier = build_mlp_classifier(embed_dim, num_classes).to(device)
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    state = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(state)
    classifier.eval()
    backbone.eval()
    return backbone, classifier, device


def run_inference(
    image: Image.Image,
    backbone,
    classifier,
    device,
    inverse_label_map: Dict[int, int],
    species_lookup: Dict[int, str],
    topk: int = 5,
) -> List[Dict]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = backbone(tensor)
        logits = classifier(embedding)
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
        data_root = Path(st.text_input("Data root", ".")).expanduser().resolve()
        checkpoint_rel = st.text_input(
            "Plant DINO checkpoint",
            "model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz",
        )
        classifier_rel = st.text_input("MLP weights", "outputs/best_mlp_classifier.pt")
        label_mapping_rel = st.text_input("Label mapping", "outputs/label_mapping.json")
        species_list_rel = st.text_input("Species list", "list/species_list.txt")
        train_list_rel = st.text_input("Train list", "list/train.txt")
        dinov2_arch = st.selectbox(
            "Backbone",
            ["dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"],
            index=0,
        )
        herbarium_limit = st.slider("Herbarium references", 2, 6, 3)
        topk = st.slider("Top-k predictions", 3, 8, 5)
        st.markdown(
            f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}",
        )

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

    try:
        backbone, classifier, device = load_models(
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
        predictions = run_inference(
            preview,
            backbone,
            classifier,
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
