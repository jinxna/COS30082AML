# COS30082AML

Quick steps to run the GUI and use the trained models.

## 1) Environment
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install --upgrade pip
pip install -r requirements.txt
```
TensorFlow 2.15 is pinned for Approach 1 Keras models; keep that version to avoid load errors.

## 2) Run the Streamlit GUI
```bash
streamlit run plant_gui.py
```
In the sidebar pick your approach and verify the paths:
- **Approach 1 (Keras)**: `output_approach_1/final_model.keras`, mapping `output_approach_1/label_mapping.json`.
- **Approach 2 (DINO+MLP)**: `output_approach_2/best_mlp_classifier.pt`, mapping `output_approach_2/label_mapping.json`, checkpoint `model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz`.
- **Approach 3 Model 1**: `output_approach_3_model_1/best_dinov2_finetune_last2.pt`, mapping `output_approach_3_model_1/label_mapping.json`.
- **Approach 3 Model 2**: `output_approach_3_model_2/final_dinov2_finetune_last2.pt`, mapping `output_approach_3_model_2/label_mapping.json`.
- **Approach 3 Model 3**: `output_approach_3_model_3/best_mlp_classifier.pt`, mapping `output_approach_2/label_mapping.json`, checkpoint `model/dinov2_patch14_reg4_onlyclassifier_then_all-pytorch-default-v3.tar.gz`.

Leave `Species list` and `Train list` as `list/species_list.txt` and `list/train.txt` unless you moved them.

## 3) Train (optional)
- `train_approach_1.py` → outputs in `output_approach_1/` (saves Keras model + label mapping).
- `train_approach_2.py` → outputs in `output_approach_2/` (DINO embeddings + MLP).
- `train_approach_3_model_1.py` → finetuned DINO+MLP in `output_approach_3_model_1/`.
- `train_approach_3_model_2.py` → finetuned DINO+MLP in `output_approach_3_model_2/`.
- `train_approach_3_model_3.py` → MLP on precomputed embeddings in `output_approach_3_model_3/`.

## 4) Data layout
Expected folders at project root: `train/` (with `herbarium/` and `photo/` subfolders), `test/`, `list/` (includes `groundtruth.txt`, `species_list.txt`, `train.txt`, `test.txt`, class pair lists). Adjust paths in scripts if your data lives elsewhere.
