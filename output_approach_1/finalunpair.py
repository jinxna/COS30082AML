#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Baseline-1  ResNet50  –  minimal unpaired fix + herbarium colour-jitter
# ------------------------------------------------------------------
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import time
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Hyperparameters
SEED                = 42
IMG_SIZE            = 224
BATCH_SIZE          = 32
VAL_SPLIT_PER_CLASS = 0.15
WARMUP_EPOCHS       = 5
FINETUNE_EPOCHS     = 20
FREEZE_UP_TO        = 170
INIT_LR_HEAD        = 5e-4
INIT_LR_FT          = 1e-4
LABEL_SMOOTHING     = 0.1
WEIGHT_DECAY        = 5e-4
DROPOUT_RATE        = 0.6
AUG_JITTER          = 0.15
AUG_RANDCROP        = 0.15
AUG_ROTATE          = 0.10
USE_CLASS_WEIGHTS   = True
EARLY_STOP_PATIENCE = 10
LR_REDUCE_PATIENCE  = 4

# Paths
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
TRAIN_DIR = ROOT / "train"
TEST_DIR  = ROOT / "test"
LIST_DIR  = ROOT / "list"
GROUNDTRUTH = LIST_DIR / "groundtruth.txt"

RESULTS   = ROOT / "results"
CKPT_DIR  = RESULTS / "checkpoints"
PLOTS_DIR = RESULTS / "plots"
REPORTS   = RESULTS / "reports"
NPY_DIR   = RESULTS / "npy"
for p in [CKPT_DIR, PLOTS_DIR, REPORTS, NPY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED); tf.random.set_seed(SEED)

# Data Loading
def _all_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG")
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def _gather_train():
    hdir = TRAIN_DIR / "herbarium"; pdir = TRAIN_DIR / "photo"
    domain = {"herbarium": set(), "photo": set()}
    all_ids = set()
    for dname, src in (("herbarium", hdir), ("photo", pdir)):
        if not src.exists(): continue
        for cls in src.iterdir():
            if cls.is_dir() and cls.name.isdigit():
                domain[dname].add(cls.name); all_ids.add(cls.name)
    class_ids = sorted(all_ids)
    id2idx = {c: i for i, c in enumerate(class_ids)}

    # unpaired = missing either domain in training
    unpaired_mask = np.array([not (c in domain["herbarium"] and c in domain["photo"])
                              for c in class_ids], dtype=bool)

    files, labels, domain_flag = [], [], []
    for dname, src in (("herbarium", hdir), ("photo", pdir)):
        if not src.exists(): continue
        for cls in src.iterdir():
            if not cls.is_dir() or cls.name not in id2idx: continue
            idx = id2idx[cls.name]
            for im in _all_images(cls):
                files.append(im)
                labels.append(idx)
                domain_flag.append(1 if dname == "herbarium" else 0)
    paired_mask = np.array([c in domain["photo"] for c in class_ids], dtype=bool)
    return (np.array(files), np.array(labels, np.int32), np.array(domain_flag, np.int32),
            class_ids, len(class_ids), id2idx, paired_mask, unpaired_mask)

def _stratified_split(files, labels, df, val_frac=0.15):
    rng = np.random.default_rng(SEED)
    per_class = defaultdict(list)
    for i, y in enumerate(labels):
        per_class[y].append(i)
    tr, va = [], []
    for idxs in per_class.values():
        idxs = rng.permutation(idxs)
        n = max(1, int(len(idxs) * val_frac))
        va.extend(idxs[:n]); tr.extend(idxs[n:])
    tr, va = np.array(tr), np.array(va)
    return (files[tr], labels[tr], df[tr],
            files[va], labels[va], df[va])

# Augmentation and Preprocessing
def _decode(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img

def _aug(img, training=False, is_herbarium=False):
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    if training:
        img = tf.image.random_flip_left_right(img)
        if AUG_ROTATE > 0:
            angle = tf.random.uniform([], -AUG_ROTATE*np.pi, AUG_ROTATE*np.pi)
            img = tf.image.rot90(img, k=tf.cast(angle/(np.pi/2), tf.int32))
        # --- cheap domain-shift band-aid ---
        if is_herbarium:
            img = tf.image.random_brightness(img, 0.20)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
        if AUG_JITTER > 0:
            img = tf.image.random_brightness(img, AUG_JITTER)
            img = tf.image.random_contrast(img, 1-AUG_JITTER, 1+AUG_JITTER)
            img = tf.image.random_saturation(img, 1-AUG_JITTER, 1+AUG_JITTER)
            img = tf.image.random_hue(img, AUG_JITTER/2)
        img = tf.clip_by_value(img, 0, 255)
    return img

def _preproc(path, y, d, training=False):
    img = _decode(path)
    img = _aug(img, training, is_herbarium=tf.cast(d, tf.bool))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, y

def make_ds(files, labels, df, training=False):
    ds = tf.data.Dataset.from_tensor_slices((files.astype(str), labels, df))
    if training: ds = ds.shuffle(min(5000, len(files)), seed=SEED)
    ds = ds.map(lambda p, y, d: _preproc(p, y, d, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Model
def build_model(n_classes):
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                         input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    logits = tf.keras.layers.Dense(n_classes, activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  bias_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    return tf.keras.Model(base.input, logits), base

# Train
top1 = tf.keras.metrics.SparseCategoricalAccuracy(name='top1')
top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')

def make_loss_fn(num_classes, smoothing=0.0):
    if smoothing == 0.0:
        return tf.keras.losses.SparseCategoricalCrossentropy()
    ce = tf.keras.losses.CategoricalCrossentropy()
    @tf.function
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        y_true_oh = y_true_oh * (1.0 - smoothing) + smoothing / num_classes
        return ce(y_true_oh, y_pred)
    return loss

def compile_head(model, n_classes):
    loss = make_loss_fn(n_classes, LABEL_SMOOTHING)
    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR_HEAD), loss=loss, metrics=[top1, top5])

def compile_ft(model, n_classes):
    loss = make_loss_fn(n_classes, LABEL_SMOOTHING)
    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR_FT), loss=loss, metrics=[top1, top5])

def train(model_tuple, ds_tr, ds_val, n_classes):
    model, base = model_tuple
    base.trainable = False
    compile_head(model, n_classes)
    cbs = [tf.keras.callbacks.EarlyStopping(patience=EARLY_STOP_PATIENCE, restore_best_weights=True,
                                           monitor='val_top1', mode='max', verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=LR_REDUCE_PATIENCE,
                                               monitor='val_top1', mode='max', verbose=1)]
    print('[phase-1] head only …')
    hist1 = model.fit(ds_tr, epochs=WARMUP_EPOCHS, validation_data=ds_val, callbacks=cbs, verbose=1)
    base.trainable = True
    for i, L in enumerate(base.layers): L.trainable = (i >= FREEZE_UP_TO)
    compile_ft(model, n_classes)
    print('[phase-2] fine-tune …')
    hist2 = model.fit(ds_tr, epochs=FINETUNE_EPOCHS, validation_data=ds_val, callbacks=cbs, verbose=1)
    hist = {k: [float(x) for x in hist1.history.get(k, [])] + [float(x) for x in hist2.history.get(k, [])]
            for k in set(hist1.history) | set(hist2.history)}
    return model, hist

# Test Set Loader
def _load_groundtruth():
    if not GROUNDTRUTH.exists():
        print('[error] groundtruth.txt missing'); return [], []
    names, cls = [], []
    with open(GROUNDTRUTH, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip().replace(',', ' ')
            parts = [p for p in line.split() if p]
            if len(parts) < 2: continue
            a, b = parts[0], parts[1]
            a = Path(a).name
            if '.' in a.lower():
                names.append(a.lower()); cls.append(b)
            else:
                b = Path(b).name
                names.append(b.lower()); cls.append(a)
    print(f'[test] groundtruth – {len(names)} entries')
    return names, cls

def _find_test_files(names, cls_ids, id2idx):
    if not TEST_DIR.exists():
        print('[error] test/ folder missing'); return np.array([]), np.array([], np.int32)
    files_by_name = {p.name.lower(): str(p) for p in TEST_DIR.rglob('*')
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    paths, labels = [], []
    for img_name, c in zip(names, cls_ids):
        key = img_name.lower()
        if key not in files_by_name or c not in id2idx: continue
        paths.append(files_by_name[key]); labels.append(id2idx[c])
    print(f'[test] matched {len(paths)}/{len(names)} files')
    return np.array(paths), np.array(labels, np.int32)

def _preflight_test(class_ids, id2idx):
    names, cls = _load_groundtruth()
    paths, labels = _find_test_files(names, cls, id2idx)
    if len(paths) == 0:
        print('\n[!] 0 test images found – aborting before training!\n'
              '    check:  (1) groundtruth.txt format  (2) test/ folder  (3) file names match\n')
        exit(1)
    return paths, labels

# evaluation
def _eval_top1_top5(model, ds, y_true):
    y_prob = model.predict(ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    top5hit = (np.argsort(-y_prob, axis=1)[:, :5] == y_true.reshape(-1, 1)).any(axis=1)
    top1 = float((y_true == y_pred).mean())
    top5 = float(top5hit.mean())
    return top1, top5, y_prob

def print_results(model, ds_val, y_val, ds_test, y_test, unpaired_mask):
    v1, v5, _ = _eval_top1_top5(model, ds_val, y_val)
    print(f'Validation Top-1: {v1*100:.2f} %')
    print(f'Validation Top-5: {v5*100:.2f} %')

    t1, t5, y_prob = _eval_top1_top5(model, ds_test, y_test)
    print(f'Test Overall Top-1: {t1*100:.2f} %')
    print(f'Test Overall Top-5: {t5*100:.2f} %')

    # unpaired = missing either domain in training
    unpaired_flag = unpaired_mask[y_test]
    for name, mask in (('Paired', ~unpaired_flag), ('Unpaired', unpaired_flag)):
        if mask.sum() == 0:
            print(f'Test {name} Top-1: N/A | Top-5: N/A'); continue
        p1 = float((y_test[mask] == np.argmax(y_prob, axis=1)[mask]).mean())
        p5 = float((np.argsort(-y_prob, axis=1)[mask][:, :5] == y_test[mask].reshape(-1, 1)).any(axis=1).mean())
        print(f'Test {name} Top-1: {p1*100:.2f} %')
        print(f'Test {name} Top-5: {p5*100:.2f} %')

# main
def main():
    global class_ids, id2idx, unpaired_mask
    (class_files, class_labels, domain_flag,
     class_ids, n_classes, id2idx, paired_mask, unpaired_mask) = _gather_train()
    test_paths, test_labels = _preflight_test(class_ids, id2idx)
    tr_f, tr_l, tr_d, val_f, val_l, val_d = _stratified_split(class_files, class_labels, domain_flag)
    print(f'\n[data] train={len(tr_f)}  val={len(val_f)}  classes={n_classes}\n')
    ds_tr  = make_ds(tr_f, tr_l, tr_d, training=True)
    ds_val = make_ds(val_f, val_l, val_d, training=False)
    model, _ = train(build_model(n_classes), ds_tr, ds_val, n_classes)
    ds_test = make_ds(test_paths, test_labels, np.zeros(len(test_paths), np.int32), training=False)
    print_results(model, ds_val, val_l, ds_test, test_labels, unpaired_mask)
    model.save(CKPT_DIR / 'final_model.keras')
    print('\n[done] model saved to', CKPT_DIR / 'final_model.keras')

if __name__ == '__main__':
    main()