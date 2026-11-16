"""Builds the GPU training notebooks.

One template, three notebooks:
  02_paper_replication : split the data the way the paper did (random 80/20, duplicates allowed)
  03_leak_free         : remove duplicates first, keep near-copies on one side of the split,
                         plus an extra test: train on farm (eye) labels, test on lab (PCR) labels
  smoke_test           : tiny run (few images, 1 epoch) to check the code before the long runs

Usage:  python build_02_train.py
"""
import os
from nb_tools import make_notebook, write_metadata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "kernels")
DATASET = "allandclive/chicken-disease-1"

# Near-duplicate limit (Hamming distance on the 64-bit dHash), chosen from notebook 01.
NEAR_DUP_MAX_HAMMING = 4


def cells_for(protocol, smoke):
    intro = {
        "paper": """
# 02 · Replicating the paper: poultry disease from droppings images (GPU)

**Paper:** Machuve et al. (2022), *Poultry diseases diagnostics models using deep learning*,
Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

**Goal of this notebook.** Rebuild the paper's experiment as closely as the text allows, and check
whether we get the same accuracies (their Table 2).

**What we copy from the paper (Section 3):**
* 8,067 images, 4 classes: Coccidiosis, Healthy, Newcastle disease (NCD), Salmonella.
* Training data = farm-labelled + lab (PCR)-labelled images mixed. **Test data = farm-labelled images only.**
* Split: 80% train / 20% test; then 15% of the training part is used for validation.
* Five models: a small CNN built from scratch (baseline), VGG16, InceptionV3, MobileNetV2, Xception.
* Three training stages for the four borrowed (pre-trained) models:
  1. **Pre-training** – the borrowed network is frozen; only the new top layers learn.
  2. **Fine-tuning** – part or all of the borrowed network is unfrozen and learns slowly.
  3. **Fine-tuning with frozen BatchNorm** – same as 2, but the BatchNorm layers keep their ImageNet statistics.
* Image size, pixel scaling, augmentation and top layers for each model, exactly as written in Sections 3.4–3.5.

**What the paper does not tell us (in the main text), and what we chose.** The optimizer, learning rates and
epochs are in their Supplementary Table 4, which we could not read. We use common Keras defaults:
Adam optimizer, learning rate 0.001 for stage 1 and 0.00001 for stages 2–3, up to 30 epochs per stage,
stop early if the validation loss does not improve for 8 epochs, keep the best epoch (judged on validation data only).
The baseline CNN is trained for 100 epochs as the paper says.
""",
        "leakfree": f"""
# 03 · Leak-free re-test of the paper (GPU)

**Paper:** Machuve et al. (2022), Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

Notebook 01 found **duplicate images** in the dataset. In notebook 02 we split the data at random, like the paper.
So some test images also sit in the training set (as a copy). The model may simply remember them.

**This notebook asks: what accuracy do the same models reach when no copy can cross the split?**

What changes compared with notebook 02 (everything else is identical):
1. Identical files (same MD5 fingerprint) are reduced to **one copy**.
2. Images that are almost the same (dHash Hamming distance ≤ {NEAR_DUP_MAX_HAMMING} of 64 bits) are put in one **group**.
   A whole group goes either to training or to testing, never to both.
3. Groups where copies carry **different disease labels** are removed (we cannot know which label is right).

**Extra experiment (Part C): do farm labels agree with the laboratory?**
Farm labels were given by eye (colour and shape of the dropping). Lab labels came from PCR, the real test.
We train MobileNetV2 (the paper's recommended model) on **farm images only** and test it on the **PCR-labelled images**.
If farm labels were as good as PCR, accuracy should stay high.
""",
    }[protocol]
    if smoke:
        intro = "# SMOKE TEST (few images, 1 epoch) – only checks that the code runs\n" + intro

    cells = [("md", intro)]

    cells.append(("code", f'''
# ============================== SETTINGS ==============================
PROTOCOL = "{protocol}"     # "paper" = random split like the paper, "leakfree" = duplicates handled
SMOKE = {smoke}              # True = tiny quick run to test the code
SEED = 42                     # fixed random seed so the split and the training can be repeated
NEAR_DUP_MAX_HAMMING = {NEAR_DUP_MAX_HAMMING}      # used only by the leak-free protocol

# Training choices that the paper's main text does not give (see the introduction).
LR_HEAD = 1e-3                # learning rate, stage 1 (only the new top layers learn)
LR_FINE = 1e-5                # learning rate, stages 2 and 3 (the borrowed network also learns)
EPOCHS_STAGE = 1 if SMOKE else 30
PATIENCE = 8                  # early stopping: wait this many epochs without improvement
EPOCHS_BASELINE = 1 if SMOKE else 100
BATCH = 64                    # batch size given in the paper
BATCH_XCEPTION_FINE = 32      # Xception fine-tuning at 299 px with batch 64 does not fit in the T4's 16 GB memory
'''))

    cells.append(("md", """
## 1. Imports and GPU check

* **TensorFlow / Keras**: the deep learning library the paper used.
* **mixed precision**: the GPU does most maths in 16-bit numbers. This is about 2× faster on a T4 GPU
  and does not change the accuracy in a meaningful way.
"""))
    cells.append(("code", '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hide TensorFlow's C++ timing messages (Python errors still show)
import gc, glob, json, time, hashlib, platform, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

import tensorflow as tf
import keras
from keras import layers

keras.utils.set_random_seed(SEED)                          # same seed for python, numpy and tensorflow
keras.mixed_precision.set_global_policy("mixed_float16")   # faster training on the GPU

OUT = f"/kaggle/working/results_{PROTOCOL}"
os.makedirs(OUT, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)    # take GPU memory only when needed
versions = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "gpu": str(gpus)}
print(versions)
json.dump(versions, open(f"{OUT}/versions.json", "w"), indent=1)
assert gpus or SMOKE, "No GPU found: switch the notebook accelerator to GPU."
'''))

    cells.append(("md", """
## 2. Load all images into memory

There are 8,067 small images (224 × 224). Together they need about 1.2 GB of memory, so we load them once.
The class list is fixed in alphabetical order; the model's output number 0, 1, 2, 3 means the class at that position.
"""))
    cells.append(("code", '''
csv_path = glob.glob("/kaggle/input/**/train_data.csv", recursive=True)[0]
IMG_DIR = os.path.join(os.path.dirname(csv_path), "Train")

df = pd.read_csv(csv_path)
df.columns = ["file", "label"]
df["source"] = np.where(df["file"].str.startswith("pcr"), "lab", "farm")   # 'pcr' prefix = lab (PCR) label
CLASSES = sorted(df["label"].unique())        # ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
SHORT = ["cocci", "healthy", "ncd", "salmo"]  # short names used in the paper's Table 2
df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})

if SMOKE:   # keep 100 images per class/source mix for a quick test
    df = df.groupby(["label", "source"], group_keys=False).apply(lambda g: g.sample(min(len(g), 25), random_state=SEED))
    df = df.reset_index(drop=True)

def load(f):
    img = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224))
    return np.asarray(img, dtype=np.uint8)

t0 = time.time()
X = np.stack([load(f) for f in df["file"]])     # shape (N, 224, 224, 3), whole numbers 0-255
Y = df["y"].values.astype(np.int32)
print(X.shape, f"loaded in {time.time()-t0:.0f} s")
print(pd.crosstab(df["label"], df["source"], margins=True))
'''))

    cells.append(("md", """
## 3. Fingerprints for duplicate checks

We need these in both notebooks: in the leak-free one to build the groups, and in the paper one to **count how many
test images have a copy in the training set**.
* **MD5**: identical files.
* **dHash**: a 64-bit code of the picture's light/dark pattern. Similar pictures have similar codes.
  The **Hamming distance** = how many of the 64 bits differ.
"""))
    cells.append(("code", '''
df["md5"] = [hashlib.md5(open(os.path.join(IMG_DIR, f), "rb").read()).hexdigest() for f in df["file"]]

def dhash_bits(arr):
    g = Image.fromarray(arr).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

Bf = np.stack([dhash_bits(x) for x in X]).astype(np.float32)
HAM = np.rint(Bf @ (1 - Bf).T + (1 - Bf) @ Bf.T).astype(np.int16)    # Hamming distance for every pair
np.fill_diagonal(HAM, 99)

# Images linked by "same file" OR "Hamming distance <= limit" form one group (connected components of a graph).
same_md5 = (df["md5"].values[:, None] == df["md5"].values[None, :])
np.fill_diagonal(same_md5, False)
LINK = same_md5 | (HAM <= NEAR_DUP_MAX_HAMMING)
n_groups, GROUP = connected_components(csr_matrix(LINK), directed=False)
df["group"] = GROUP
print("images:", len(df), "| groups:", n_groups, "| images that share a group:", int((df.groupby("group")["file"].transform("size") > 1).sum()))
'''))

    if protocol == "paper":
        split_md = """
## 4. Split the data the way the paper did

1. Take the **farm-labelled** images and split them 80% / 20% at random, keeping class proportions (stratified).
   The 20% part is the **test set** (the paper tested on farm images only).
2. Add **all lab-labelled** images to the 80% part. This is the training pool.
3. Take 15% of the training pool as the **validation set**. It is used only to stop training and pick the best epoch.

At the end we count how many test images have a copy (same group) in training. This is the leakage.
"""
        split_code = '''
farm = df.index[df["source"] == "farm"].values
lab = df.index[df["source"] == "lab"].values
farm_train, test_idx = train_test_split(farm, test_size=0.20, stratify=Y[farm], random_state=SEED)
pool = np.concatenate([farm_train, lab])
train_idx, val_idx = train_test_split(pool, test_size=0.15, stratify=Y[pool], random_state=SEED)
'''
    else:
        split_md = f"""
## 4. Leak-free split

1. **Remove conflicting groups**: groups whose copies carry more than one label.
2. **Keep one copy of identical files** (same MD5). Near-copies stay, but inside one group.
3. **Group-aware split** with `StratifiedGroupKFold`: it cuts the data in 5 parts so that each group sits in exactly
   one part, and each part has about the same class mix. Part 1 = test candidates (~20%).
4. Test set = the **farm** images in that part (same rule as the paper). Lab images of that part are left out,
   so that no copy of them can leak.
5. The other 4 parts = training pool. Validation (~15%) is cut from it the same group-aware way.
"""
        split_code = '''
n_lab_per_group = df.groupby("group")["label"].transform("nunique")
conflict = df.index[n_lab_per_group > 1].values
print("images in groups with conflicting labels (removed):", len(conflict))
print(df.loc[conflict, ["file", "label", "group"]].sort_values("group").to_string())

keep = df.drop(index=conflict).drop_duplicates("md5").index.values       # one copy per identical file
print("images kept:", len(keep), "of", len(df))

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
pool_pos, test_pos = next(sgkf.split(keep, Y[keep], groups=GROUP[keep]))
test_part, pool = keep[test_pos], keep[pool_pos]
test_idx = test_part[df.loc[test_part, "source"].values == "farm"]        # test = farm images only

sgkf_val = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)   # 1/7 = ~14% validation
tr_pos, va_pos = next(sgkf_val.split(pool, Y[pool], groups=GROUP[pool]))
train_idx, val_idx = pool[tr_pos], pool[va_pos]
'''
    cells.append(("md", split_md))
    cells.append(("code", split_code + '''
def leak_count(a, b):
    """How many images in set a have a copy (same group) in set b."""
    return int(np.isin(GROUP[a], GROUP[b]).sum())

split_info = pd.DataFrame({
    "set": ["train", "validation", "test"],
    "images": [len(train_idx), len(val_idx), len(test_idx)],
    "farm": [int((df.loc[i, "source"] == "farm").sum()) for i in (train_idx, val_idx, test_idx)],
    "lab": [int((df.loc[i, "source"] == "lab").sum()) for i in (train_idx, val_idx, test_idx)],
})
print(split_info.to_string(index=False))
print(pd.crosstab(df.loc[test_idx, "label"], "test images"))
leak = {
    "test_images": len(test_idx),
    "test_with_identical_copy_in_train": int(np.isin(df.loc[test_idx, "md5"], df.loc[train_idx, "md5"]).sum()),
    "test_with_near_copy_in_train_or_val": leak_count(test_idx, np.concatenate([train_idx, val_idx])),
}
print(leak)
split_info.to_csv(f"{OUT}/split_sizes.csv", index=False)
json.dump(leak, open(f"{OUT}/leakage.json", "w"), indent=1)
'''))

    cells.append(("md", """
## 5. Model settings copied from the paper

Each model got its own image size, pixel scaling and augmentation in the paper (Sections 3.4–3.5).
**Augmentation** = random changes to training images (flip, rotate, shift, darken) so the model sees more variety.
Test images are never augmented.

Notes on details that matter:
* `brightness (0.1, 0.7)`: in Keras' old `ImageDataGenerator`, this **multiplies** every pixel by a random number
  between 0.1 and 0.7. So every training image becomes darker (10–70% of its brightness), while test images are not.
  We copy this as written.
* Pixel scaling: the paper used 0–1 for InceptionV3, MobileNetV2 and Xception, and −1 to 1 for VGG16.
  (The ImageNet weights of these networks were originally trained with other scalings; we copy the paper anyway.)
* Which layers are unfrozen in stage 2–3:
  * VGG16: "the last two pre-trained layers" → `block5_conv3` and `block5_pool`.
  * InceptionV3: everything after the layer `mixed8` (the top two Inception blocks; this matches the Keras guide).
  * MobileNetV2: the whole network.
  * Xception: "132 layers" – Keras' Xception has exactly 132 layers, so this is the whole network.
* **Batch size**: 64 as in the paper, except Xception in stages 2–3, which uses 32. At 299 px with the whole network
  unfrozen, batch 64 ran out of the T4 GPU's memory in our test run.
* **VGG16 has no BatchNorm layers.** So for VGG16, stage 3 is the same as stage 2; any difference is random.
"""))
    cells.append(("code", '''
MODELS = {
    #             input size, pixel scaling, augmentation settings
    "BaselineCNN": dict(size=128, scale="0_1",  aug=dict(flip="horizontal_and_vertical", rot=90, shift_w=0.5, bright=(0.1, 0.7))),
    "VGG16":       dict(size=224, scale="-1_1", aug=dict(flip="horizontal_and_vertical", rot=90, shift_w=0.5, shift_h=0.5, bright=(0.1, 0.7))),
    "InceptionV3": dict(size=299, scale="0_1",  aug=dict(rot=40, shift_w=0.2, shift_h=0.2, shear=0.2, zoom=0.2)),
    "MobileNetV2": dict(size=224, scale="0_1",  aug=dict(flip="vertical", rot=45, bright=(0.2, 1.0))),
    "Xception":    dict(size=299, scale="0_1",  aug=dict(flip="horizontal_and_vertical", bright=(0.1, 0.7))),
}

# Paper results (Table 2) to compare with ours: accuracy % for stages 1, 2, 3 and F1 {cocci, healthy, ncd, salmo}.
PAPER = {
    "BaselineCNN": dict(acc=[83.06, None, None], f1=[0.92, 0.86, 0.14, 0.81]),
    "VGG16":       dict(acc=[85.85, 94.65, 95.01], f1=[0.98, 0.95, 0.77, 0.95]),
    "InceptionV3": dict(acc=[94.79, 94.94, 95.45], f1=[0.98, 0.94, 0.77, 0.95]),
    "MobileNetV2": dict(acc=[87.46, 97.14, 98.02], f1=[0.99, 0.97, 0.94, 0.99]),
    "Xception":    dict(acc=[88.27, 97.36, 98.24], f1=[0.97, 0.98, 0.98, 0.99]),
}
'''))

    cells.append(("md", """
## 6. Data pipeline (tf.data)

For each batch of images: resize to the model's size → random augmentation (training only) → scale the pixels.
This runs on the GPU side in parallel with training, so the GPU does not wait.
"""))
    cells.append(("code", '''
def make_augmenter(a):
    """Build a chain of Keras random layers from the paper's settings.
    dtype="float32": with mixed precision switched on, Keras layers would otherwise output 16-bit numbers."""
    steps = []
    if a.get("flip"):
        steps.append(layers.RandomFlip(a["flip"], dtype="float32"))
    if a.get("rot"):     # Keras uses a fraction of a full turn: 90 degrees = 0.25
        steps.append(layers.RandomRotation(a["rot"] / 360, fill_mode="nearest", dtype="float32"))
    if a.get("shift_w") or a.get("shift_h"):
        steps.append(layers.RandomTranslation(a.get("shift_h", 0.0), a.get("shift_w", 0.0), fill_mode="nearest", dtype="float32"))
    if a.get("shear"):
        if hasattr(layers, "RandomShear"):
            steps.append(layers.RandomShear(x_factor=a["shear"], y_factor=a["shear"], fill_mode="nearest", dtype="float32"))
        else:
            print("RandomShear not in this Keras version: shear skipped")
    if a.get("zoom"):
        steps.append(layers.RandomZoom(a["zoom"], fill_mode="nearest", dtype="float32"))
    return steps

def make_ds(idx, cfg, train, batch=None):
    batch = batch or BATCH
    size, aug = cfg["size"], make_augmenter(cfg["aug"]) if train else []
    bright = cfg["aug"].get("bright") if train else None

    def prep(x, y):
        x = tf.cast(x, tf.float32)                       # 0-255
        if size != 224:
            x = tf.image.resize(x, (size, size))
        for layer in aug:
            x = tf.cast(layer(x, training=True), tf.float32)   # keep 32-bit numbers in the pipeline
        if bright:                                       # ImageDataGenerator-style brightness: multiply
            f = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], bright[0], bright[1])
            x = tf.clip_by_value(x * f, 0.0, 255.0)
        x = x / 255.0 if cfg["scale"] == "0_1" else x / 127.5 - 1.0
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((X[idx], Y[idx]))
    if train:
        ds = ds.shuffle(len(idx), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch).map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
'''))

    cells.append(("md", """
## 7. Building the models

* **Baseline CNN** (Section 3.4): one convolution layer with 32 filters, then four blocks of
  convolution → max pooling → dropout 0.2, then dense layers with softmax. The paper does not give the filter numbers
  of the four blocks or the dense layer size; we use 32-64-128-128 filters and a 256-unit dense layer.
* **Borrowed (pre-trained) networks**: the ImageNet version without its top, plus the new top layers from the paper.
* The last layer always has 4 outputs (one per class) and softmax, which turns them into probabilities.
"""))
    cells.append(("code", '''
def build_baseline():
    inp = keras.Input((128, 128, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(4, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out), None

def build_transfer(name):
    size = MODELS[name]["size"]
    app = {"VGG16": keras.applications.VGG16, "InceptionV3": keras.applications.InceptionV3,
           "MobileNetV2": keras.applications.MobileNetV2, "Xception": keras.applications.Xception}[name]
    base = app(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    inp = keras.Input((size, size, 3))
    x = base(inp)                      # no 'training' flag: BatchNorm follows each layer's trainable setting
    if name == "VGG16":
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dense(1072, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    elif name == "InceptionV3":
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
    else:                              # MobileNetV2 and Xception
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
    out = layers.Dense(4, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out), base

def set_stage(name, base, stage):
    """Stage 1: base frozen. Stage 2: unfreeze the paper's layers. Stage 3: as 2 but BatchNorm stays frozen."""
    if stage == 1:
        base.trainable = False
        return
    base.trainable = True
    names = [l.name for l in base.layers]
    if name == "VGG16":
        first = len(base.layers) - 2                     # last two layers
    elif name == "InceptionV3":
        first = names.index("mixed8") + 1                 # everything after mixed8
    elif name == "Xception":
        first = len(base.layers) - 132                    # "132 layers" = all of Keras' Xception
    else:
        first = 0                                         # MobileNetV2: the whole network
    for i, l in enumerate(base.layers):
        l.trainable = i >= max(first, 0)
        if stage == 3 and isinstance(l, layers.BatchNormalization):
            l.trainable = False                           # Keras then also runs it in inference mode
'''))

    cells.append(("md", """
## 8. Training and testing functions

* **Loss**: sparse categorical cross-entropy (the standard loss for one-label, many-class problems).
* **Early stopping**: watches the validation loss; keeps the weights of the best epoch.
* **Testing**: we report accuracy with a 95% confidence interval (Wilson method), macro F1
  (the plain average of the four class F1 scores, so the small NCD class counts as much as the others),
  and precision, recall and F1 for each class, like the paper.
* Every result is written to `results.csv` as soon as it is ready, so nothing is lost if the run stops later.
"""))
    cells.append(("code", '''
def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return 100*(centre-half), 100*(centre+half)

RESULTS_CSV = f"{OUT}/results.csv"
rows = pd.read_csv(RESULTS_CSV).to_dict("records") if os.path.exists(RESULTS_CSV) else []

def evaluate(model, name, stage, cfg, idx, tag="test"):
    prob = model.predict(make_ds(idx, cfg, train=False), verbose=0)
    pred, true = prob.argmax(1), Y[idx]
    acc = accuracy_score(true, pred)
    lo, hi = wilson(int((pred == true).sum()), len(idx))
    p, r, f, _ = precision_recall_fscore_support(true, pred, labels=range(4), zero_division=0)
    row = {"protocol": PROTOCOL, "eval_set": tag, "model": name, "stage": stage, "n_test": len(idx),
           "accuracy": round(100*acc, 2), "acc_ci_low": round(lo, 2), "acc_ci_high": round(hi, 2),
           "macro_f1": round(f1_score(true, pred, average="macro"), 4)}
    for k, s in enumerate(SHORT):
        row[f"precision_{s}"], row[f"recall_{s}"], row[f"f1_{s}"] = round(p[k], 4), round(r[k], 4), round(f[k], 4)
    rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    cm = confusion_matrix(true, pred, labels=range(4))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(f"{OUT}/cm_{tag}_{name}_stage{stage}.csv")
    print(f"  {tag}: {name} stage {stage}: accuracy {100*acc:.2f}% (95% CI {lo:.1f}-{hi:.1f}), "
          f"macro F1 {row['macro_f1']:.3f}, F1 per class {[round(x, 2) for x in f]}")
    return row

def fit(model, name, stage, cfg, lr, epochs, tr, va, early_stop=True):
    batch = BATCH_XCEPTION_FINE if (name.endswith("Xception") and stage > 1) else BATCH
    # jit_compile=False: no XLA compiler. XLA asked for one 7 GB block for Xception and crashed in our test run.
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                  jit_compile=False)
    n_train = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)] if early_stop else []
    t0 = time.time()
    h = model.fit(make_ds(tr, cfg, True, batch), validation_data=make_ds(va, cfg, False, batch), epochs=epochs, callbacks=cb, verbose=2)
    hist = pd.DataFrame(h.history); hist["epoch"] = np.arange(1, len(hist) + 1)
    hist.to_csv(f"{OUT}/history_{name}_stage{stage}.csv", index=False)
    print(f"  {name} stage {stage}: {len(hist)} epochs, {n_train:,} trainable weights, {(time.time()-t0)/60:.1f} min")
'''))

    cells.append(("md", """
## 9. Run all five models

For each borrowed network: stage 1 (top only) → save weights → stage 2 (fine-tune) → test.
Then go back to the stage-1 weights → stage 3 (fine-tune with frozen BatchNorm) → test.
Stages 2 and 3 therefore start from the same point and get the same number of epochs, so they can be compared fairly.
If one model crashes, the error is written down and the next model still runs.
Xception runs first because it needs the most GPU memory; the result tables are put back in the paper's order at the end.
"""))
    cells.append(("code", '''
# Run order: the biggest model (Xception at 299 px) first, while the GPU memory is still empty.
RUN_ORDER = ["Xception", "InceptionV3", "VGG16", "MobileNetV2", "BaselineCNN"]
done = {(r["model"], r["stage"]) for r in rows if r["eval_set"] == "test"}
for name in RUN_ORDER:
    cfg = MODELS[name]
    print(f"===== {name} =====")
    try:
        if name == "BaselineCNN":
            if (name, 1) not in done:
                model, _ = build_baseline()
                fit(model, name, 1, cfg, LR_HEAD, EPOCHS_BASELINE, train_idx, val_idx, early_stop=False)
                evaluate(model, name, 1, cfg, test_idx)
                model.save(f"{OUT}/model_{name}.keras")
            continue
        model, base = build_transfer(name)
        set_stage(name, base, 1)
        fit(model, name, 1, cfg, LR_HEAD, EPOCHS_STAGE, train_idx, val_idx)
        evaluate(model, name, 1, cfg, test_idx)
        stage1_weights = f"/tmp/{name}_stage1.weights.h5"
        model.save_weights(stage1_weights)
        for stage in (2, 3):
            model.load_weights(stage1_weights)
            set_stage(name, base, stage)
            fit(model, name, stage, cfg, LR_FINE, EPOCHS_STAGE, train_idx, val_idx)
            evaluate(model, name, stage, cfg, test_idx)
            if name == "MobileNetV2" and stage == 3:
                model.save(f"{OUT}/model_{name}_stage3.keras")    # the paper's recommended model
    except Exception:
        traceback.print_exc()
        open(f"{OUT}/errors.txt", "a").write(f"{name}\\n{traceback.format_exc()}\\n")
    finally:
        keras.backend.clear_session(); gc.collect()
'''))

    if protocol == "leakfree":
        cells.append(("md", """
## 10. Part C – Train on farm (eye) labels, test on lab (PCR) labels

* Training pool: the farm images of section 4's training pool (so the farm test set stays unseen).
* Test set: PCR-labelled images whose group has **no** member in the training pool (no copies can leak).
* Model: MobileNetV2 with the stage 1 → stage 3 recipe (the paper's recommended setup).

We also check the reverse direction on the same model: its accuracy on the leak-free farm test set from section 4.
If the accuracy on PCR images is much lower than on farm images, the model has learned the **farm labelling habit**
(colour and shape rules) more than the disease itself.
"""))
        cells.append(("code", '''
farm_keep = pool[df.loc[pool, "source"].values == "farm"]   # farm images of the training pool only
lab_all = df.index[(df["source"] == "lab")].values
lab_all = lab_all[~np.isin(lab_all, conflict)]
lab_test = lab_all[~np.isin(GROUP[lab_all], GROUP[farm_keep])]         # lab images with no copy among farm images
print("PCR test images:", len(lab_test), "| dropped because a farm copy exists:", len(lab_all) - len(lab_test))
print(pd.crosstab(df.loc[lab_test, "label"], "PCR test images"))

sgkf_c = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
tr_pos, va_pos = next(sgkf_c.split(farm_keep, Y[farm_keep], groups=GROUP[farm_keep]))
c_train, c_val = farm_keep[tr_pos], farm_keep[va_pos]

name, cfg = "MobileNetV2", MODELS["MobileNetV2"]
try:
    model, base = build_transfer(name)
    set_stage(name, base, 1)
    fit(model, "C_" + name, 1, cfg, LR_HEAD, EPOCHS_STAGE, c_train, c_val)
    set_stage(name, base, 3)
    fit(model, "C_" + name, 3, cfg, LR_FINE, EPOCHS_STAGE, c_train, c_val)
    evaluate(model, "C_farm_only_" + name, 3, cfg, lab_test, tag="pcr_test")
    farm_test_unseen = test_idx[~np.isin(GROUP[test_idx], GROUP[c_train])]
    evaluate(model, "C_farm_only_" + name, 3, cfg, farm_test_unseen, tag="farm_test")
except Exception:
    traceback.print_exc()
finally:
    keras.backend.clear_session(); gc.collect()
'''))

    cells.append(("md", """
## Final · Our results next to the paper's Table 2
"""))
    cells.append(("code", '''
res = pd.read_csv(RESULTS_CSV)
t = res[res["eval_set"] == "test"].pivot_table(index="model", columns="stage", values="accuracy")
t = t.reindex([m for m in MODELS if m in t.index])          # paper's order
t.columns = [f"ours_stage{c}" for c in t.columns]
for s in (1, 2, 3):
    t[f"paper_stage{s}"] = [PAPER.get(m, {}).get("acc", [None]*3)[s-1] for m in t.index]
t = t[sorted(t.columns, key=lambda c: (c[-1], c))]
print(t.round(2).to_string())
t.to_csv(f"{OUT}/compare_with_paper.csv")
print()
print(res[["eval_set", "model", "stage", "n_test", "accuracy", "acc_ci_low", "acc_ci_high", "macro_f1",
           "f1_cocci", "f1_healthy", "f1_ncd", "f1_salmo"]].to_string(index=False))
'''))
    cells.append(("code", '''
# Learning curves for the two models the paper discusses (MobileNetV2 and Xception, stage 3).
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, name in enumerate(["MobileNetV2", "Xception"]):
    p = f"{OUT}/history_{name}_stage3.csv"
    if os.path.exists(p):
        h = pd.read_csv(p)
        ax[k].plot(h["epoch"], h["loss"], label="training loss")
        ax[k].plot(h["epoch"], h["val_loss"], label="validation loss")
        ax[k].set_title(f"{name}, stage 3 ({PROTOCOL})"); ax[k].set_xlabel("epoch"); ax[k].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/curves_stage3.png", dpi=130); plt.show()
'''))
    return cells


for protocol, smoke, folder, slug, nbname in [
    ("paper", False, "02_paper_replication", "poultry-feces-02-paper-replication", "02_paper_replication.ipynb"),
    ("leakfree", False, "03_leak_free", "poultry-feces-03-leak-free", "03_leak_free.ipynb"),
    ("leakfree", True, "smoke_test", "poultry-feces-smoke-test", "smoke_test.ipynb"),
]:
    out = os.path.join(ROOT, folder)
    os.makedirs(out, exist_ok=True)
    make_notebook(cells_for(protocol, smoke), os.path.join(out, nbname))
    write_metadata(out, slug, slug, nbname, gpu=True, datasets=[DATASET])
    print("built", folder)
