"""Writes the GPU training notebooks from one template.

    02_paper_replication   the paper's split: random 80/20, duplicates left in
    03_leak_free           copies kept on one side of the split, plus Part C (train on field labels, test on PCR)
    smoke_test             a few images and 1 epoch, to catch bugs before the long runs

Run:  python build_02_train.py
"""
import os
from nb_tools import make_notebook, write_metadata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "kernels")
DATASET = "allandclive/chicken-disease-1"

NEAR_DUP_MAX_HAMMING = 4   # dHash limit, picked by eye from the pairs in notebook 01


def cells_for(protocol, smoke):
    intro = {
        "paper": """
# Poultry Feces 02: Paper Replication

**Paper:** Machuve et al. (2022), *Poultry diseases diagnostics models using deep learning*,
Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

Here I rebuild the paper's experiment as closely as its text allows, and check whether the accuracies in its
Table 2 come out again.

**Taken from the paper (Section 3)**
* 8,067 images, 4 classes: Coccidiosis, Healthy, Newcastle disease (NCD), Salmonella.
* Training data: field-labelled and PCR-labelled images mixed. Test data: field-labelled images only.
* Split: 80% train, 20% test, then 15% of the training part for validation.
* Five models: a small CNN trained from scratch (the baseline), VGG16, InceptionV3, MobileNetV2 and Xception.
* Three stages for the four pre-trained networks:
  1. *Pre-training*: the network is frozen and only the new top layers learn.
  2. *Fine-tuning*: part or all of the network is unfrozen and learns slowly.
  3. *Fine-tuning with frozen BatchNorm*: as 2, but the BatchNorm layers keep their ImageNet statistics.
* Image size, pixel scaling, augmentation and top layers per model, as written in Sections 3.4 and 3.5.

**Not in the main text, so chosen by me.** The optimizer, learning rates and epochs are in the paper's
Supplementary Table 4, which I could not get. I used common Keras choices: Adam, learning rate 0.001 for stage 1
and 0.00001 for stages 2 and 3, at most 30 epochs per stage, early stopping after 8 epochs without a better
validation loss, keeping the best epoch. The baseline CNN gets 100 epochs, as in the paper.
""",
        "leakfree": f"""
# Poultry Feces 03: Leak-Free Re-Test

**Paper:** Machuve et al. (2022), Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

Notebook 01 found duplicate images. Notebook 02 split the data at random, like the paper, so some test images
also sit in the training set as a copy, and the model may just remember them.

**The question here: what do the same models score when no copy can cross the split?**

Changes from notebook 02 (everything else is the same):
1. Identical files (same MD5) are cut down to one copy.
2. Images that are almost the same (dHash distance of {NEAR_DUP_MAX_HAMMING} bits or less out of 64) form a group.
   A whole group goes to training or to testing, never to both.
3. Groups whose copies carry different labels are dropped, because there is no way to tell which label is right.

**Part C: do field labels agree with the laboratory?**
The field labels come from a vet's eye (the paper) or from where the photo was taken (the Zenodo record).
The PCR labels come from a laboratory test for the pathogen. I train MobileNetV2, the model the paper recommends,
on field-labelled images only and then score it against the PCR labels. If the two kinds of label agreed,
the accuracy would stay high.
""",
    }[protocol]
    if smoke:
        intro = "# SMOKE TEST (few images, 1 epoch): only checks that the code runs\n" + intro

    cells = [("md", intro)]

    cells.append(("code", f'''
# Settings
PROTOCOL = "{protocol}"     # "paper" = random split like the paper, "leakfree" = copies kept apart
SMOKE = {smoke}
SEED = 42
NEAR_DUP_MAX_HAMMING = {NEAR_DUP_MAX_HAMMING}      # only used by the leak-free protocol

# not given in the paper's main text, see the introduction
LR_HEAD = 1e-3                # stage 1, only the new top learns
LR_FINE = 1e-5                # stages 2 and 3
EPOCHS_STAGE = 1 if SMOKE else 30
PATIENCE = 8
EPOCHS_BASELINE = 1 if SMOKE else 100
BATCH = 64                    # from the paper
BATCH_XCEPTION_FINE = 32      # Xception at 299 px, fully unfrozen, does not fit in 16 GB at batch 64
'''))

    cells.append(("md", """
## 1. Imports and GPU check

TensorFlow/Keras, as in the paper. Mixed precision lets the T4 do most of the maths in 16-bit numbers, which is
roughly twice as fast and makes no real difference to accuracy.
"""))
    cells.append(("code", '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # quieter TensorFlow logs (Python errors still show)
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

keras.utils.set_random_seed(SEED)                          # python, numpy and tensorflow in one call
keras.mixed_precision.set_global_policy("mixed_float16")

OUT = f"/kaggle/working/results_{PROTOCOL}"
os.makedirs(OUT, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
versions = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "gpu": str(gpus)}
print(versions)
json.dump(versions, open(f"{OUT}/versions.json", "w"), indent=1)
assert gpus or SMOKE, "No GPU found: switch the notebook accelerator to GPU."
'''))

    cells.append(("md", """
## 2. Load the images

8,067 images of 224 x 224 pixels take about 1.2 GB, so they all go into memory once. The classes are sorted
alphabetically, and the model's outputs 0 to 3 follow that order.
"""))
    cells.append(("code", '''
csv_path = glob.glob("/kaggle/input/**/train_data.csv", recursive=True)[0]
IMG_DIR = os.path.join(os.path.dirname(csv_path), "Train")

df = pd.read_csv(csv_path)
df.columns = ["file", "label"]
df["source"] = np.where(df["file"].str.startswith("pcr"), "lab", "farm")   # pcr prefix = PCR label
CLASSES = sorted(df["label"].unique())        # Coccidiosis, Healthy, New Castle Disease, Salmonella
SHORT = ["cocci", "healthy", "ncd", "salmo"]  # names used in the paper's Table 2
df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})

if SMOKE:   # 25 per class and source
    df = df.groupby(["label", "source"], group_keys=False).apply(lambda g: g.sample(min(len(g), 25), random_state=SEED))
    df = df.reset_index(drop=True)

def load(f):
    img = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224))
    return np.asarray(img, dtype=np.uint8)

t0 = time.time()
X = np.stack([load(f) for f in df["file"]])     # (N, 224, 224, 3), values 0-255
Y = df["y"].values.astype(np.int32)
print(X.shape, f"loaded in {time.time()-t0:.0f} s")
print(pd.crosstab(df["label"], df["source"], margins=True))
'''))

    cells.append(("md", """
## 3. Fingerprints

Both notebooks need these. The leak-free one builds its groups from them, and the paper one uses them to count how
many test images have a copy in training.
* **MD5** finds identical files.
* **dHash** is a 64-bit code of the light and dark pattern in a picture. The Hamming distance is the number of bits
  that differ between two codes; a small distance means the pictures look almost the same.
"""))
    cells.append(("code", '''
df["md5"] = [hashlib.md5(open(os.path.join(IMG_DIR, f), "rb").read()).hexdigest() for f in df["file"]]

def dhash_bits(arr):
    g = Image.fromarray(arr).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

Bf = np.stack([dhash_bits(x) for x in X]).astype(np.float32)
HAM = np.rint(Bf @ (1 - Bf).T + (1 - Bf) @ Bf.T).astype(np.int16)    # distance for every pair
np.fill_diagonal(HAM, 99)

# same file OR close dHash -> linked; linked images form a group (connected components)
same_md5 = (df["md5"].values[:, None] == df["md5"].values[None, :])
np.fill_diagonal(same_md5, False)
LINK = same_md5 | (HAM <= NEAR_DUP_MAX_HAMMING)
n_groups, GROUP = connected_components(csr_matrix(LINK), directed=False)
df["group"] = GROUP
print("images:", len(df), "| groups:", n_groups, "| images that share a group:", int((df.groupby("group")["file"].transform("size") > 1).sum()))
'''))

    if protocol == "paper":
        split_md = """
## 4. Split the way the paper did

1. Split the field-labelled images 80/20 at random, keeping the class mix (stratified). The 20% is the test set;
   the paper tested on field images only.
2. Add all PCR-labelled images to the 80%. That is the training pool.
3. Take 15% of the pool for validation. It only decides when to stop and which epoch to keep.

At the end I count how many test images have a copy (same group) in training. That number is the leakage.
"""
        split_code = '''
farm = df.index[df["source"] == "farm"].values
lab = df.index[df["source"] == "lab"].values
farm_train, test_idx = train_test_split(farm, test_size=0.20, stratify=Y[farm], random_state=SEED)
pool = np.concatenate([farm_train, lab])
train_idx, val_idx = train_test_split(pool, test_size=0.15, stratify=Y[pool], random_state=SEED)
'''
    else:
        split_md = """
## 4. Leak-free split

1. Drop groups whose copies carry more than one label.
2. Keep one copy of each identical file. Near-copies stay, inside their group.
3. `StratifiedGroupKFold` cuts the data into 5 parts so that each group sits in one part and each part has about
   the same class mix. The first part (about 20%) holds the test candidates.
4. The test set is the field images of that part, the same rule as the paper. Its PCR images are left out, so no
   copy of them can leak.
5. The other 4 parts are the training pool. Validation (about 14%) is cut from it in the same group-aware way.
"""
        split_code = '''
n_lab_per_group = df.groupby("group")["label"].transform("nunique")
conflict = df.index[n_lab_per_group > 1].values
print("images in groups with conflicting labels (removed):", len(conflict))
print(df.loc[conflict, ["file", "label", "group"]].sort_values("group").to_string())

keep = df.drop(index=conflict).drop_duplicates("md5").index.values
print("images kept:", len(keep), "of", len(df))

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
pool_pos, test_pos = next(sgkf.split(keep, Y[keep], groups=GROUP[keep]))
test_part, pool = keep[test_pos], keep[pool_pos]
test_idx = test_part[df.loc[test_part, "source"].values == "farm"]

sgkf_val = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)   # 1 of 7 parts, ~14%
tr_pos, va_pos = next(sgkf_val.split(pool, Y[pool], groups=GROUP[pool]))
train_idx, val_idx = pool[tr_pos], pool[va_pos]
'''
    cells.append(("md", split_md))
    cells.append(("code", split_code + '''
def leak_count(a, b):
    """How many images in a have a copy (same group) in b."""
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
## 5. Model settings from the paper

Each model had its own image size, pixel scaling and augmentation (Sections 3.4 and 3.5). Augmentation means random
changes to the training images (flips, turns, shifts, darkening) so the model sees more variety. Test images are
never changed.

Details that matter:
* `brightness (0.1, 0.7)`: in Keras' old `ImageDataGenerator` this multiplies every pixel by a random number between
  0.1 and 0.7. Every training image gets darker (10 to 70% of its brightness) while test images do not. Copied as written.
* Pixel scaling: 0 to 1 for InceptionV3, MobileNetV2 and Xception, -1 to 1 for VGG16, as in the paper, even though
  the ImageNet weights were trained with other scalings.
* Layers unfrozen in stages 2 and 3:
  * VGG16: "the last two pre-trained layers", so `block5_conv3` and `block5_pool`.
  * InceptionV3: everything after `mixed8`, the top two Inception blocks, which is also what the Keras guide does.
  * MobileNetV2: the whole network.
  * Xception: "132 layers". Keras' Xception has exactly 132, so the whole network.
* Batch size 64 as in the paper, except Xception in stages 2 and 3, which needs 32. At 299 px with everything
  unfrozen, 64 ran out of memory on the T4 in the smoke test.
* VGG16 has no BatchNorm layers, so for VGG16 stage 3 is the same as stage 2 and any difference is random.
"""))
    cells.append(("code", '''
MODELS = {
    #              input size, pixel scaling, augmentation
    "BaselineCNN": dict(size=128, scale="0_1",  aug=dict(flip="horizontal_and_vertical", rot=90, shift_w=0.5, bright=(0.1, 0.7))),
    "VGG16":       dict(size=224, scale="-1_1", aug=dict(flip="horizontal_and_vertical", rot=90, shift_w=0.5, shift_h=0.5, bright=(0.1, 0.7))),
    "InceptionV3": dict(size=299, scale="0_1",  aug=dict(rot=40, shift_w=0.2, shift_h=0.2, shear=0.2, zoom=0.2)),
    "MobileNetV2": dict(size=224, scale="0_1",  aug=dict(flip="vertical", rot=45, bright=(0.2, 1.0))),
    "Xception":    dict(size=299, scale="0_1",  aug=dict(flip="horizontal_and_vertical", bright=(0.1, 0.7))),
}

# Table 2 of the paper: accuracy % for stages 1-3, and F1 for cocci, healthy, ncd, salmo
PAPER = {
    "BaselineCNN": dict(acc=[83.06, None, None], f1=[0.92, 0.86, 0.14, 0.81]),
    "VGG16":       dict(acc=[85.85, 94.65, 95.01], f1=[0.98, 0.95, 0.77, 0.95]),
    "InceptionV3": dict(acc=[94.79, 94.94, 95.45], f1=[0.98, 0.94, 0.77, 0.95]),
    "MobileNetV2": dict(acc=[87.46, 97.14, 98.02], f1=[0.99, 0.97, 0.94, 0.99]),
    "Xception":    dict(acc=[88.27, 97.36, 98.24], f1=[0.97, 0.98, 0.98, 0.99]),
}
'''))

    cells.append(("md", """
## 6. Input pipeline

Each batch is resized to the model's size, augmented (training only) and scaled. tf.data runs this alongside
training so the GPU is not left waiting.
"""))
    cells.append(("code", '''
def make_augmenter(a):
    """Keras random layers for the paper's settings.
    dtype="float32" because under mixed precision these layers would otherwise hand back float16."""
    steps = []
    if a.get("flip"):
        steps.append(layers.RandomFlip(a["flip"], dtype="float32"))
    if a.get("rot"):     # Keras wants a fraction of a full turn: 90 degrees = 0.25
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
            x = tf.cast(layer(x, training=True), tf.float32)
        if bright:                                       # ImageDataGenerator brightness is a multiply
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
## 7. Models

* **Baseline CNN** (Section 3.4): one convolution with 32 filters, then four blocks of convolution, max pooling and
  dropout 0.2, then dense layers and softmax. The paper does not give the filters of the four blocks or the size of the
  dense layer, so I used 32, 64, 128, 128 filters and 256 units.
* **Pre-trained networks**: the ImageNet model without its top, plus the top layers the paper describes.
* The last layer always has 4 outputs with softmax, one probability per class.
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
    x = base(inp)                      # no training= flag, so BatchNorm follows each layer's trainable setting
    if name == "VGG16":
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dense(1072, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    elif name == "InceptionV3":
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
    else:                              # MobileNetV2, Xception
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
    out = layers.Dense(4, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out), base

def set_stage(name, base, stage):
    """1: base frozen. 2: unfreeze the layers the paper names. 3: like 2 with BatchNorm kept frozen."""
    if stage == 1:
        base.trainable = False
        return
    base.trainable = True
    names = [l.name for l in base.layers]
    if name == "VGG16":
        first = len(base.layers) - 2
    elif name == "InceptionV3":
        first = names.index("mixed8") + 1
    elif name == "Xception":
        first = len(base.layers) - 132                    # = 0, all of Keras' Xception
    else:
        first = 0                                         # MobileNetV2
    for i, l in enumerate(base.layers):
        l.trainable = i >= max(first, 0)
        if stage == 3 and isinstance(l, layers.BatchNormalization):
            l.trainable = False                           # in Keras 3 this also puts it in inference mode
'''))

    cells.append(("md", """
## 8. Training and scoring

* Loss: sparse categorical cross-entropy, the usual one for a single label out of several classes.
* Early stopping watches the validation loss and restores the best epoch.
* Scores: accuracy with a Wilson 95% interval, macro F1 (the plain mean of the four class F1 scores, so the small NCD
  class counts as much as the others), and precision, recall and F1 per class, as in the paper.
* Each result goes into `results.csv` as soon as it exists, so a crash later on loses nothing.
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
    # no XLA: it asked for one 7 GB block for Xception and crashed the smoke test
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
## 9. Run the five models

For each pre-trained network: stage 1, save the weights, stage 2, test. Then reload the stage 1 weights, stage 3,
test. Stages 2 and 3 start from the same point with the same number of epochs, so they can be compared fairly.
If a model crashes, the error is logged and the next model still runs. Xception goes first because it needs the most
GPU memory.
"""))
    cells.append(("code", '''
RUN_ORDER = ["Xception", "InceptionV3", "VGG16", "MobileNetV2", "BaselineCNN"]   # biggest first, while memory is empty
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
                model.save(f"{OUT}/model_{name}_stage3.keras")
    except Exception:
        traceback.print_exc()
        open(f"{OUT}/errors.txt", "a").write(f"{name}\\n{traceback.format_exc()}\\n")
    finally:
        keras.backend.clear_session(); gc.collect()
'''))

    if protocol == "leakfree":
        cells.append(("md", """
## 10. Part C: train on field labels, test on PCR labels

* Training: the field images of the training pool from section 4, so the field test set stays unseen.
* Test: every PCR image whose group has no member in that training data.
* Model: MobileNetV2, stage 1 then stage 3.

The same model is also scored on the leak-free field test set. A big gap between the two scores would mean the model
learned what decides a field label rather than what the laboratory finds.
"""))
        cells.append(("code", '''
farm_keep = pool[df.loc[pool, "source"].values == "farm"]   # field images of the training pool only
lab_all = df.index[(df["source"] == "lab")].values
lab_all = lab_all[~np.isin(lab_all, conflict)]
lab_test = lab_all[~np.isin(GROUP[lab_all], GROUP[farm_keep])]
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
## Results next to the paper's Table 2
"""))
    cells.append(("code", '''
res = pd.read_csv(RESULTS_CSV)
t = res[res["eval_set"] == "test"].pivot_table(index="model", columns="stage", values="accuracy")
t = t.reindex([m for m in MODELS if m in t.index])          # back to the paper's order
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
# learning curves for the two models the paper talks about most
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


for protocol, smoke, folder, slug, title, nbname in [
    ("paper", False, "02_paper_replication", "poultry-feces-02-paper-replication", "Poultry Feces 02: Paper Replication",
     "02_paper_replication.ipynb"),
    ("leakfree", False, "03_leak_free", "poultry-feces-03-leak-free", "Poultry Feces 03: Leak-Free", "03_leak_free.ipynb"),
    ("leakfree", True, "smoke_test", "poultry-feces-smoke-test", "Poultry Feces Smoke Test", "smoke_test.ipynb"),
]:
    out = os.path.join(ROOT, folder)
    os.makedirs(out, exist_ok=True)
    make_notebook(cells_for(protocol, smoke), os.path.join(out, nbname))
    write_metadata(out, slug, title, nbname, gpu=True, datasets=[DATASET], private=smoke)
    print("built", folder)
