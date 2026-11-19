"""Writes the second-run notebooks, 04 and 05, and their smoke tests.

The first run (02, 03) showed the paper can be repeated. The second run asks why the numbers look the way they do.
The questions were written down before the runs:

  04  H1  Is the accuracy stable over 5 duplicate-free folds, and does the paper's training recipe matter?
          Each fold's model is also scored on the PCR images of its test fold, which it never saw.
  05  H2  Can the label be predicted from the floor alone, with the dropping painted out?
      H3  Can PCR labels be learned as well as field labels, at the same sample size?
      H4  Can a model tell a PCR photo from a field photo?
      H5  How far does colour alone get? (colour histogram + logistic regression, no deep learning)

Both notebooks rebuild the same folds from the same seed, so they can be compared fold by fold.

Run:  python build_second_run.py
"""
import os
from nb_tools import make_notebook, write_metadata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "kernels")
DATASET = "allandclive/chicken-disease-1"


# cells used by both notebooks

def settings_cell(smoke, nb):
    return ("code", f'''
SMOKE = {smoke}
SEED = 42
N_FOLDS = 2 if SMOKE else 5
NEAR_DUP_MAX_HAMMING = 4   # same grouping rule as notebook 03

# MobileNetV2, the model the paper recommends. Stage 1 trains the new top only; stage 3 unfreezes the whole
# network but keeps BatchNorm frozen, which was the best setting in the paper and in my first run.
LR_HEAD, LR_FINE = 1e-3, 1e-5
EPOCHS_HEAD = 1 if SMOKE else 20
EPOCHS_FINE = 1 if SMOKE else 30
PATIENCE_HEAD, PATIENCE_FINE = 5, 8
BATCH = 64

OUT = "/kaggle/working/results_{nb}"
''')


IMPORTS = ("code", '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc, glob, json, time, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             confusion_matrix, roc_auc_score)

import tensorflow as tf
import keras
from keras import layers

keras.mixed_precision.set_global_policy("mixed_float16")
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
os.makedirs(OUT, exist_ok=True)

versions = {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "gpus": len(tf.config.list_physical_devices("GPU"))}
print(versions)
json.dump(versions, open(f"{OUT}/versions.json", "w"))
''')


DATA_MD = ("md", """
## Data, duplicate groups and folds

Same preparation as notebook 03, so the clean set is the same:

1. Read the 8,067 images. A `pcr` prefix in the file name means a laboratory (PCR) label.
2. Group images that are the same photo: identical files (same MD5) or nearly identical pictures (dHash distance
   of 4 bits or less out of 64). A group never sits on both sides of a split.
3. Drop groups whose images carry different labels, and keep one copy of each identical file.
4. Cut the clean set into 5 folds with `StratifiedGroupKFold` (seed 42): about the same class mix in each fold,
   each group in one fold only. Every fold takes one turn as the test fold.
""")

DATA_CODE = ("code", '''
csv_path = glob.glob("/kaggle/input/**/train_data.csv", recursive=True)[0]
IMG_DIR = os.path.join(os.path.dirname(csv_path), "Train")
df = pd.read_csv(csv_path)
df.columns = ["file", "label"]
df["source"] = np.where(df["file"].str.startswith("pcr"), "lab", "farm")
CLASSES = sorted(df["label"].unique())
SHORT = ["cocci", "healthy", "ncd", "salmo"]
df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})

if SMOKE:
    df = (df.groupby(["label", "source"], group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 40), random_state=SEED))
            .reset_index(drop=True))

def load(f):
    img = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
    return np.asarray(img if img.size == (224, 224) else img.resize((224, 224)), dtype=np.uint8)

X = np.stack([load(f) for f in df["file"]])
Y = df["y"].values.astype(np.int32)
df["md5"] = [hashlib.md5(open(os.path.join(IMG_DIR, f), "rb").read()).hexdigest() for f in df["file"]]

def dhash_bits(arr):
    a = np.asarray(Image.fromarray(arr).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

B = np.stack([dhash_bits(x) for x in X]).astype(np.float32)
ham = np.rint(B @ (1 - B).T + (1 - B) @ B.T).astype(np.int16)
same = df["md5"].values[:, None] == df["md5"].values[None, :]
link = same | (ham <= NEAR_DUP_MAX_HAMMING)
np.fill_diagonal(link, False)
_, GROUP = connected_components(csr_matrix(link), directed=False)
del ham, same, link

conflict = df.groupby(GROUP)["label"].transform("nunique").values > 1
keep = df.index[~conflict & ~df.duplicated("md5")].values
print(f"{len(df)} images, {conflict.sum()} removed (conflicting labels), {len(keep)} kept")

FOLD = np.full(len(df), -1)
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for k, (_, test_pos) in enumerate(sgkf.split(keep, Y[keep], groups=GROUP[keep])):
    FOLD[keep[test_pos]] = k
df["fold"] = FOLD
df[["file", "label", "source", "md5", "fold"]].assign(group=GROUP).to_csv(f"{OUT}/folds.csv", index=False)
print(pd.crosstab([df.loc[keep, "source"], df.loc[keep, "label"]], df.loc[keep, "fold"]))
''')


PIPE_MD = ("md", """
## Training recipes and the model

Two recipes, everything else (model, layers, learning rates, epochs) the same.

| | Paper recipe | Corrected recipe |
|---|---|---|
| Pixel scaling | 0 to 1 (as in the paper) | -1 to 1 (what MobileNetV2's ImageNet weights expect) |
| Brightness | multiply by 0.2 to 1.0 (training images only ever get darker) | multiply by 0.8 to 1.2 (darker or brighter) |
| Flips | vertical | vertical and horizontal |
| Rotation | up to 45 degrees | up to 45 degrees |

The model is MobileNetV2 with the paper's top: global average pooling, dropout 0.2, softmax. It trains in two
steps, the top alone (stage 1) and then the whole network with BatchNorm frozen (stage 3). Validation data (about
1/7 of the training folds, split by group as well) decides when to stop and which epoch to keep. The test fold is
never used for any choice.

`mask_centre=True` paints the middle of each image grey after augmentation and leaves a 16-pixel frame of floor.
Notebook 05 uses it for the floor-only test.
""")

PIPE_CODE = ("code", '''
RECIPES = {
    "paper":     dict(scale="0_1",  flip="vertical",                rot=45, bright=(0.2, 1.0)),
    "corrected": dict(scale="-1_1", flip="horizontal_and_vertical", rot=45, bright=(0.8, 1.2)),
}
FRAME = 16   # 32 px was tried first, but droppings often reached into a frame that wide
ring = np.zeros((224, 224, 1), np.float32)
ring[:FRAME], ring[-FRAME:], ring[:, :FRAME], ring[:, -FRAME:] = 1, 1, 1, 1
RING = tf.constant(ring)

def make_ds(idx, recipe, train, mask_centre=False, labels=None, batch=BATCH, seed=SEED):
    r = RECIPES[recipe]
    labels = Y[idx] if labels is None else labels
    aug = [layers.RandomFlip(r["flip"], seed=seed, dtype="float32"),
           layers.RandomRotation(r["rot"] / 360, fill_mode="nearest", seed=seed, dtype="float32")] if train else []

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        for a in aug:
            x = tf.cast(a(x, training=True), tf.float32)
        if train:
            f = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], *r["bright"], seed=seed)
            x = tf.clip_by_value(x * f, 0.0, 255.0)
        if mask_centre:
            x = x * RING + 127.5 * (1.0 - RING)       # grey centre, frame untouched
        x = x / 255.0 if r["scale"] == "0_1" else x / 127.5 - 1.0
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((X[idx], labels))
    if train:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

def build_model(n_classes=4):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    inp = keras.Input((224, 224, 3))
    x = layers.GlobalAveragePooling2D()(base(inp))
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out), base

def split_val(pool, seed):
    """About 1/7 of the pool for validation, split by group."""
    cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    tr, va = next(cv.split(pool, Y[pool], groups=GROUP[pool]))
    return pool[tr], pool[va]

def train(train_idx, val_idx, recipe, seed, mask_centre=False, labels_of=None, n_classes=4, tag=""):
    """Stage 1 (top only), then stage 3 (all layers, BatchNorm frozen)."""
    keras.utils.set_random_seed(seed)
    model, base = build_model(n_classes)
    lab = (lambda i: Y[i]) if labels_of is None else labels_of
    tr = make_ds(train_idx, recipe, True, mask_centre, lab(train_idx), seed=seed)
    va = make_ds(val_idx, recipe, False, mask_centre, lab(val_idx))
    log = []
    for stage, lr, epochs, patience in ((1, LR_HEAD, EPOCHS_HEAD, PATIENCE_HEAD), (3, LR_FINE, EPOCHS_FINE, PATIENCE_FINE)):
        base.trainable = stage == 3
        if stage == 3:
            for l in base.layers:
                if isinstance(l, layers.BatchNormalization):
                    l.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"], jit_compile=False)
        stop = keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True)
        h = model.fit(tr, validation_data=va, epochs=epochs, callbacks=[stop], verbose=0)
        hist = pd.DataFrame(h.history).assign(stage=stage, epoch=lambda d: np.arange(1, len(d) + 1))
        log.append(hist)
        best = int(np.argmin(h.history["val_loss"]))
        print(f"   {tag} stage {stage}: {len(hist)} epochs, best val_loss {h.history['val_loss'][best]:.3f} "
              f"(val acc {h.history['val_accuracy'][best]:.3f})")
    pd.concat(log).to_csv(f"{OUT}/history_{tag}.csv", index=False)
    return model

def predict(model, idx, recipe, mask_centre=False):
    return model.predict(make_ds(idx, recipe, False, mask_centre, np.zeros(len(idx), np.int32)), verbose=0)

def scores(y, p):
    """Accuracy, balanced accuracy, macro F1 and recall per class."""
    pred = p.argmax(1)
    labels = list(range(p.shape[1]))
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    out = {"n": len(y), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0)}
    if p.shape[1] == 4:
        out.update({f"recall_{s}": r for s, r in zip(SHORT, rec)})
    return out

def mean_ci(values):
    """Mean and 95% CI over folds, t distribution with n-1 degrees of freedom."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return v.mean(), np.nan, np.nan
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half
''')


def cells_04(smoke):
    c = [("md", """
# Poultry Feces 04: Stability and Recipe

**Paper:** Machuve et al. (2022), Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

The first run gave one number per model from one split, and one split can be lucky or unlucky. Here MobileNetV2,
the model the paper recommends for farmers' phones, is trained 5 times, once per fold, with two training recipes.
That is 10 models, and a mean with a confidence interval for each recipe.

**Questions (written down before the run)**
* **H1a.** On field photos, what is the mean accuracy over 5 duplicate-free folds, and how much does it move?
* **H1b.** Does the paper's recipe (pixels scaled 0 to 1, training images always darkened) cost accuracy against a
  corrected recipe? Compared fold by fold, since both recipes see exactly the same folds.
* **H1c.** The models train on field and PCR photos mixed, as in the paper. How do they score on the PCR photos of
  the test fold, which they never saw? This is the paper's own design, scored against the laboratory.
"""), settings_cell(smoke, "04"), IMPORTS, DATA_MD, DATA_CODE, PIPE_MD, PIPE_CODE,
        ("md", """
## Run: 5 folds x 2 recipes

For fold k the test sets are the field photos and the PCR photos of fold k. Training uses all clean photos of the
other folds, both kinds mixed. Predictions on the test fold are kept, so at the end every clean photo has exactly
one out-of-fold prediction per recipe.
"""),
        ("code", '''
rows, oof = [], []
for k in range(N_FOLDS):
    test_farm = keep[(FOLD[keep] == k) & (df.loc[keep, "source"].values == "farm")]
    test_lab = keep[(FOLD[keep] == k) & (df.loc[keep, "source"].values == "lab")]
    pool = keep[FOLD[keep] != k]
    tr, va = split_val(pool, SEED + k)
    assert not set(GROUP[pool]) & set(GROUP[np.concatenate([test_farm, test_lab])]), "a group crosses the split"
    for recipe in RECIPES:
        t0 = time.time()
        print(f"fold {k}, {recipe} recipe: train {len(tr)}, val {len(va)}, test farm {len(test_farm)}, test lab {len(test_lab)}")
        model = train(tr, va, recipe, seed=SEED + k, tag=f"fold{k}_{recipe}")
        for name, idx in (("farm", test_farm), ("pcr", test_lab)):
            p = predict(model, idx, recipe)
            rows.append({"fold": k, "recipe": recipe, "test_set": name, **scores(Y[idx], p)})
            oof.append(pd.DataFrame({"file": df.loc[idx, "file"].values, "fold": k, "recipe": recipe, "test_set": name,
                                     "y": Y[idx], "pred": p.argmax(1), **{f"p_{s}": p[:, j] for j, s in enumerate(SHORT)}}))
        r = rows[-2]
        print(f"   farm acc {r['accuracy']:.4f} | PCR acc {rows[-1]['accuracy']:.4f} | {(time.time()-t0)/60:.1f} min")
        pd.DataFrame(rows).to_csv(f"{OUT}/fold_scores.csv", index=False)
        pd.concat(oof).to_csv(f"{OUT}/oof_predictions.csv", index=False)
        del model; keras.backend.clear_session(); gc.collect()
'''),
        ("md", """
## Results

* Mean and 95% confidence interval over the 5 folds, per recipe and test set.
* Recipes compared in pairs: corrected minus paper in each fold, the mean difference with its 95% interval, and a
  paired t-test. Five folds make a weak test, so read the interval before the p-value.
"""),
        ("code", '''
res = pd.DataFrame(rows)
summary = []
for (recipe, test_set), g in res.groupby(["recipe", "test_set"]):
    row = {"recipe": recipe, "test_set": test_set, "folds": len(g)}
    for m in ["accuracy", "balanced_accuracy", "macro_f1"] + [f"recall_{s}" for s in SHORT]:
        mu, lo, hi = mean_ci(g[m])
        row[m] = round(100 * mu, 2); row[f"{m}_lo"] = round(100 * lo, 2); row[f"{m}_hi"] = round(100 * hi, 2)
    row["accuracy_sd"] = round(100 * g["accuracy"].std(ddof=1), 2)
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(f"{OUT}/summary.csv", index=False)
print(summary[["recipe", "test_set", "folds", "accuracy", "accuracy_lo", "accuracy_hi", "accuracy_sd",
               "balanced_accuracy", "recall_cocci", "recall_healthy", "recall_ncd", "recall_salmo"]].to_string(index=False))

paired = []
for test_set in ["farm", "pcr"]:
    for m in ["accuracy", "balanced_accuracy"]:
        a = res[(res.recipe == "corrected") & (res.test_set == test_set)].sort_values("fold")[m].values
        b = res[(res.recipe == "paper") & (res.test_set == test_set)].sort_values("fold")[m].values
        d = 100 * (a - b)
        mu, lo, hi = mean_ci(d)
        p = stats.ttest_rel(a, b).pvalue if len(d) > 1 else np.nan
        paired.append({"test_set": test_set, "metric": m, "mean_diff_points": round(mu, 2),
                       "ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "paired_t_p": round(p, 4),
                       "folds_corrected_better": int((d > 0).sum())})
paired = pd.DataFrame(paired)
paired.to_csv(f"{OUT}/paired_recipe_test.csv", index=False)
print(paired.to_string(index=False))
'''),
        ("code", '''
# every clean photo predicted once, by the fold model that never saw it
o = pd.concat(oof)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, test_set, title in ((ax[0], "farm", "Farm labels"), (ax[1], "pcr", "PCR labels")):
    s = o[(o.recipe == "corrected") & (o.test_set == test_set)]
    cm = confusion_matrix(s["y"], s["pred"], labels=range(4))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(f"{OUT}/oof_cm_corrected_{test_set}.csv")
    a.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks(range(4), ["Cocci", "Healthy", "NCD", "Salmo"]); a.set_yticks(range(4), ["Cocci", "Healthy", "NCD", "Salmo"])
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(f"{title}, all folds (n = {len(s)})")
plt.tight_layout(); plt.savefig(f"{OUT}/oof_confusion.png", dpi=150); plt.show()
'''),
    ]
    return c


def cells_05(smoke):
    c = [("md", """
# Poultry Feces 05: Shortcuts and Labels

**Paper:** Machuve et al. (2022), Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

In notebook 03 a model trained on field labels agreed with the laboratory (PCR) only about 72% of the time.
This notebook tests four explanations. Each one has its test and its reading fixed in advance.

| | Question | Test | How to read it |
|---|---|---|---|
| **H2** | Does the floor give the label away? | Train and test on images whose centre is painted grey (only a 16-pixel frame of floor is left). | Far above chance means the scene, not the dropping, carries label information (a shortcut). |
| **H3** | Can PCR labels be learned as well as field labels? | 5-fold CV on PCR photos only, and on a field sample of the same size and class counts. | If PCR is much lower, the disease shows less in the photo than the field labels suggest. |
| **H4** | Were PCR and field photos taken differently? | Logistic regression on frozen ImageNet features predicts field vs PCR. | AUC near 0.5: the sets look alike. Near 1: they differ, and a model trained on one meets a new kind of photo in the other. |
| **H5** | How far does colour alone get? | A colour histogram plus logistic regression, no deep learning. | The paper's field-labelling rules are colour and shape rules. If colour alone comes close to the deep model on field labels, the deep model is mostly learning those rules. |

All tests use the same clean set and the same 5 folds as notebook 04.
"""), settings_cell(smoke, "05"), IMPORTS, DATA_MD, DATA_CODE, PIPE_MD, PIPE_CODE,
        ("md", """
## H2: floor only

First a look at what the mask leaves. Some droppings reach the edge of the photo, so a few frames still show a bit
of dropping. That can only push the floor-only score up, so read it as an upper limit.
"""),
        ("code", '''
show = np.random.RandomState(0).choice(keep, 12, replace=False)
fig, ax = plt.subplots(2, 12, figsize=(18, 3.4))   # top: photo, bottom: what the model sees
masked = (X[show].astype(np.float32) * ring + 127.5 * (1 - ring)).astype(np.uint8)
for j, i in enumerate(show):
    ax[0, j].imshow(X[i]); ax[1, j].imshow(masked[j]); ax[0, j].set_title(df.loc[i, "label"][:10], fontsize=7)
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/mask_examples.png", dpi=120); plt.show()
'''),
        ("code", '''
rows = []
farm_keep = keep[df.loc[keep, "source"].values == "farm"]
for k in range(N_FOLDS):
    test = farm_keep[FOLD[farm_keep] == k]
    tr, va = split_val(farm_keep[FOLD[farm_keep] != k], SEED + k)
    model = train(tr, va, "corrected", seed=SEED + k, mask_centre=True, tag=f"H2_fold{k}")
    p = predict(model, test, "corrected", mask_centre=True)
    rows.append({"test": "H2_floor_only", "fold": k, **scores(Y[test], p)})
    print(f"   fold {k}: floor-only accuracy {rows[-1]['accuracy']:.4f}, balanced {rows[-1]['balanced_accuracy']:.4f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/scores.csv", index=False)
    del model; keras.backend.clear_session(); gc.collect()

# for comparison: always guessing the biggest class, and 25% chance for balanced accuracy
majority = pd.Series(Y[farm_keep]).value_counts(normalize=True).max()
print(f"majority-class accuracy on farm photos: {majority:.4f}; balanced-accuracy chance: 0.25")
'''),
        ("md", """
## H3: can PCR labels be learned from photos?

* **PCR set:** the clean PCR photos. Train on 4 folds, test on the fifth.
* **Field control:** a random sample of clean field photos with the same number per class as the PCR set, drawn once
  (seed 42). Same folds, same training.

Size and class mix are matched, so a large gap would point at the labels, not at the amount of data.
"""),
        ("code", '''
lab_keep = keep[df.loc[keep, "source"].values == "lab"]
need = pd.Series(Y[lab_keep]).value_counts()
rng = np.random.RandomState(SEED)
farm_match = np.concatenate([rng.choice(farm_keep[Y[farm_keep] == c], min(n, (Y[farm_keep] == c).sum()), replace=False)
                             for c, n in need.items()])
print("PCR set per class:", need.sort_index().to_dict())
print("farm control per class:", pd.Series(Y[farm_match]).value_counts().sort_index().to_dict())

for name, subset in (("H3_pcr_only", lab_keep), ("H3_farm_matched", farm_match)):
    for k in range(N_FOLDS):
        test = subset[FOLD[subset] == k]
        rest = subset[FOLD[subset] != k]
        if len(test) == 0 or len(np.unique(Y[rest])) < 4:
            continue
        tr, va = split_val(rest, SEED + k)
        model = train(tr, va, "corrected", seed=SEED + k, tag=f"{name}_fold{k}")
        p = predict(model, test, "corrected")
        rows.append({"test": name, "fold": k, **scores(Y[test], p)})
        print(f"   {name} fold {k}: accuracy {rows[-1]['accuracy']:.4f}, balanced {rows[-1]['balanced_accuracy']:.4f}")
        pd.DataFrame(rows).to_csv(f"{OUT}/scores.csv", index=False)
        del model; keras.backend.clear_session(); gc.collect()
'''),
        ("md", """
## H4 and H5: tests on fixed features, no fine-tuning

* **Frozen features:** MobileNetV2 with its ImageNet weights and no training. Each photo becomes 1,280 numbers.
* **Colour features:** a 3-D colour histogram in HSV (8 hue x 4 saturation x 4 value bins = 128 numbers, summing to 1
  per photo), plus the mean and spread of each RGB channel.
* Both go into a logistic regression (standardised inputs, balanced class weights), fitted on the training folds
  only and scored on the same 5 folds.
"""),
        ("code", '''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

extractor = keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
EMB = extractor.predict(tf.data.Dataset.from_tensor_slices(X[keep]).batch(128)
                        .map(lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0), verbose=0).astype(np.float32)
pos = {i: j for j, i in enumerate(keep)}            # image index -> row in EMB

def colour_features(arr):
    hsv = np.asarray(Image.fromarray(arr).convert("HSV"), dtype=np.float32) / 256.0
    h, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 4, 4), range=((0, 1), (0, 1), (0, 1)))
    rgb = arr.reshape(-1, 3).astype(np.float32) / 255.0
    return np.concatenate([h.ravel() / h.sum(), rgb.mean(0), rgb.std(0)])

COL = np.stack([colour_features(X[i]) for i in keep]).astype(np.float32)

def cv_logreg(feats, idx, target, name):
    """Logistic regression over the 5 folds; returns the out-of-fold probabilities."""
    F = feats[[pos[i] for i in idx]]
    oof = np.zeros((len(idx), len(np.unique(target))))
    for k in range(N_FOLDS):
        te, tr = FOLD[idx] == k, FOLD[idx] != k
        if te.sum() == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
        clf.fit(F[tr], target[tr])
        oof[te] = clf.predict_proba(F[te])
        rows.append({"test": name, "fold": k, **scores(target[te], oof[te])})
    pd.DataFrame(rows).to_csv(f"{OUT}/scores.csv", index=False)
    return oof

# H4: field (0) vs PCR (1), from deep features and from colour
is_lab = (df.loc[keep, "source"].values == "lab").astype(int)
for feats, name in ((EMB, "H4_source_deep"), (COL, "H4_source_colour")):
    p = cv_logreg(feats, keep, is_lab, name)
    auc = roc_auc_score(is_lab, p[:, 1])
    per_class = {CLASSES[c]: round(roc_auc_score(is_lab[Y[keep] == c], p[Y[keep] == c, 1]), 3) for c in range(4)}
    print(f"{name}: farm-vs-lab AUC {auc:.3f}; within each disease class {per_class}")
    json.dump({"auc": auc, "auc_within_class": per_class}, open(f"{OUT}/{name}_auc.json", "w"), indent=1)

# H5: the disease label from colour, and from frozen deep features as a no-fine-tuning reference
for feats, fname in ((COL, "colour"), (EMB, "deep_frozen")):
    for src, idx in (("farm", farm_keep), ("pcr", lab_keep)):
        cv_logreg(feats, idx, Y[idx], f"H5_{fname}_{src}")
'''),
        ("md", """
## Summary table

Mean and 95% confidence interval over the folds for every test. Chance for balanced accuracy is 25%.
The full-photo MobileNetV2 results are in notebook 04.
"""),
        ("code", '''
res = pd.DataFrame(rows)
out = []
for name, g in res.groupby("test", sort=False):
    row = {"test": name, "folds": len(g), "n_total": int(g["n"].sum())}
    for m in ["accuracy", "balanced_accuracy", "macro_f1"]:
        mu, lo, hi = mean_ci(g[m])
        row[m] = round(100 * mu, 2); row[f"{m}_lo"] = round(100 * lo, 2); row[f"{m}_hi"] = round(100 * hi, 2)
    if "recall_ncd" in g:
        row["recall_ncd"] = round(100 * g["recall_ncd"].mean(), 2)
    out.append(row)
summary = pd.DataFrame(out)
summary.to_csv(f"{OUT}/summary.csv", index=False)
print(summary.to_string(index=False))
'''),
    ]
    return c


for nb, smoke, folder, slug, title in [
    ("04", False, "04_stability_and_recipe", "poultry-feces-04-stability-and-recipe", "Poultry Feces 04: Stability and Recipe"),
    ("05", False, "05_shortcuts_and_labels", "poultry-feces-05-shortcuts-and-labels", "Poultry Feces 05: Shortcuts and Labels"),
    ("04", True, "smoke_test_04", "poultry-feces-smoke-test-04", "Poultry Feces Smoke Test 04"),
    ("05", True, "smoke_test_05", "poultry-feces-smoke-test-05", "Poultry Feces Smoke Test 05"),
]:
    out = os.path.join(ROOT, folder)
    os.makedirs(out, exist_ok=True)
    cells = cells_04(smoke) if nb == "04" else cells_05(smoke)
    if smoke:
        cells = [("md", "# SMOKE TEST: tiny data, 1 epoch, 2 folds. Only checks that the code runs.")] + cells
    name = f"{folder}.ipynb"
    make_notebook(cells, os.path.join(out, name))
    write_metadata(out, slug, title, name, gpu=True, datasets=[DATASET], private=smoke)
    print("built", folder)
