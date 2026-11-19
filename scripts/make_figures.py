"""Draws every figure in figures/ from the CSV and JSON files in results/.

    python scripts/make_figures.py                         # charts only
    python scripts/make_figures.py --images path/to/Train  # charts plus the three photo figures

The photo figures need the Kaggle images (allandclive/chicken-disease-1), which are not stored in this repository.
Without --images the existing photo figures are left as they are.

One colour means one thing in every figure:
    navy    the paper's own numbers
    blue    our rebuild on the paper's split
    green   our tests with copies kept apart, scored on field labels
    orange  scored on laboratory (PCR) labels
    grey    shortcut tests (floor only, colour only)
"""
import argparse
import os
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("DROPCHECK_RESULTS", os.path.join(ROOT, "results"))
FIG = os.path.join(ROOT, "figures")

NAVY, BLUE, GREEN, ORANGE = "#1b3f73", "#2a78d6", "#3a9a5b", "#d9822b"
GREY, PALE = "#8a8a8a", "#c8c8c8"
INK = "#222222"

plt.rcParams.update({
    "font.size": 14.4, "axes.titlesize": 15.6, "axes.labelsize": 14.4, "xtick.labelsize": 13.8, "ytick.labelsize": 13.8,
    "legend.fontsize": 13.8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

CLASSES = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]   # spelling as in the dataset
CLASS_NAMES = ["Coccidiosis", "Healthy", "Newcastle\ndisease", "Salmonella"]
SHORT = ["cocci", "healthy", "ncd", "salmo"]
MODELS = ["BaselineCNN", "VGG16", "InceptionV3", "MobileNetV2", "Xception"]
PAPER_ACC = {"BaselineCNN": 83.06, "VGG16": 95.01, "InceptionV3": 95.45, "MobileNetV2": 98.02, "Xception": 98.24}
PAPER_F1_MOBILENET = [0.99, 0.97, 0.94, 0.99]   # Table 2 of the paper, stage 3


def rnd(v, digits=1):
    """Round half up, so 80.05 prints as 80.1 like in the tables (Python's format would give 80.0)."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP))


def title(fig, text, sub=None):
    # plain fig.text rather than suptitle: tight_layout would reserve its own space for a suptitle on top of rect
    fig.text(0.01, 0.995, text, ha="left", va="top", fontsize=17.4, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 0.925, sub, ha="left", va="top", fontsize=13.2, color="#555555")


def top(sub):
    return 0.86 if sub else 0.93


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("  ", name)


def bar_labels(ax, bars, digits=1, suffix="", horizontal=False, pad=0.6, size=13.2):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if np.isnan(v):
            continue
        text = rnd(v, digits) + suffix
        if horizontal:
            ax.text(v + pad, b.get_y() + b.get_height() / 2, text, va="center", fontsize=size, fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + pad, text, ha="center", fontsize=size, fontweight="bold")


def legend_below(ax, ncol, y=-0.16):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=1.2)


def results(folder):
    return pd.read_csv(f"{RES}/{folder}/results.csv")


def row(res, model, stage, eval_set="test"):
    return res[(res.model == model) & (res.stage == stage) & (res.eval_set == eval_set)].iloc[0]


def recall_from_cm(path):
    cm = pd.read_csv(path, index_col=0).values
    return 100 * np.diag(cm) / cm.sum(1), 100 * np.trace(cm) / cm.sum()


# charts

def fig_summary():
    """The whole project in one chart."""
    s4 = pd.read_csv(f"{RES}/04_stability_and_recipe/summary.csv").set_index(["recipe", "test_set"])
    s5 = pd.read_csv(f"{RES}/05_shortcuts_and_labels/summary.csv").set_index("test")
    rp, rl = results("02_paper_split"), results("03_leak_free")
    items = [
        ("The paper reports", PAPER_ACC["MobileNetV2"], NAVY),
        ("Our rebuild, the paper's split", row(rp, "MobileNetV2", 3)["accuracy"], BLUE),
        ("Copies kept apart", row(rl, "MobileNetV2", 3)["accuracy"], GREEN),
        ("Scored against the lab (PCR)", row(rl, "C_farm_only_MobileNetV2", 3, "pcr_test")["accuracy"], ORANGE),
        ("Lab photos also in training *", s4.loc[("corrected", "pcr"), "accuracy"], ORANGE),
        ("Dropping hidden, floor only *", s5.loc["H2_floor_only", "accuracy"], GREY),
        ("Colour only, no deep learning *", s5.loc["H5_colour_farm", "accuracy"], GREY),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(items))[::-1]
    bars = ax.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.66)
    bar_labels(ax, bars, suffix="%", horizontal=True, pad=0.4, size=14.4)
    ax.set_yticks(y, [n for n, _, _ in items], fontsize=14.4)
    ax.set_xlim(50, 102)
    ax.set_xlabel("Test accuracy, MobileNetV2 (%)")
    ax.tick_params(axis="y", length=0)
    title(fig, "98% on field labels, about 88% against the laboratory")
    fig.text(0.01, -0.02, "* mean of 5 folds. Bars start at 50%.", fontsize=12, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, top(False)))
    save(fig, "summary.png")


def fig_replication():
    """Paper vs our rebuild vs the duplicate-free split, final stage of each model."""
    rp, rl = results("02_paper_split"), results("03_leak_free")
    stage = lambda m: 1 if m == "BaselineCNN" else 3
    series = [("Paper (Table 2)", NAVY, [PAPER_ACC[m] for m in MODELS]),
              ("Our rebuild, paper's split", BLUE, [row(rp, m, stage(m))["accuracy"] for m in MODELS]),
              ("Copies kept apart", GREEN, [row(rl, m, stage(m))["accuracy"] for m in MODELS])]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    w = 0.28
    for k, (name, colour, vals) in enumerate(series):
        bars = ax.bar(np.arange(len(MODELS)) + (k - 1) * w, vals, w, color=colour, label=name)
        bar_labels(ax, bars, pad=0.4, size=11)
    ax.set_xticks(range(len(MODELS)), ["Baseline CNN", "VGG16", "InceptionV3", "MobileNetV2", "Xception"])
    ax.set_ylim(60, 101)
    ax.set_ylabel("Test accuracy (%)")
    legend_below(ax, 3, y=-0.1)
    title(fig, "The paper's numbers come out again for its main models")
    fig.tight_layout(rect=(0, 0, 1, top(False)))
    save(fig, "replication.png")


def per_class_bars(ax, series, ylim=(0, 108)):
    w = 0.8 / len(series)
    for k, (name, colour, vals) in enumerate(series):
        x = np.arange(4) + (k - (len(series) - 1) / 2) * w
        bars = ax.bar(x, vals, w, color=colour, label=name)
        bar_labels(ax, bars, digits=0, pad=1.2, size=12.6)
    ax.set_xticks(range(4), CLASS_NAMES)
    ax.set_ylim(*ylim)


def fig_per_class_f1():
    """MobileNetV2 F1 per class: paper, our rebuild, and copies kept apart."""
    rp, rl = results("02_paper_split"), results("03_leak_free")
    a, b = row(rp, "MobileNetV2", 3), row(rl, "MobileNetV2", 3)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    per_class_bars(ax, [("Paper (Table 2)", NAVY, [100 * v for v in PAPER_F1_MOBILENET]),
                        ("Our rebuild, paper's split", BLUE, [100 * a[f"f1_{s}"] for s in SHORT]),
                        ("Copies kept apart", GREEN, [100 * b[f"f1_{s}"] for s in SHORT])], ylim=(50, 106))
    ax.set_ylabel("F1 score x 100")
    legend_below(ax, 3, y=-0.21)
    title(fig, "Removing copies hurts Newcastle disease most",
          "MobileNetV2, final stage. F1 combines how many cases are found and how many alarms are right.")
    fig.tight_layout(rect=(0, 0, 1, top(True)))
    save(fig, "per_class_f1.png")


def fig_field_vs_pcr():
    """One model trained on field photos, scored against field labels and against PCR labels (notebook 03, part C)."""
    field, field_all = recall_from_cm(f"{RES}/03_leak_free/cm_farm_test_C_farm_only_MobileNetV2_stage3.csv")
    pcr, pcr_all = recall_from_cm(f"{RES}/03_leak_free/cm_pcr_test_C_farm_only_MobileNetV2_stage3.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    per_class_bars(ax, [(f"Field labels (all classes {rnd(field_all)}%)", GREEN, field),
                        (f"Lab (PCR) labels (all classes {rnd(pcr_all)}%)", ORANGE, pcr)])
    ax.set_ylabel("Cases found (recall, %)")
    legend_below(ax, 2, y=-0.21)
    title(fig, "Trained on field photos only, it matches the lab 7 times in 10",
          "MobileNetV2 trained on field photos, then scored against two answer keys.")
    fig.tight_layout(rect=(0, 0, 1, top(True)))
    save(fig, "field_vs_pcr.png")


def fig_per_class_second_run():
    """Notebook 04, corrected recipe: recall per class on unseen field and PCR photos, mean of 5 folds."""
    s4 = pd.read_csv(f"{RES}/04_stability_and_recipe/summary.csv").set_index(["recipe", "test_set"])
    f, p = s4.loc[("corrected", "farm")], s4.loc[("corrected", "pcr")]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    per_class_bars(ax, [(f"Field photos (all classes {rnd(f['accuracy'])}%)", GREEN, [f[f"recall_{s}"] for s in SHORT]),
                        (f"Lab (PCR) photos (all classes {rnd(p['accuracy'])}%)", ORANGE, [p[f"recall_{s}"] for s in SHORT])])
    ax.set_ylabel("Cases found (recall, %)")
    legend_below(ax, 2, y=-0.21)
    title(fig, "With lab photos in training the gap shrinks, but stays",
          "MobileNetV2 trained on field and PCR photos together, as in the paper. Mean of 5 folds.")
    fig.tight_layout(rect=(0, 0, 1, top(True)))
    save(fig, "per_class_second_run.png")


def fig_second_run():
    """Every second-run test in one chart: accuracy with the 95% interval over 5 folds."""
    s4 = pd.read_csv(f"{RES}/04_stability_and_recipe/summary.csv").set_index(["recipe", "test_set"])
    s5 = pd.read_csv(f"{RES}/05_shortcuts_and_labels/summary.csv").set_index("test")
    pick = lambda r: (r["accuracy"], r["accuracy_lo"], r["accuracy_hi"])
    nan = (np.nan, np.nan, np.nan)
    rows = [
        ("Whole photo,\nnetwork fine-tuned", pick(s4.loc[("corrected", "farm")]), pick(s4.loc[("corrected", "pcr")])),
        ("Whole photo,\nnetwork not trained (frozen)", pick(s5.loc["H5_deep_frozen_farm"]), pick(s5.loc["H5_deep_frozen_pcr"])),
        ("Same-size sets,\n1,217 photos each", pick(s5.loc["H3_farm_matched"]), pick(s5.loc["H3_pcr_only"])),
        ("Floor only,\ndropping hidden", pick(s5.loc["H2_floor_only"]), nan),
        ("Colour only,\nno deep learning", pick(s5.loc["H5_colour_farm"]), pick(s5.loc["H5_colour_pcr"])),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(rows))[::-1] * 1.0
    h = 0.36
    for k, (name, colour) in enumerate([("Field labels", GREEN), ("Lab (PCR) labels", ORANGE)]):
        vals = np.array([r[1 + k] for r in rows])
        ys = y + (h / 2 if k == 0 else -h / 2)
        bars = ax.barh(ys, vals[:, 0], h, color=colour, label=name, zorder=2)
        ok = ~np.isnan(vals[:, 0])
        ax.errorbar(vals[ok, 0], ys[ok], xerr=[vals[ok, 0] - vals[ok, 1], vals[ok, 2] - vals[ok, 0]],
                    fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3, zorder=3)
        for b, (m, lo, hi) in zip(bars, vals):   # value inside the bar, left of the interval
            if not np.isnan(m):
                ax.text(lo - 0.6, b.get_y() + b.get_height() / 2, rnd(m), va="center", ha="right",
                        fontsize=13.2, fontweight="bold", color="white", zorder=4)
    ax.text(50.6, y[3] - h / 2, "not tested", va="center", fontsize=12, color="#666666", style="italic")
    ax.axvline(PAPER_ACC["MobileNetV2"], color=NAVY, lw=1.6, ls="--", zorder=1)
    ax.text(PAPER_ACC["MobileNetV2"], 1.01, "paper: 98.0", transform=ax.get_xaxis_transform(), color=NAVY,
            ha="center", va="bottom", fontsize=12.6, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=13.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(50, 104)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_xlabel("Test accuracy (%), mean of 5 folds with 95% interval")
    legend_below(ax, 2, y=-0.13)
    title(fig, "What the accuracy is made of")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "second_run.png")


# photo figures (need the images)

def load(img_dir, f):
    from PIL import Image
    return np.asarray(Image.open(os.path.join(img_dir, f)).convert("RGB").resize((224, 224)))


def fig_class_examples(img_dir):
    """Three field photos of each class, no file names."""
    folds = pd.read_csv(f"{RES}/04_stability_and_recipe/folds.csv")
    clean = folds[(folds.fold >= 0) & (folds.source == "farm")]
    fig, ax = plt.subplots(3, 4, figsize=(8.6, 7.0), gridspec_kw=dict(hspace=0.05, wspace=0.05))
    for c, cls in enumerate(CLASSES):
        files = clean[clean.label == cls].sample(3, random_state=7)["file"]
        for r, f in enumerate(files):
            ax[r, c].imshow(load(img_dir, f))
            ax[r, c].axis("off")
        ax[0, c].set_title(CLASS_NAMES[c].replace("\n", " "), fontsize=15.6, fontweight="bold", pad=8)
    title(fig, "Field photos from the dataset, one column per class")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9)
    save(fig, "class_examples.png")


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def fig_copies(img_dir):
    """One example of each kind of copy found by notebook 01."""
    t = pd.read_csv(f"{RES}/01_data_audit/image_table.csv")
    lab = dict(zip(t.file, t.label))
    exact = t[t.duplicated("md5", keep=False) & (t.source == "farm")].sort_values(["md5", "file"])
    exact_pair = exact.groupby("md5")["file"].apply(list).iloc[0][:2]
    # a near-copy: different file, same label, dHash distance 1 to 4
    farm = t[t.source == "farm"].reset_index(drop=True)
    near_pair = None
    for i in range(len(farm)):
        for j in range(i + 1, min(i + 400, len(farm))):
            if farm.md5[i] != farm.md5[j] and farm.label[i] == farm.label[j] and 1 <= hamming(farm.dhash[i], farm.dhash[j]) <= 4:
                near_pair = [farm.file[i], farm.file[j]]
                break
        if near_pair:
            break
    pairs = [("Identical file,\nsame label", exact_pair),
             ("Identical file,\ntwo labels", ["pcrcocci.228.jpg", "pcrhealthy.83.jpg"]),
             ("Saved again,\ntwo labels", ["pcrcocci.118.jpg", "pcrhealthy.108.jpg"]),
             ("Near-copy,\nsame label", near_pair)]
    fig, ax = plt.subplots(2, 4, figsize=(8.6, 5.6))
    for c, (kind, (f1, f2)) in enumerate(pairs):
        two_labels = lab[f1] != lab[f2]
        for r, f in enumerate((f1, f2)):
            ax[r, c].imshow(load(img_dir, f))
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s in ax[r, c].spines.values():
                s.set_visible(True); s.set_color(ORANGE if two_labels else GREEN); s.set_linewidth(3)
            ax[r, c].set_xlabel(lab[f].replace("New Castle Disease", "Newcastle disease"), fontsize=13.2,
                                fontweight="bold" if two_labels else "normal", color=ORANGE if two_labels else INK)
        ax[0, c].set_title(kind, fontsize=14.4, fontweight="bold", pad=8)
    title(fig, "The same photo can appear twice, sometimes with two labels")
    fig.text(0.01, -0.01, "Orange frame: the two copies carry different labels, so at least one label is wrong.",
             fontsize=12.6, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, top(False)))
    save(fig, "copies.png")
    return pairs


def fig_floor_only(img_dir):
    """What the floor-only model sees, next to what it scores."""
    s4 = pd.read_csv(f"{RES}/04_stability_and_recipe/summary.csv").set_index(["recipe", "test_set"])
    s5 = pd.read_csv(f"{RES}/05_shortcuts_and_labels/summary.csv").set_index("test")
    folds = pd.read_csv(f"{RES}/04_stability_and_recipe/folds.csv")
    clean = folds[(folds.fold >= 0) & (folds.source == "farm")]
    majority = 100 * clean.label.value_counts(normalize=True).max()

    frame = 16                                  # same frame as notebook 05
    ring = np.zeros((224, 224, 1), np.float32)
    ring[:frame], ring[-frame:], ring[:, :frame], ring[:, -frame:] = 1, 1, 1, 1

    W, H = 10.5, 4.15                           # inches; panels are placed by hand so nothing overlaps
    fig = plt.figure(figsize=(W, H))
    side, gap, left = 1.45, 0.14, 0.75
    rows_y = [H - 0.7 - side, H - 0.7 - 2 * side - 0.12]
    for c, cls in enumerate(CLASSES):
        f = clean[clean.label == cls].sample(1, random_state=11)["file"].iloc[0]
        x = load(img_dir, f).astype(np.float32)
        for r, img in enumerate((x, x * ring + 127.5 * (1 - ring))):
            a = fig.add_axes([(left + c * (side + gap)) / W, rows_y[r] / H, side / W, side / H])
            a.imshow(img.astype(np.uint8))
            a.axis("off")
            if r == 0:
                a.set_title(CLASS_NAMES[c].replace("\n", " "), fontsize=12, fontweight="bold", pad=4)
    for r, text in enumerate(("The photo", "What the\nmodel sees")):
        fig.text(0.5 / W, (rows_y[r] + side / 2) / H, text, rotation=90, va="center", ha="center",
                 fontsize=13.2, color="#444444")

    items = [("Whole photo", s4.loc[("corrected", "farm"), "accuracy"], GREEN),
             ("Floor only, dropping hidden", s5.loc["H2_floor_only", "accuracy"], GREY),
             ("Always guess the commonest class", majority, PALE)]
    bx = left + 4 * side + 3 * gap + 0.55
    ax = fig.add_axes([bx / W, rows_y[1] / H, (W - bx - 0.2) / W, (2 * side + 0.12) / H])
    y = np.arange(3)[::-1] * 1.0
    bars = ax.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.5)
    bar_labels(ax, bars, suffix="%", horizontal=True, pad=1.5, size=14.4)
    for yi, (name, _, _) in zip(y, items):      # names above the bars instead of tick labels
        ax.text(0, yi + 0.31, name, va="bottom", fontsize=13.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.4, 2.75)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Accuracy on field labels (%)")
    title(fig, "With the dropping hidden, the model still gets 9 in 10 right")
    save(fig, "floor_only.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder with the Kaggle images (Train/)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    print("charts")
    fig_summary()
    fig_replication()
    fig_per_class_f1()
    fig_field_vs_pcr()
    fig_per_class_second_run()
    fig_second_run()
    if args.images:
        print("photo figures")
        fig_class_examples(args.images)
        print("   copy pairs:", fig_copies(args.images))
        fig_floor_only(args.images)
