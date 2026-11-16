"""Rebuild every figure in figures/ from the saved results in results/.

    python scripts/make_figures.py

Needs pandas and matplotlib only. No images or models are read, so the figures can be checked against the CSV files.
"""
import os
import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("DROPCHECK_RESULTS", os.path.join(ROOT, "results"))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

NAVY, BLUE, GREEN, GREY = "#1f3a68", "#4a7fc1", "#5a9e5a", "#9a9a9a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})

MODELS = ["BaselineCNN", "VGG16", "InceptionV3", "MobileNetV2", "Xception"]
LABELS = ["Baseline CNN", "VGG16", "InceptionV3", "MobileNetV2", "Xception"]
PAPER = {"BaselineCNN": 83.06, "VGG16": 95.01, "InceptionV3": 95.45, "MobileNetV2": 98.02, "Xception": 98.24}  # Table 2
SHORT = ["Cocci", "Healthy", "NCD", "Salmo"]


def final_accuracy(path, model):
    r = pd.read_csv(path)
    stage = 1 if model == "BaselineCNN" else 3
    row = r[(r["model"] == model) & (r["stage"] == stage) & (r["eval_set"] == "test")]
    return float(row["accuracy"].iloc[0])


def fig_replication():
    """Paper vs our replication vs the duplicate-free split, final stage of each model."""
    series = [("Paper (Table 2)", NAVY, [PAPER[m] for m in MODELS]),
              ("Ours, paper-style split", BLUE, [final_accuracy(f"{RES}/02_paper_split/results.csv", m) for m in MODELS]),
              ("Ours, duplicate-free split", GREEN, [final_accuracy(f"{RES}/03_leak_free/results.csv", m) for m in MODELS])]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.26
    for k, (name, colour, vals) in enumerate(series):
        x = np.arange(len(MODELS)) + (k - 1) * w
        ax.bar(x, vals, w, color=colour, label=name)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(MODELS)), LABELS)
    ax.set_ylim(60, 100)
    ax.set_ylabel("Test accuracy on field photos (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.set_title("The paper's numbers repeat for its main models", loc="left", pad=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig01_replication.png")
    plt.close()


def confusion(ax, cm, title):
    cm = np.asarray(cm)
    share = cm / cm.sum(1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if share[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), SHORT)
    ax.set_yticks(range(4), SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_field_vs_pcr():
    """The same farm-trained MobileNetV2 scored against field labels and against PCR labels (notebook 03, part C)."""
    a = pd.read_csv(f"{RES}/03_leak_free/cm_farm_test_C_farm_only_MobileNetV2_stage3.csv", index_col=0)
    b = pd.read_csv(f"{RES}/03_leak_free/cm_pcr_test_C_farm_only_MobileNetV2_stage3.csv", index_col=0)
    acc = lambda m: 100 * np.trace(m.values) / m.values.sum()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    confusion(ax[0], a.values, f"Field labels: {acc(a):.1f}% correct")
    confusion(ax[1], b.values, f"PCR labels: {acc(b):.1f}% correct")
    fig.suptitle("One model trained on field photos, scored against field labels and against the laboratory",
                 x=0.02, ha="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig03_field_vs_pcr.png")
    plt.close()


def fig_second_run():
    """Balanced accuracy with 95% confidence intervals over 5 folds for every second-run test."""
    s4 = pd.read_csv(f"{RES}/04_stability_and_recipe/summary.csv")
    s5 = pd.read_csv(f"{RES}/05_shortcuts_and_labels/summary.csv")
    rows = []
    for recipe, name in (("paper", "paper recipe"), ("corrected", "corrected recipe")):
        for test_set, tname in (("farm", "field labels"), ("pcr", "PCR labels")):
            r = s4[(s4.recipe == recipe) & (s4.test_set == test_set)].iloc[0]
            rows.append((f"H1  MobileNetV2, {name}: {tname}", r, BLUE if test_set == "farm" else NAVY))
    names5 = {"H2_floor_only": ("H2  Floor only (centre masked): field labels", BLUE),
              "H3_farm_matched": ("H3  Field photos, PCR-sized sample: field labels", BLUE),
              "H3_pcr_only": ("H3  PCR photos only: PCR labels", NAVY),
              "H5_colour_farm": ("H5  Colour histogram only: field labels", GREEN),
              "H5_colour_pcr": ("H5  Colour histogram only: PCR labels", GREEN),
              "H5_deep_frozen_farm": ("H5  Frozen ImageNet features: field labels", GREY),
              "H5_deep_frozen_pcr": ("H5  Frozen ImageNet features: PCR labels", GREY)}
    for key, (name, colour) in names5.items():
        r = s5[s5.test == key]
        if len(r):
            rows.append((name, r.iloc[0], colour))
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.0))
    for i, (name, r, colour) in enumerate(rows[::-1]):
        m = r["balanced_accuracy"]
        lo, hi = max(r["balanced_accuracy_lo"], 0), min(r["balanced_accuracy_hi"], 100)   # a t interval can pass 0 or 100
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=colour, capsize=2, ms=4)
        ax.text(hi + 1, i, f"{m:.1f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7.5)
    ax.axvline(25, color=GREY, lw=0.8, ls="--")
    ax.text(25.5, -0.6, "chance (25%)", fontsize=7, color=GREY)
    ax.set_xlim(15, 108)
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel("Balanced accuracy (%), mean and 95% CI over 5 folds")
    fig.suptitle("Second run: how well each label can be predicted, and from what", x=0.02, ha="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig04_second_run.png")
    plt.close()


def fig_oof_confusion():
    """Out-of-fold confusion matrices of notebook 04 (corrected recipe): every clean photo predicted once."""
    a = pd.read_csv(f"{RES}/04_stability_and_recipe/oof_cm_corrected_farm.csv", index_col=0)
    b = pd.read_csv(f"{RES}/04_stability_and_recipe/oof_cm_corrected_pcr.csv", index_col=0)
    acc = lambda m: 100 * np.trace(m.values) / m.values.sum()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    confusion(ax[0], a.values, f"Field photos (n = {a.values.sum()}): {acc(a):.1f}%")
    confusion(ax[1], b.values, f"PCR photos (n = {b.values.sum()}): {acc(b):.1f}%")
    fig.suptitle("Trained on field and PCR photos mixed, as in the paper (5 folds, unseen photos)",
                 x=0.02, ha="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig05_oof_confusion.png")
    plt.close()


def copy_audit_figures():
    for src, dst in (("pairs_h4.png", "fig02_near_copies.png"), ("class_examples.png", "fig00_class_examples.png")):
        if os.path.exists(f"{FIG}/{src}"):
            shutil.move(f"{FIG}/{src}", f"{FIG}/{dst}")
    m = f"{RES}/05_shortcuts_and_labels/mask_examples.png"
    if os.path.exists(m):
        shutil.copy(m, f"{FIG}/fig06_floor_only_mask.png")


if __name__ == "__main__":
    fig_replication()
    fig_field_vs_pcr()
    copy_audit_figures()
    if os.path.exists(f"{RES}/04_stability_and_recipe/summary.csv") and os.path.exists(f"{RES}/05_shortcuts_and_labels/summary.csv"):
        fig_second_run()
        fig_oof_confusion()
    print(sorted(os.listdir(FIG)))
