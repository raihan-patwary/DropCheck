"""Builds notebook 01: data audit (CPU only, about 5 minutes on Kaggle)."""
import os
from nb_tools import make_notebook, write_metadata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "kernels", "01_data_audit")
os.makedirs(OUT, exist_ok=True)

cells = [
("md", """
# 01 · Data audit: poultry droppings images (Machuve et al. 2022)

**Paper:** Machuve D, Nwankwo E, Mduma N, Mbelwa J (2022). *Poultry diseases diagnostics models using deep learning.*
Frontiers in Artificial Intelligence 5: 733345. https://doi.org/10.3389/frai.2022.733345

**What the paper did, in short.** The authors photographed chicken droppings on farms in Tanzania with phones.
They labelled each image as one of four classes: Coccidiosis, Healthy, Newcastle disease (NCD) or Salmonella.
1,255 images were labelled by a laboratory test (PCR). 6,812 images were labelled by a vet or field officer by eye.
Then they trained five image models. Their best model, Xception, reached 98.24% test accuracy.

**What this notebook does.** Before we train anything, we check the data itself:
1. Do the image counts match Table 1 of the paper?
2. Are there **duplicate images**? If one copy goes into training and the other copy into the test set,
   the model has already seen the test image. The test score then looks better than it really is.
   This problem is called **data leakage**.
3. Are there duplicates that carry **two different disease labels**? That would be a labelling error.

Nothing here needs a GPU.
"""),
("code", """
# --- Imports -------------------------------------------------------------
# os/glob: find files.  hashlib: make a fingerprint (MD5) of each file.
# numpy/pandas: numbers and tables.  PIL: open images.  matplotlib: plots.
import os, glob, hashlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

OUT = "/kaggle/working/audit"          # everything we save goes here
os.makedirs(OUT, exist_ok=True)

# Kaggle mounts the dataset under /kaggle/input. The exact sub-folder can change,
# so we search for the label file instead of typing a fixed path.
csv_path = glob.glob("/kaggle/input/**/train_data.csv", recursive=True)[0]
IMG_DIR = os.path.join(os.path.dirname(csv_path), "Train")
print("label file :", csv_path)
print("image folder:", IMG_DIR)
"""),
("md", """
## 1. Image counts compared with Table 1 of the paper

File names tell us two things. The prefix `pcr` means the image was labelled by the **laboratory (PCR)**.
No prefix means it was labelled **on the farm** by eye. Example: `pcrsalmo.12.jpg` is a lab-confirmed Salmonella image.
"""),
("code", """
df = pd.read_csv(csv_path)                     # two columns: images (file name), label (class)
df.columns = ["file", "label"]

# Source of the label: 'lab' if the file name starts with 'pcr', otherwise 'farm'.
df["source"] = np.where(df["file"].str.startswith("pcr"), "lab", "farm")

# Short class code taken from the file name (cocci / healthy / ncd / salmo).
df["code"] = df["file"].str.replace("pcr", "", regex=False).str.split(".").str[0]

# Check that the label column and the file-name code always agree.
agree = pd.crosstab(df["code"], df["label"])
print(agree, "\\n")

counts = pd.crosstab(df["label"], df["source"], margins=True, margins_name="Total")
# Values copied from Table 1 of the paper (lab, farm).
paper = pd.DataFrame({"lab_paper": [373, 347, 186, 349], "farm_paper": [2103, 2057, 376, 2276]},
                     index=["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"])
table1 = counts.join(paper)
print(table1)
table1.to_csv(f"{OUT}/table1_check.csv")
"""),
("md", """
## 2. Image size

The Kaggle copy of the dataset was resized to 224 × 224 pixels by the uploader. We check a sample of images.
"""),
("code", """
sizes = pd.Series([Image.open(os.path.join(IMG_DIR, f)).size for f in df["file"].sample(500, random_state=0)])
print(sizes.value_counts())
"""),
("md", """
## 3. Exact duplicates

We compute an **MD5 hash** for every file. A hash is a short fingerprint of the file bytes.
Two files with the same MD5 are byte-for-byte identical copies.
"""),
("code", """
def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

df["md5"] = [md5_of(os.path.join(IMG_DIR, f)) for f in df["file"]]

group_size = df.groupby("md5")["file"].transform("size")
dups = df[group_size > 1].sort_values("md5")
n_groups = dups["md5"].nunique()
n_extra = len(dups) - n_groups                       # copies beyond the first one
print(f"duplicate groups: {n_groups}, extra copies: {n_extra} ({100*n_extra/len(df):.1f}% of all images)")

# Which classes / sources do the duplicates come from?
print(dups.groupby(["source", "label"]).size())

# Duplicates that carry more than one label = contradictory labels.
conflict = dups.groupby("md5")["label"].nunique()
conflict_md5 = conflict[conflict > 1].index
print("\\nidentical images with DIFFERENT labels:")
print(dups[dups["md5"].isin(conflict_md5)][["file", "label", "source"]])
"""),
("md", """
## 4. Near-duplicates (same photo, slightly changed)

Exact MD5 misses copies that were re-saved, slightly cropped or re-compressed.
We use a **difference hash (dHash)**:
1. shrink the image to 9 × 8 pixels in grey;
2. for each row, compare each pixel with its right neighbour (brighter = 1, darker = 0);
3. this gives 64 bits (a 64-digit binary code).

Two images whose codes differ in only a few bits (small **Hamming distance**) look almost the same.
We count near-duplicate pairs at several distance limits, so we can see how sensitive the result is.
"""),
("code", """
def dhash_bits(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()          # 64 True/False values

bits = np.stack([dhash_bits(os.path.join(IMG_DIR, f)) for f in df["file"]])   # shape (N, 64)
B = bits.astype(np.uint8)

# Hamming distance between every pair of images, computed with matrix algebra:
# distance = (bits that are 1 in a and 0 in b) + (bits that are 0 in a and 1 in b).
# float32 lets numpy use its fast matrix routines; we round back to whole numbers after.
Bf = B.astype(np.float32)
ham = np.rint(Bf @ (1 - Bf).T + (1 - Bf) @ Bf.T).astype(np.int16)   # (N, N) matrix
np.fill_diagonal(ham, 99)                            # ignore each image vs itself
print("matrix shape:", ham.shape)
"""),
("code", """
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

rows = []
for t in [0, 2, 4, 6, 8, 10]:
    adj = csr_matrix(ham <= t)
    n_comp, comp = connected_components(adj, directed=False)
    comp_size = np.bincount(comp)
    lab = pd.Series(df["label"].values).groupby(comp).nunique()
    rows.append({
        "max_hamming": t,
        "images_with_a_near_copy": int((comp_size[comp] > 1).sum()),
        "groups": int((comp_size > 1).sum()),
        "largest_group": int(comp_size.max()),
        "groups_with_conflicting_labels": int((lab > 1).sum()),
    })
near = pd.DataFrame(rows)
print(near.to_string(index=False))
near.to_csv(f"{OUT}/near_duplicate_sensitivity.csv", index=False)
"""),
("md", """
## 5. Look at the duplicates

Numbers are not enough. We plot some near-duplicate pairs to see whether they are really the same photo.
"""),
("code", """
def show_pairs(t, n=8, fname=None):
    i, j = np.where(np.triu(ham <= t, 1))
    if len(i) == 0:
        print("no pairs at", t); return
    pick = np.random.RandomState(1).choice(len(i), min(n, len(i)), replace=False)
    fig, ax = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.8))
    for k, p in enumerate(pick):
        for r, idx in enumerate([i[p], j[p]]):
            ax[r, k].imshow(Image.open(os.path.join(IMG_DIR, df["file"][idx])))
            ax[r, k].set_title(df["file"][idx], fontsize=7); ax[r, k].axis("off")
    fig.suptitle(f"Pairs with Hamming distance <= {t}")
    plt.tight_layout()
    if fname: plt.savefig(f"{OUT}/{fname}", dpi=120)
    plt.show()

show_pairs(0, fname="pairs_h0.png")
show_pairs(4, fname="pairs_h4.png")
show_pairs(8, fname="pairs_h8.png")
"""),
("md", """
## 6. Example images of each class
"""),
("code", """
fig, ax = plt.subplots(4, 6, figsize=(12, 8.5))
for r, lab in enumerate(sorted(df["label"].unique())):
    files = df[df["label"] == lab]["file"].sample(6, random_state=3).tolist()
    for c, f in enumerate(files):
        ax[r, c].imshow(Image.open(os.path.join(IMG_DIR, f))); ax[r, c].axis("off")
        ax[r, c].set_title(f, fontsize=7)
    ax[r, 0].text(-20, 112, lab, rotation=90, va="center", ha="right", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/class_examples.png", dpi=120); plt.show()
"""),
("code", """
# Save the per-image table (file, label, source, md5, dHash as hex) for the training notebooks.
df["dhash"] = [hex(int("".join(map(str, b)), 2)) for b in B]
df.to_csv(f"{OUT}/image_table.csv", index=False)
print(df.head())
"""),
]

make_notebook(cells, os.path.join(OUT, "01_data_audit.ipynb"))
write_metadata(OUT, "poultry-feces-01-data-audit", "poultry-feces-01-data-audit",
               "01_data_audit.ipynb", gpu=False, datasets=["allandclive/chicken-disease-1"], private=False)
print("built", OUT)
