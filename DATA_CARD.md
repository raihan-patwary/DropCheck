# Data card

## Source

| | Field set | PCR set |
| --- | --- | --- |
| Zenodo record | [10.5281/zenodo.4628934](https://doi.org/10.5281/zenodo.4628934) (version 2) | [10.5281/zenodo.5801834](https://doi.org/10.5281/zenodo.5801834) (version 3) |
| Licence | CC BY 4.0 | CC BY 4.0 |
| Images | 6,812 | 1,255 |
| File names | `cocci.N.jpg`, `healthy.N.jpg`, `ncd.N.jpg`, `salmo.N.jpg` | the same with a `pcr` prefix |
| Paper | Machuve et al. (2022), *Frontiers in Artificial Intelligence* 5: 733345, [10.3389/frai.2022.733345](https://doi.org/10.3389/frai.2022.733345) | same |

Photos were taken with phones (Open Data Kit app) in the Arusha and Kilimanjaro regions of Tanzania between 2020 and 2021.

This project uses the Kaggle copy [`allandclive/chicken-disease-1`](https://www.kaggle.com/datasets/allandclive/chicken-disease-1).
It holds both sets (8,067 images) resized to 224 x 224 pixels, with a `train_data.csv` of file names and labels.
The counts match Table 1 of the paper exactly, class by class (`results/01_data_audit/table1_check.csv`).

## How the labels were made

**PCR set.** Each photo was linked by barcode to a faecal sample. The sample was tested by PCR at the NM-AIST molecular
laboratory for Coccidiosis, Newcastle disease and Salmonella. The paper and the Zenodo record agree on this.

**Field set. The paper and the Zenodo record describe it differently.**

* The paper (Section 3.1) says the 6,812 images were labelled "with the assistance from the animal health professionals
  i.e., veterinarian and field officer", and gives colour and shape rules (Coccidiosis dark brown and flat, Newcastle
  green, Salmonella white and loose, healthy grey or white and solid).
* The Zenodo record says the healthy and Coccidiosis photos "were taken from poultry farms", while chickens "were
  inoculated for Salmonella disease and fecal images taken from the diseased chickens ... after one week", and
  "inoculated for Newcastle disease and fecal images ... were taken within three days".

If the Zenodo description is right, two of the four field classes were photographed on infected study birds and two
on ordinary farms. The class would then be tied to where the photo was taken. We do not know which description is
complete, so this repository calls these labels **field labels**, not "eye labels" and not "lab-confirmed".

Later papers describe the 6,812 field images as "PCR-verified" (for example Dhungana et al. 2025,
*AgriEngineering* 7: 278). Neither the paper nor the Zenodo record supports that.

## Problems found (notebook 01)

| Check | Result | How certain |
| --- | --- | --- |
| Byte-identical copies (same MD5) | 222 groups, 280 extra copies (3.5% of images), almost all field Coccidiosis and Salmonella | certain |
| Identical files with different labels | 3 pairs: pcrcocci.228 = pcrhealthy.83, pcrhealthy.153 = pcrsalmo.332, pcrhealthy.306 = pcrsalmo.202 | certain |
| Re-saved copies with different labels | at least 3 more: pcrcocci.118 = pcrhealthy.108, pcrcocci.205 = pcrhealthy.171, pcrcocci.215 = pcrhealthy.183 (dHash distance 0, checked by eye) | certain for these 3 |
| Near-copies (dHash distance 4 bits or less) | 749 images in 307 groups | includes some false matches: different droppings on plain grey floors |

`results/01_data_audit/near_duplicate_sensitivity.csv` shows how these counts change with the distance limit.

## Clean set used for the leak-free and second-run tests

1. Group images linked by the same MD5 or a dHash distance of 4 bits or less (connected components).
2. Remove every group that holds more than one label.
3. Keep one copy of each identical file.
4. Split with `StratifiedGroupKFold` (5 folds, seed 42), so a group never sits on both sides of a split.

Grouping on dHash also joins some photos that are not true copies. That can only make a test harder, never easier.

## What is not in this repository

No images and no trained models. All numbers and figures are rebuilt from the Kaggle data by the notebooks in
`kernels/`. The few example photos in `figures/` are from the dataset above (CC BY 4.0, Machuve et al. 2021).
