"""Small helper: turn a list of ("md", text) / ("code", text) cells into a Jupyter notebook.

We write the notebooks from Python so the code lives in plain .py files that are easy to read and edit.
"""
import json
import textwrap


def make_notebook(cells, path):
    nb_cells = []
    for kind, text in cells:
        text = textwrap.dedent(text).strip("\n")
        cell = {
            "cell_type": "markdown" if kind == "md" else "code",
            "metadata": {},
            "source": text,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def write_metadata(folder, slug, title, code_file, gpu, datasets, private=True):
    meta = {
        "id": f"mdraihanpatwary02/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    with open(f"{folder}/kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
