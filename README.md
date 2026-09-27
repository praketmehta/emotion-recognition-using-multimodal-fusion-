# Emotion Recognition Using Multimodal Fusion

Multimodal speech emotion recognition (SER) using early, late, deep, and gated fusion of audio and text/visual modalities, evaluated on RAVDESS and MELD.

## Datasets

**RAVDESS** — 24 actors, 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised), 1,440 speech clips. Speaker-independent split (actors 1–18 train, 19–20 val, 21–24 test) to avoid speaker leakage.

**MELD** — Conversational, spontaneous, imbalanced emotion dataset from *Friends*, using audio and text modalities with the official train/dev/test split.

## Repository structure

configs/          RAVDESS label and split CSVs
src/
  data/           Dataset parsing and split logic
  features/       MFCC and WavLM (layer-weighted) feature extraction
  models/         Fusion architectures, training and evaluation loops
cache/            Extracted feature embeddings (gitignored, regenerate locally)
results/          Metrics logs and model checkpoints (checkpoints gitignored)
main.ipynb        Exploratory pipeline notebook

## Setup

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Requires ffmpeg on PATH for MELD audio extraction.

## Reproducing the RAVDESS baseline

1. Download RAVDESS speech audio from Zenodo (record 1188976, Audio_Speech_Actors_01-24.zip) and extract to ravdessDATASET/.
2. Run the parsing and split cell in main.ipynb to generate configs/ravdess_split.csv.
3. Run WavLM layer-weighted feature extraction to populate cache/ravdess_wavlm_layers/.
4. Run the training cell to train the layer-weighted classifier; best checkpoint saves to results/.

## Metrics

Models are evaluated with weighted accuracy (WA), unweighted accuracy (UA, i.e. balanced accuracy / mean per-class recall), and macro-F1.

## Status

Work in progress — mid-semester milestone complete for RAVDESS; MELD pipeline and fusion architectures (early/late/deep/gated) in progress.
