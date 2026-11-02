# NOVA PIXEL

An experimental text-to-image research project built from randomly initialized weights.

**Current status:** v0.1 scaffold. This repository contains a small educational diffusion baseline; it is not a capable image generator yet.

## Principles
- No pretrained image-generation weights.
- Model architecture and training code are ours to modify.
- Dataset choice and training are explicit; no dataset is silently downloaded.
- Content settings are implemented in our own application code.
- Safety rules for sexual content involving minors, non-consensual sexual imagery, and exploitation are not user-disableable.

## Quick start (Google Colab or Linux)
```bash
pip install -r requirements.txt
```

Prepare a dataset folder:
```text
data/
  images/
    0001.jpg
    0002.jpg
  captions.csv
```

`captions.csv` must contain two columns: `filename,caption`.

Train:
```bash
python train.py --data_dir data --epochs 10 --batch_size 16
```

Generate:
```bash
python sample.py --checkpoint checkpoints/latest.pt --prompt "a red bicycle beside a tree"
```

The first model uses a tiny byte-level text encoder and a compact conditional denoiser. It is intended for learning and debugging, not high-quality results. Meaningful text-to-image quality requires a large, well-captioned dataset, substantial compute, and extensive training.

## Project layout
- `config.py` — model and diffusion settings
- `model.py` — randomly initialized text-conditioned denoising network
- `dataset.py` — local image/caption dataset loader
- `diffusion.py` — forward noising and reverse sampling
- `rules.py` — editable prompt-category settings
- `train.py` — training loop
- `sample.py` — prompt-to-image inference
