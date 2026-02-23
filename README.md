# Census Income — Classification & Segmentation

## Setup

Requires Python 3.9+. Install dependencies:

```bash
pip install -r requirements.txt
```

Place `census-bureau.data` and `census-bureau.columns` in the `data/` directory.

## Running

All commands must be run from the project root (`takehomeproject_jpmc/`).

**1. Train classification models:**
```bash
python src/classification/train.py
```

**2. Evaluate classification models** (requires step 1):
```bash
python src/classification/evaluate.py
```

**3. Run segmentation:**
```bash
python src/segmentation/segmentation.py
```

All outputs (models, plots) are saved to `outputs/`.
