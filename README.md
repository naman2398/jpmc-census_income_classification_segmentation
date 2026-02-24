# Census Income — Classification & Segmentation

## Setup

Requires Python 3.9+.

**1. Clone the repository:**
```bash
git clone https://github.com/naman2398/jpmc-census_income_classification_segmentation.git
cd jpmc-census_income_classification_segmentation
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv .venv
```
- macOS / Linux:
  ```bash
  source .venv/bin/activate
  ```
- Windows:
  ```bash
  .venv\Scripts\activate
  ```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

Place `census-bureau-raw.csv` and `census-bureau.columns` in the `data/` directory.

## Running

All commands must be run from the project root (`jpmc-census_income_classification_segmentation`).

**1. Train classification models:**
```bash
python3 src/classification/train.py
```

**2. Evaluate classification models** (requires step 1):
```bash
python3 src/classification/evaluate.py
```

**3. Run segmentation:**
```bash
python3 src/segmentation/segmentation.py
```

All outputs (models, plots) are saved to `outputs/`.
