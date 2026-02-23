import pandas as pd
import numpy as np
import re


DATA_PATH = "data/census-bureau.data"
COLUMNS_PATH = "data/census-bureau.columns"

NIU_PATTERN = re.compile(r"not in universe", re.IGNORECASE)

# Data-backed drops (EDA feature audit — simple criteria):
#   - Structural: weight (sampling weight), year (temporal split variable)
#   - Redundancy: detailed recodes → major codes (52→24, 47→15 categories)
#   - Near-constant (top value >95%): fill inc questionnaire, reason for unemployment
#   - "Not in universe" >90%: enroll in edu, labor union, region/state prev res
#   - High missingness (>40% ?): migration codes, migration prev res in sunbelt
#   - Multiple weak signals: live in this house (50.7% "Not in universe")
COLUMNS_TO_DROP = [
    "weight",
    "year",
    "detailed industry recode",
    "detailed occupation recode",
    "fill inc questionnaire for veteran's admin",
    "reason for unemployment",
    "enroll in edu inst last wk",
    "member of a labor union",
    "region of previous residence",
    "state of previous residence",
    "live in this house 1 year ago",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "migration prev res in sunbelt",
]


def load_data():
    columns = open(COLUMNS_PATH).read().strip().split("\n")
    df = pd.read_csv(DATA_PATH, header=None, names=columns)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    df["label"] = (df["label"] == "50000+.").astype(int)
    return df


def prepare_features(df):
    weight = df["weight"].copy()
    year = df["year"].copy()
    label = df["label"].copy()

    feature_df = df.drop(columns=["label"] + COLUMNS_TO_DROP, errors="ignore")

    feature_df["has_capital_gains"] = (feature_df["capital gains"] > 0).astype(int)
    feature_df["has_capital_losses"] = (feature_df["capital losses"] > 0).astype(int)
    feature_df["has_dividends"] = (feature_df["dividends from stocks"] > 0).astype(int)

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    feature_df[categorical_cols] = feature_df[categorical_cols].fillna("Missing")

    return feature_df, numeric_cols, categorical_cols, label, weight, year
