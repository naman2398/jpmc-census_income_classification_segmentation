import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_loader import load_data, prepare_features
from silhouette_analysis import plot_silhouette_analysis

OUT = "outputs/segmentation"


def preprocess_for_clustering(feature_df, numeric_cols, categorical_cols):
    df_encoded = feature_df.copy()
    df_encoded[categorical_cols] = df_encoded[categorical_cols].fillna("Missing")
    df_encoded[numeric_cols] = df_encoded[numeric_cols].fillna(0)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    return X_scaled


def reduce_dimensions(X_scaled, variance_threshold=0.85):
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_pca.shape[1]} components capture {variance_threshold*100:.0f}% variance")
    return X_pca, pca


def find_optimal_k(X_pca, k_range=range(2, 8)):
    sil_scores = plot_silhouette_analysis(X_pca, k_range, OUT)

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(k_range), inertias, "bo-")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Plot")

    axes[1].plot(list(k_range), sil_scores, "ro-")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs k")

    plt.tight_layout()
    plt.savefig(f"{OUT}/cluster_selection.png", dpi=150)
    plt.close()

    return sil_scores


def profile_clusters(X_pca, df, k):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_pca)

    # --- PCA scatter ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="tab10", alpha=0.3, s=5)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters in PCA Space")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cluster_scatter.png", dpi=150)
    plt.close()


    PROFILE_CATS = ["education", "sex", "marital stat", "tax filer stat", "race",
                    "family members under 18"]
    PROFILE_NUMS = {
        "age": "mean",
        "weeks worked in year": "mean",
        "wage per hour": "mean",
        "capital gains": "mean",
        "dividends from stocks": "mean",
        "label": "mean",        # income >50K rate
    }

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Cluster Profiles (k={k})")
    print(f"{'='*60}")
    for c in range(k):
        cdf = df[df["cluster"] == c]
        print(f"\n--- Cluster {c} (n={len(cdf):,}, {len(cdf)/len(df)*100:.1f}%) ---")
        for num_col, agg in PROFILE_NUMS.items():
            val = cdf[num_col].mean()
            label_str = "Income >50K rate" if num_col == "label" else num_col.title()
            fmt = f"{val*100:.1f}%" if num_col == "label" else f"{val:.1f}"
            print(f"  {label_str}: {fmt}")
        for col in PROFILE_CATS:
            top = cdf[col].value_counts().head(1)
            print(f"  Top {col}: {top.index[0]} ({top.iloc[0]/len(cdf)*100:.0f}%)")

    # --- Numeric bar charts ---
    num_profile = df.groupby("cluster").agg(PROFILE_NUMS).round(2)
    n_num = len(num_profile.columns)
    ncols = 3
    nrows = (n_num + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flat
    for ax, col in zip(axes, num_profile.columns):
        vals = num_profile[col]
        bars = ax.bar(vals.index.astype(str), vals.values, color="steelblue")
        ax.set_title("Income >50K Rate" if col == "label" else col.title(), fontsize=11)
        ax.set_xlabel("Cluster")
        ax.tick_params(axis="x", rotation=0)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.1%}" if col == "label" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    for ax in list(axes)[n_num:]:
        ax.set_visible(False)
    plt.suptitle("Cluster Numeric Profiles", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/segment_profiles_numeric.png", dpi=150)
    plt.close()

    # --- Categorical distribution stacked bars (combined 2×3 figure) ---
    ncols_cat = 3
    nrows_cat = (len(PROFILE_CATS) + ncols_cat - 1) // ncols_cat
    fig, axes_cat = plt.subplots(nrows_cat, ncols_cat,
                                  figsize=(21, 5 * nrows_cat))
    axes_cat = axes_cat.flat

    for ax, col in zip(axes_cat, PROFILE_CATS):
        top_vals = df[col].value_counts().head(5).index.tolist()
        def bucket(x, _top=top_vals):
            return x if x in _top else "Other"
        col_bucketed = df[col].map(bucket)
        ct = pd.crosstab(df["cluster"], col_bucketed, normalize="index") * 100
        ct = ct.reindex(columns=sorted(ct.columns))
        ct.plot(kind="bar", stacked=True, colormap="tab20", ax=ax)
        ax.set_title(f"Distribution of {col.title()} by Cluster (%)", fontsize=11)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("% of cluster")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(loc="upper right", fontsize=7, title=col.title())

    for ax in list(axes_cat)[len(PROFILE_CATS):]:
        ax.set_visible(False)

    plt.suptitle("Categorical Distributions by Cluster", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT}/segment_cat_combined.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df


def main():
    os.makedirs(OUT, exist_ok=True)

    df = load_data()
    feature_df, numeric_cols, categorical_cols, label, weight, year = prepare_features(df)

    X_scaled = preprocess_for_clustering(feature_df, numeric_cols, categorical_cols)
    X_pca, pca = reduce_dimensions(X_scaled)

    k_range = range(2, 8)
    sil_scores = find_optimal_k(X_pca, k_range)

    best_k = list(k_range)[np.argmax(sil_scores)]
    if best_k == 2:
        best_k = 5
        print(f"Overriding k=2 (trivial) → using k={best_k} for meaningful segments")
    else:
        print(f"Selected k={best_k} (highest silhouette)")

    df_clustered = profile_clusters(X_pca, df, best_k)

    print(f"\nPlots saved to {OUT}/")


if __name__ == "__main__":
    main()
