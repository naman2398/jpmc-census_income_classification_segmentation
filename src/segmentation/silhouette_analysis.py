"""Per-k silhouette analysis plots (blade + scatter) for KMeans clustering."""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette_analysis(X, k_range, out_dir, sample_size=30000, random_state=42):
    """
    Reference : https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    idx = rng.choice(n, min(sample_size, n), replace=False)
    X_sub = X[idx]

    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=random_state, n_init=10)
        labels = km.fit_predict(X_sub)

        sil_avg = silhouette_score(X_sub, labels)
        sil_scores.append(sil_avg)
        sample_sil = silhouette_samples(X_sub, labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Left panel: silhouette blade plot ---
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X_sub) + (k + 1) * 10])

        y_lower = 10
        for i in range(k):
            vals = np.sort(sample_sil[labels == i])
            y_upper = y_lower + len(vals)
            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * len(vals), str(i))
            y_lower = y_upper + 10

        ax1.axvline(x=sil_avg, color="red", linestyle="--")
        ax1.set_title("Silhouette Plot")
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster Label")
        ax1.set_yticks([])

        # --- Right panel: PCA scatter ---
        colors = cm.nipy_spectral(labels.astype(float) / k)
        ax2.scatter(X_sub[:, 0], X_sub[:, 1], marker=".", s=20, lw=0,
                    alpha=0.7, c=colors, edgecolor="k")

        centers = km.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white",
                    alpha=1, s=200, edgecolor="k")
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")

        ax2.set_title("Clustered Data (PCA)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")

        plt.suptitle(f"Silhouette Analysis — k={k}  (avg={sil_avg:.3f})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/silhouette_k{k}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"k={k}: silhouette={sil_avg:.3f}")

    return sil_scores
