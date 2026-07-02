"""
PRODIGY_ML_02 — Customer Segmentation using K-Means Clustering
Prodigy Infotech Machine Learning Internship

Task: Create a K-means clustering algorithm to group customers of a
      retail store based on their purchase history.

Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
         (Download Mall_Customers.csv)

Instructions:
    1. Download Mall_Customers.csv from the Kaggle link above
    2. Place it in the same directory as this script
    3. Run: python task02_customer_segmentation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  PRODIGY_ML_02 — Customer Segmentation (K-Means)")
print("=" * 60)

try:
    df = pd.read_csv("Mall_Customers.csv")
    print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    # Standardise column names
    df.columns = df.columns.str.replace(" ", "_").str.replace(r"[()]", "", regex=True)
except FileNotFoundError:
    print("\n⚠️  Mall_Customers.csv not found. Generating synthetic dataset...")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "CustomerID":    range(1, n + 1),
        "Gender":        np.random.choice(["Male", "Female"], n),
        "Age":           np.random.randint(18, 70, n),
        "Annual_Income_k$": np.random.randint(15, 140, n),
        "Spending_Score_1-100": np.random.randint(1, 100, n),
    })
    # Add realistic cluster structure
    centers = [(20, 80), (80, 80), (50, 50), (20, 20), (80, 20)]
    for i, (inc, sp) in enumerate(centers):
        idx = slice(i * 40, (i + 1) * 40)
        df.loc[df.index[idx], "Annual_Income_k$"] = (
            np.random.normal(inc, 8, 40).clip(15, 140).astype(int))
        df.loc[df.index[idx], "Spending_Score_1-100"] = (
            np.random.normal(sp, 8, 40).clip(1, 100).astype(int))
    print(f"   Synthetic dataset generated: {len(df)} samples")

# ─────────────────────────────────────────────
# 2. DATA EXPLORATION
# ─────────────────────────────────────────────
# Rename columns for convenience
col_map = {}
for c in df.columns:
    if "income" in c.lower():    col_map[c] = "Annual_Income"
    if "spending" in c.lower():  col_map[c] = "Spending_Score"
    if "age" in c.lower():       col_map[c] = "Age"
    if "gender" in c.lower():    col_map[c] = "Gender"
df.rename(columns=col_map, inplace=True)

print("\n📊 Data Overview:")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 3. FEATURE SELECTION & SCALING
# ─────────────────────────────────────────────
feature_cols = [c for c in ["Annual_Income", "Spending_Score", "Age"] if c in df.columns]
X = df[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n🔧 Features for clustering: {feature_cols}")

# ─────────────────────────────────────────────
# 4. OPTIMAL K — ELBOW + SILHOUETTE
# ─────────────────────────────────────────────
K_range = range(2, 11)
inertias, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task-02 | Choosing Optimal K", fontsize=14, fontweight="bold")

axes[0].plot(K_range, inertias, "bo-", lw=2, ms=7)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, sil_scores, "rs-", lw=2, ms=7)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")
axes[1].grid(True, alpha=0.3)

best_k = K_range[np.argmax(sil_scores)]
axes[1].axvline(best_k, color="green", linestyle="--", lw=2,
                label=f"Best K = {best_k}")
axes[1].legend()

plt.tight_layout()
plt.savefig("task02_optimal_k.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n🔍 Optimal K = {best_k}  (Silhouette = {max(sil_scores):.4f})")
print("📈 K-selection plot saved → task02_optimal_k.png")

# ─────────────────────────────────────────────
# 5. TRAIN FINAL K-MEANS MODEL
# ─────────────────────────────────────────────
OPTIMAL_K = best_k
kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df["Cluster"])
print(f"\n✅ Model trained  |  K={OPTIMAL_K}  |  Silhouette={sil:.4f}")

# ─────────────────────────────────────────────
# 6. CLUSTER PROFILES
# ─────────────────────────────────────────────
print("\n📊 Cluster Profiles:")
profile = df.groupby("Cluster")[feature_cols].mean().round(2)
profile["Count"] = df.groupby("Cluster").size()
print(profile.to_string())

# Label clusters meaningfully (heuristic)
LABELS = {
    c: f"Cluster {c}" for c in range(OPTIMAL_K)
}
if "Spending_Score" in df.columns and "Annual_Income" in df.columns:
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    cent_df = pd.DataFrame(centroids_orig, columns=feature_cols)
    for c in range(OPTIMAL_K):
        inc  = cent_df.loc[c, "Annual_Income"]   if "Annual_Income"   in cent_df.columns else 50
        sp   = cent_df.loc[c, "Spending_Score"]  if "Spending_Score"  in cent_df.columns else 50
        if inc > 70 and sp > 60:   LABELS[c] = f"High-Value ({c})"
        elif inc > 70 and sp < 40: LABELS[c] = f"High-Income / Low-Spend ({c})"
        elif inc < 40 and sp > 60: LABELS[c] = f"Low-Income / High-Spend ({c})"
        elif inc < 40 and sp < 40: LABELS[c] = f"Budget Shoppers ({c})"
        else:                       LABELS[c] = f"Average Customers ({c})"

# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────
COLORS = plt.cm.tab10(np.linspace(0, 1, OPTIMAL_K))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Task-02 | K-Means Customer Segmentation", fontsize=14, fontweight="bold")

# --- Plot 1: Income vs Spending ---
for c in range(OPTIMAL_K):
    mask = df["Cluster"] == c
    axes[0].scatter(df.loc[mask, "Annual_Income"], df.loc[mask, "Spending_Score"],
                    color=COLORS[c], label=LABELS[c], alpha=0.7, s=60, edgecolors="white", lw=0.5)

# Plot centroids
centroids_inv = scaler.inverse_transform(kmeans.cluster_centers_)
cent_df_plot = pd.DataFrame(centroids_inv, columns=feature_cols)
if "Annual_Income" in cent_df_plot and "Spending_Score" in cent_df_plot:
    axes[0].scatter(cent_df_plot["Annual_Income"], cent_df_plot["Spending_Score"],
                    marker="*", s=300, color="black", zorder=5, label="Centroids")

axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1–100)")
axes[0].set_title("Income vs Spending Score")
axes[0].legend(fontsize=7, loc="upper left")
axes[0].grid(True, alpha=0.3)

# --- Plot 2: PCA 2D view ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
for c in range(OPTIMAL_K):
    mask = df["Cluster"] == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=COLORS[c], label=LABELS[c], alpha=0.7, s=60, edgecolors="white", lw=0.5)
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[1].set_title("PCA 2D Projection of Clusters")
axes[1].legend(fontsize=7, loc="upper left")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task02_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Cluster visualisation saved → task02_clusters.png")

# ─────────────────────────────────────────────
# 8. CLUSTER SIZE BAR CHART
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
counts = df.groupby("Cluster").size()
bars = ax.bar([LABELS[c] for c in counts.index], counts.values,
               color=[COLORS[c] for c in counts.index], edgecolor="white")
ax.bar_label(bars, padding=3)
ax.set_title("Task-02 | Customer Count per Cluster", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("task02_cluster_sizes.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Cluster sizes plot saved → task02_cluster_sizes.png")

print("\n✅ Task-02 complete!\n")
