"""
PRODIGY_ML_03 — Image Classification: Cats vs Dogs using SVM
Prodigy Infotech Machine Learning Internship

Task: Implement a support vector machine (SVM) to classify images of
      cats and dogs from the Kaggle dataset.

Dataset: https://www.kaggle.com/c/dogs-vs-cats/data
         (Download and extract train.zip — contains cat.0.jpg, dog.0.jpg, …)

Instructions:
    1. Download and extract the dataset from Kaggle
    2. Set DATASET_DIR below to the folder containing the images
    3. Run: python task03_cats_dogs_svm.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ──────────────────────────────
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, accuracy_score)
from sklearn.pipeline import Pipeline

# ── Try OpenCV / PIL for image loading ────────
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    try:
        from PIL import Image
    except ImportError:
        print("Install opencv-python or Pillow:  pip install opencv-python pillow")
        sys.exit(1)

# ──────────────────────────────────────────────
# CONFIGURATION — edit this path
# ──────────────────────────────────────────────
DATASET_DIR = "train"          # Folder with cat.0.jpg, dog.0.jpg …
IMG_SIZE    = (64, 64)         # Resize all images to this
MAX_IMAGES  = 2000             # Per class (keep small for SVM speed)
N_COMPONENTS_PCA = 150         # PCA components before SVM

# ─────────────────────────────────────────────
print("=" * 60)
print("  PRODIGY_ML_03 — Cats vs Dogs SVM Classifier")
print("=" * 60)


def load_image(path: str) -> np.ndarray:
    """Load and resize image to IMG_SIZE, return as float32 array."""
    if USE_CV2:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
    else:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        img = np.array(img)
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# 1. LOAD IMAGES
# ─────────────────────────────────────────────
images, labels = [], []

dataset_path = Path(DATASET_DIR)
if not dataset_path.exists():
    print(f"\n⚠️  '{DATASET_DIR}' not found. Creating synthetic noise dataset for demo…")
    print("   (Replace DATASET_DIR with the real Kaggle dataset path)\n")
    np.random.seed(42)
    n_each = 300
    # Simulate cat images (slightly blueish noise)
    for _ in range(n_each):
        img = np.random.rand(*IMG_SIZE, 3).astype(np.float32)
        img[:, :, 2] = np.clip(img[:, :, 2] + 0.15, 0, 1)   # blue bias → cats
        images.append(img)
        labels.append(0)  # cat
    # Simulate dog images (slightly reddish noise)
    for _ in range(n_each):
        img = np.random.rand(*IMG_SIZE, 3).astype(np.float32)
        img[:, :, 0] = np.clip(img[:, :, 0] + 0.15, 0, 1)   # red bias → dogs
        images.append(img)
        labels.append(1)  # dog
    print(f"   Synthetic dataset: {len(images)} images  (cats={n_each}, dogs={n_each})")
else:
    print(f"\n📂 Loading images from '{DATASET_DIR}' …")
    counts = {"cat": 0, "dog": 0}
    for filepath in sorted(dataset_path.glob("*.jpg")):
        name = filepath.name.lower()
        if name.startswith("cat") and counts["cat"] < MAX_IMAGES:
            img = load_image(str(filepath))
            if img is not None:
                images.append(img)
                labels.append(0)
                counts["cat"] += 1
        elif name.startswith("dog") and counts["dog"] < MAX_IMAGES:
            img = load_image(str(filepath))
            if img is not None:
                images.append(img)
                labels.append(1)
                counts["dog"] += 1
        if counts["cat"] >= MAX_IMAGES and counts["dog"] >= MAX_IMAGES:
            break
    if len(images) == 0:
        print("❌ No images found! Check DATASET_DIR and naming (cat.0.jpg / dog.0.jpg).")
        sys.exit(1)
    print(f"   Loaded {counts['cat']} cat images + {counts['dog']} dog images")

X_raw = np.array(images)   # shape: (N, H, W, 3)
y     = np.array(labels)
CLASS_NAMES = ["Cat", "Dog"]

# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION — HOG-like flattening + PCA
# ─────────────────────────────────────────────
# Flatten pixels (grayscale) → PCA for dimensionality reduction
X_gray = 0.299 * X_raw[:, :, :, 0] + \
         0.587 * X_raw[:, :, :, 1] + \
         0.114 * X_raw[:, :, :, 2]         # to grayscale
X_flat = X_gray.reshape(len(X_raw), -1)    # (N, H*W)

print(f"\n🔧 Raw feature shape : {X_flat.shape}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 4. PIPELINE: Scaler → PCA → SVM
# ─────────────────────────────────────────────
n_comp = min(N_COMPONENTS_PCA, X_train.shape[0] - 1, X_train.shape[1])
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=n_comp, whiten=True, random_state=42)),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
])

print(f"\n🚀 Training SVM pipeline (PCA={n_comp} components, RBF kernel) …")
pipe.fit(X_train, y_train)

y_pred       = pipe.predict(X_test)
y_pred_prob  = pipe.predict_proba(X_test)[:, 1]
accuracy     = accuracy_score(y_test, y_pred)

print(f"\n{'─'*40}")
print("  MODEL PERFORMANCE")
print(f"{'─'*40}")
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task-03 | Cats vs Dogs — SVM Results", fontsize=14, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix  (Accuracy={accuracy*100:.1f}%)")

# PCA variance explained
pca_model = pipe.named_steps["pca"]
cumvar = np.cumsum(pca_model.explained_variance_ratio_)
axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, lw=2, color="#4C72B0")
axes[1].axhline(90, color="red", linestyle="--", lw=1.5, label="90% variance")
axes[1].set_xlabel("Number of PCA Components")
axes[1].set_ylabel("Cumulative Variance Explained (%)")
axes[1].set_title("PCA: Variance Explained")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task03_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Results plot saved → task03_results.png")

# ─────────────────────────────────────────────
# 6. SAMPLE PREDICTIONS GRID
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
fig.suptitle("Task-03 | Sample Predictions", fontsize=13, fontweight="bold")
idx_cat = np.where(y_test == 0)[0][:8]
idx_dog = np.where(y_test == 1)[0][:8]

for row, indices in enumerate([idx_cat, idx_dog]):
    for col, idx in enumerate(indices):
        ax = axes[row][col]
        img_gray = X_test[idx].reshape(IMG_SIZE)
        ax.imshow(img_gray, cmap="gray")
        pred  = CLASS_NAMES[y_pred[idx]]
        true  = CLASS_NAMES[y_test[idx]]
        color = "green" if pred == true else "red"
        ax.set_title(f"Pred:{pred}\nTrue:{true}", fontsize=6, color=color)
        ax.axis("off")

plt.tight_layout()
plt.savefig("task03_sample_preds.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Sample predictions saved → task03_sample_preds.png")

print("\n✅ Task-03 complete!\n")
