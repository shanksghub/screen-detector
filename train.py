"""
train_16feat.py - Train with ALL 16 features (optimized extractor)
Usage: python train_16feat.py --real_dir data/real --screen_dir data/screen --out model_16feat.joblib
"""

import argparse
import glob
import os
import time
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Import the OPTIMIZED 16-feature extractor
from feature_extraction import extract_features_fast, load_image

FEATURE_NAMES = [
    "fft_low", "fft_mid", "fft_high", "fft_peak",
    "grad_axis", "grad_entropy",
    "sat_mean", "val_mean", "highlight_ratio", "uniq_intensity",
    "laplacian_var",
    "long_line_count",
    "very_straight_edge_count", "min_edge_residual",
    "interior_blur_ratio", "interior_grid_periodicity"
]

def load_folder(path, label):
    X, y = [], []
    files = glob.glob(os.path.join(path, "*"))
    total = len(files)
    
    print(f"  Loading {total} images from {path}...")
    for idx, fp in enumerate(files, 1):
        img = load_image(fp)
        if img is None:
            continue
        try:
            feats = extract_features_fast(img)
            X.append(feats)
            y.append(label)
        except Exception as e:
            print(f"    Skip {os.path.basename(fp)}: {e}")
        
        # Progress indicator
        if idx % 10 == 0 or idx == total:
            print(f"    Processed {idx}/{total}", end='\r', flush=True)
    
    print(f"    Loaded {len(X)} images from {path}")
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="data/real")
    ap.add_argument("--screen_dir", default="data/screen")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    print("\n" + "="*60)
    print("TRAINING WITH ALL 16 FEATURES (OPTIMIZED)")
    print("="*60)

    # Load data
    print("\nLoading data...")
    Xr, yr = load_folder(args.real_dir, 0)
    Xs, ys = load_folder(args.screen_dir, 1)
    X = np.array(Xr + Xs)
    y = np.array(yr + ys)
    print(f"\n  ✅ Loaded {len(Xr)} real, {len(Xs)} screen images")
    print(f"  ✅ Feature vector size: {X.shape[1]} features")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Best hyperparameters from tuning
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced"
    )

    # Cross-validation
    n_splits = min(5, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print("\n" + "="*60)
    
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    # Train on full dataset
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    clf.fit(X_scaled, y)

    # Training accuracy (for reference)
    train_pred = clf.predict(X_scaled)
    train_acc = np.mean(train_pred == y)
    print(f"\n  📊 Training accuracy: {train_acc:.4f} ")

    # Save model
    joblib.dump({
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.329
    }, args.out)
    
    print("\n" + "="*60)
    print(f"✅ Saved model to {args.out}")
    print("="*60)

if __name__ == "__main__":
    main()