"""
cc_16feat.py - Predict with ALL 16 features (max accuracy)
Usage: python cc_16feat.py some_image.jpg
"""

import sys
import time
import cv2
import joblib
import numpy as np
from feature_extraction import extract_features_fast, load_image

MODEL_PATH = "model.joblib"  #  16-feature model
_BUNDLE = None

def get_model():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    # 1. Model load (cached)
    t0 = time.perf_counter()
    bundle = get_model()
    model_load_ms = (time.perf_counter() - t0) * 1000

    model, scaler = bundle["model"], bundle["scaler"]
    threshold = bundle.get("threshold", 0.329)

    # 2. Image load
    t0 = time.perf_counter()
    img = load_image(image_path)
    image_load_ms = (time.perf_counter() - t0) * 1000

    if img is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    # 3. Feature extraction (ALL 16 features)
    t0 = time.perf_counter()
    feats = extract_features_fast(img).reshape(1, -1)
    feature_ms = (time.perf_counter() - t0) * 1000

    # 4. Scaling + prediction
    t0 = time.perf_counter()
    feats_scaled = scaler.transform(feats)
    score = float(model.predict_proba(feats_scaled)[0][1])
    predict_ms = (time.perf_counter() - t0) * 1000

    # 5. Total
    total_ms = feature_ms + predict_ms

    # Output
    print(f"{score:.4f}")

    decision = "PHOTO of a SCREEN (1)" if score >= threshold else "REAL PHOTO (0)"
    print(f"[decision] {decision} (threshold={threshold:.3f}, score={score:.4f}), if score is ≥ 0.329, then image is a PHOTO of a SCREEN (1) and if < 0.329 image is a REAL PHOTO (0))", file=sys.stderr)
    
    print(f"\n{'─'*50}", file=sys.stderr)
    print(f"{'─'*50}", file=sys.stderr)
    print(f"  Feature extract: {feature_ms:7.2f} ms  (16 features)", file=sys.stderr)
    print(f"  Prediction:      {predict_ms:7.2f} ms", file=sys.stderr)
    print(f"{'─'*50}", file=sys.stderr)
    print(f"  TOTAL:           {total_ms:7.2f} ms", file=sys.stderr)
    print(f"{'─'*50}", file=sys.stderr)
    print(f"  Features:        {feats.shape[1]} / 16", file=sys.stderr)

if __name__ == "__main__":
    main()