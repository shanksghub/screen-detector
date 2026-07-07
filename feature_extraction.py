"""
feature_extraction_fast_16feat.py - ALL 16 features (optimized for speed)
"""

import cv2
import numpy as np

TARGET_MAX_DIM = 900

def load_image(path):
    """Load image with faster decode (half resolution)"""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        img = cv2.imread(path)
    return img

def _resize(img, max_dim=900):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return img

def _fft_features(gray):
    """Frequency domain features (moire patterns) - 4 features"""
    small = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(small.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = r.max()
    total = mag.sum() + 1e-6

    low_mask = r < max_r * 0.05
    mid_mask = (r >= max_r * 0.05) & (r < max_r * 0.35)
    high_mask = r >= max_r * 0.35

    low_e = mag[low_mask].sum() / total
    mid_e = mag[mid_mask].sum() / total
    high_e = mag[high_mask].sum() / total

    mid_vals = mag[mid_mask]
    peak_ratio = float(mid_vals.max() / (mid_vals.mean() + 1e-6)) if mid_vals.size else 0.0

    return [float(low_e), float(mid_e), float(high_e), peak_ratio]

def _gradient_orientation_features(gray):
    """Gradient features (axis-aligned edges from screens) - 2 features"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    thresh = mag.mean() + mag.std()
    mask = mag > thresh
    if mask.sum() < 10:
        return [0.0, 0.0]

    angs = (np.arctan2(gy[mask], gx[mask]) * 180 / np.pi) % 180
    hist, _ = np.histogram(angs, bins=36, range=(0, 180))
    hist = hist / (hist.sum() + 1e-6)

    axis_bins = list(range(0, 3)) + list(range(15, 21)) + list(range(33, 36))
    axis_energy = float(hist[axis_bins].sum())
    nz = hist[hist > 0]
    entropy = float(-(nz * np.log(nz)).sum())

    return [axis_energy, entropy]

def _color_features(img_bgr):
    """Color features (glare, saturation, brightness) - 4 features"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_mean = float(s.mean() / 255.0)
    val_mean = float(v.mean() / 255.0)
    highlight_mask = (v > 240) & (s < 30)
    highlight_ratio = float(highlight_mask.sum() / highlight_mask.size)
    uniq_ratio = float(len(np.unique(v)) / 256.0)
    return [sat_mean, val_mean, highlight_ratio, uniq_ratio]

def _sharpness_features(gray):
    """Sharpness (laplacian variance) - 1 feature"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return [float(lap.var())]

def _line_features(gray, edges, lines):
    """Long straight lines (bezels/frames) - 1 feature"""
    if lines is None:
        return [0.0]
    diag = np.sqrt(gray.shape[0] ** 2 + gray.shape[1] ** 2)
    long_lines = 0
    for x1, y1, x2, y2 in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length > diag * 0.25:
            long_lines += 1
    return [float(long_lines)]

def _edge_straightness_features(gray, contours):
    """Edge straightness (too-perfect straight edges) - 2 features"""
    diag = np.hypot(gray.shape[0], gray.shape[1])
    residuals = []
    for c in contours:
        if len(c) < 30:
            continue
        length = cv2.arcLength(c, False)
        if length < diag * 0.15:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        residuals.append(float(d.mean()))
    if not residuals:
        return [0.0, 0.0]
    residuals = np.array(residuals)
    very_straight_count = float((residuals < 1.5).sum())
    min_residual = float(residuals.min())
    return [very_straight_count, min_residual]

def _interior_grid_blur_features(gray, lines):
    """Interior grid/blur (screen grid patterns) - 2 features"""
    if lines is None or len(lines) < 2:
        return [0.0, 0.0]

    # Find parallel lines
    infos = []
    for x1, y1, x2, y2 in lines:
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi % 180
        infos.append((angle, x1, y1, x2, y2))

    best_pair, best_dist = None, None
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a1, a2 = infos[i][0], infos[j][0]
            if abs(a1 - a2) > 5 and abs(abs(a1 - a2) - 180) > 5:
                continue
            x1, y1, x2, y2 = infos[i][1:]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            x3, y3, x4, y4 = infos[j][1:]
            num = abs((y4 - y3) * mx - (x4 - x3) * my + x4 * y3 - y4 * x3)
            den = np.hypot(y4 - y3, x4 - x3) + 1e-6
            dist = num / den
            if dist < 15:
                continue
            if best_dist is None or dist < best_dist:
                best_dist, best_pair = dist, (infos[i], infos[j])

    if best_pair is None:
        return [0.0, 0.0]

    pts = np.array([best_pair[0][1:], best_pair[1][1:]]).reshape(-1, 2)
    x_min, y_min = np.maximum(pts.min(axis=0).astype(int), 0)
    x_max = min(gray.shape[1], int(pts[:, 0].max()))
    y_max = min(gray.shape[0], int(pts[:, 1].max()))
    if x_max - x_min < 10 or y_max - y_min < 10:
        return [0.0, 0.0]

    interior = gray[y_min:y_max, x_min:x_max]

    interior_lap_var = cv2.Laplacian(interior, cv2.CV_64F).var()
    whole_lap_var = cv2.Laplacian(gray, cv2.CV_64F).var() + 1e-6
    blur_ratio = float(interior_lap_var / whole_lap_var)

    grid_periodicity = 0.0
    if interior.shape[1] > 20:
        row_profile = interior.mean(axis=0).astype(np.float32)
        row_profile -= row_profile.mean()
        ac = np.correlate(row_profile, row_profile, mode="full")
        ac = ac[len(ac) // 2:]
        ac = ac / (ac[0] + 1e-6)
        if len(ac) > 10:
            grid_periodicity = float(ac[3:].max())

    return [blur_ratio, grid_periodicity]

def extract_features_fast(img_bgr):
    """Extract ALL 16 features (optimized)"""
    img_bgr = _resize(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute shared expensive operations ONCE
    edges = cv2.Canny(gray, 60, 150)
    min_len = max(20, min(gray.shape) // 3)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=min_len, maxLineGap=10)
    lines = None if raw_lines is None else np.asarray(raw_lines).reshape(-1, 4)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    feats = []
    feats += _fft_features(gray)                       # 4 features
    feats += _gradient_orientation_features(gray)      # 2 features
    feats += _color_features(img_bgr)                  # 4 features
    feats += _sharpness_features(gray)                 # 1 feature
    feats += _line_features(gray, edges, lines)        # 1 feature
    feats += _edge_straightness_features(gray, contours)  # 2 features
    feats += _interior_grid_blur_features(gray, lines)    # 2 features
    # TOTAL: 16 features

    return np.array(feats, dtype=np.float32)