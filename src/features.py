import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label, regionprops, moments_hu

def _mask_pixels(image_bgr, mask):
    if mask is None:
        return image_bgr.reshape(-1, 3)
    m = mask.astype(bool)
    return image_bgr[m]

def _stats_channel(x):
    x = x.astype(np.float32)
    med = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    mad = np.median(np.abs(x - med))
    mean = float(np.mean(x))
    std = float(np.std(x))
    return [mean, std, med, iqr, mad]

def _colorfulness(bgr):
    r = bgr[:, 2]; g = bgr[:, 1]; b = bgr[:, 0]
    rg = r - g; yb = 0.5 * (r + g) - b
    std_rg = np.std(rg); std_yb = np.std(yb)
    mean_rg = np.mean(np.abs(rg)); mean_yb = np.mean(np.abs(yb))
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def _histograms_channels(arr, bins=32):
    h_b = np.histogram(arr[:, 0], bins=bins, range=(0, 255), density=True)[0]
    h_g = np.histogram(arr[:, 1], bins=bins, range=(0, 255), density=True)[0]
    h_r = np.histogram(arr[:, 2], bins=bins, range=(0, 255), density=True)[0]
    return list(h_b) + list(h_g) + list(h_r)

def _lbp_features(gray):
    feats = []
    for radius in [1, 2, 3]:
        p = 8 * radius
        lbp = local_binary_pattern(gray, P=p, R=radius, method='uniform')
        h = np.histogram(lbp.ravel(), bins=np.arange(0, p + 3), density=True)[0]
        feats.extend(list(h))
    return feats

def _glcm_features(gray):
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    g = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']:
        props.append(float(graycoprops(g, prop).mean()))
    p = g[:, :, 0, 0]
    p = p / (p.sum() + 1e-10)
    entropy = float(-np.sum(p * np.log2(p + 1e-10)))
    return props + [entropy]

def _gabor_bank(gray):
    feats = []
    for lam in [4.0, 8.0]:
        for theta in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            k = cv2.getGaborKernel((9, 9), 3.0, theta, lam, 0.5, 0)
            f = cv2.filter2D(gray, cv2.CV_32F, k)
            feats.extend([float(np.mean(f)), float(np.std(f))])
    return feats

def _hog_summary(gray):
    v = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return [float(np.mean(v)), float(np.std(v))]

def _skeleton_metrics(mask):
    if mask is None:
        return [0.0, 0.0]
    m = (mask > 0).astype(np.uint8)
    sk = skeletonize(m.astype(bool)).astype(np.uint8)
    total = float(m.sum())
    length = float(sk.sum())
    ratio = float(length / (total + 1e-5))
    return [length, ratio]

def _shape_features(mask):
    if mask is None:
        return [0.0] * 12
    labeled = sk_label(mask)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return [0.0] * 12
    r = max(regions, key=lambda x: x.area)
    area = float(r.area)
    per = float(r.perimeter if r.perimeter != 0 else 1e-5)
    aspect = float(r.bbox[3] / (r.bbox[2] + 1e-5))
    circ = float(4 * np.pi * area / (per ** 2))
    sol = float(r.solidity)
    ext = float(r.extent)
    hu = list(moments_hu(r.image))
    return [area, per, aspect, circ, sol, ext] + [float(x) for x in hu]

def _convexity_metrics(mask):
    if mask is None:
        return [0.0, 0.0]
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0.0, 0.0]
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    area = float(cv2.contourArea(c))
    hull_area = float(cv2.contourArea(hull))
    return [hull_area, float(area / (hull_area + 1e-5))]

def _dct_energy(gray):
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    d = cv2.dct(g)
    h, w = d.shape
    k = 8
    low = float(np.sum(d[:k, :k] ** 2))
    total = float(np.sum(d ** 2) + 1e-9)
    return [low, float(low / total)]

def extract_features_extended(image_bgr, mask=None):
    pixels = _mask_pixels(image_bgr, mask)
    bgr_stats = _stats_channel(pixels[:, 0]) + _stats_channel(pixels[:, 1]) + _stats_channel(pixels[:, 2])
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_pixels = _mask_pixels(hsv, mask)
    hsv_stats = _stats_channel(hsv_pixels[:, 0]) + _stats_channel(hsv_pixels[:, 1]) + _stats_channel(hsv_pixels[:, 2])
    ratios = [float(np.mean(pixels[:, 1]) / (np.mean(pixels[:, 2]) + 1e-5)), float(np.mean(pixels[:, 1]) / (np.mean(pixels[:, 0]) + 1e-5)), float(np.mean(pixels[:, 2]) / (np.mean(pixels[:, 0]) + 1e-5))]
    color_hist = _histograms_channels(pixels, bins=32)
    colorfulness = [_colorfulness(pixels)]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = _lbp_features(gray)
    glcm = _glcm_features(gray)
    gabor = _gabor_bank(gray)
    hog_sum = _hog_summary(gray)
    skel = _skeleton_metrics(mask)
    shape = _shape_features(mask)
    conv = _convexity_metrics(mask)
    dct = _dct_energy(gray)
    feats = bgr_stats + hsv_stats + ratios + color_hist + colorfulness + lbp + glcm + gabor + hog_sum + skel + shape + conv + dct
    names = []
    names += [f"b_mean", "b_std", "b_med", "b_iqr", "b_mad", "g_mean", "g_std", "g_med", "g_iqr", "g_mad", "r_mean", "r_std", "r_med", "r_iqr", "r_mad"]
    names += [f"h_mean", "h_std", "h_med", "h_iqr", "h_mad", "s_mean", "s_std", "s_med", "s_iqr", "s_mad", "v_mean", "v_std", "v_med", "v_iqr", "v_mad"]
    names += ["ratio_G_R", "ratio_G_B", "ratio_R_B"]
    names += [f"hist_b_{i}" for i in range(32)] + [f"hist_g_{i}" for i in range(32)] + [f"hist_r_{i}" for i in range(32)]
    names += ["colorfulness"]
    names += [f"lbp_{i}" for i in range(len(lbp))]
    names += ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "glcm_dissimilarity", "glcm_ASM", "glcm_entropy"]
    names += [f"gabor_{i}" for i in range(len(gabor))]
    names += ["hog_mean", "hog_std"]
    names += ["skeleton_length", "skeleton_ratio"]
    names += ["area", "perimeter", "aspect_ratio", "circularity", "solidity", "extent", "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]
    names += ["hull_area", "convexity_ratio"]
    names += ["dct_low_energy", "dct_low_ratio"]
    return feats, names
