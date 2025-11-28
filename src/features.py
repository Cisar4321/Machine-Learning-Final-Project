import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.morphology import skeletonize
from skimage.measure import label as sk_label, regionprops, moments_hu
from skimage.segmentation import slic

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
    feats = [float(x) if np.isfinite(x) else 0.0 for x in feats]
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

def _lab_stats(image_bgr, mask=None):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    pixels = _mask_pixels(lab, mask)
    return _stats_channel(pixels[:, 0]) + _stats_channel(pixels[:, 1]) + _stats_channel(pixels[:, 2])

def _excess_indices(image_bgr, mask=None):
    pixels = _mask_pixels(image_bgr, mask)
    b = pixels[:, 0].astype(np.float32)
    g = pixels[:, 1].astype(np.float32)
    r = pixels[:, 2].astype(np.float32)
    exg = 2 * g - r - b
    exr = 1.4 * r - g
    vari = (g - r) / (g + r - b + 1e-5)
    feats = _stats_channel(exg) + _stats_channel(exr) + _stats_channel(vari)
    names = ["exg_mean", "exg_std", "exg_med", "exg_iqr", "exg_mad", "exr_mean", "exr_std", "exr_med", "exr_iqr", "exr_mad", "vari_mean", "vari_std", "vari_med", "vari_iqr", "vari_mad"]
    return feats, names

def _lesion_metrics(image_bgr, mask=None):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_dark_green = np.array([20, 40, 20])
    upper_dark_green = np.array([80, 255, 255])
    lower_light_green = np.array([30, 10, 40])
    upper_light_green = np.array([90, 255, 255])
    m1 = cv2.inRange(hsv, lower_dark_green, upper_dark_green)
    m2 = cv2.inRange(hsv, lower_light_green, upper_light_green)
    green = cv2.bitwise_or(m1, m2)
    leaf = green
    if mask is not None:
        leaf = cv2.bitwise_and(leaf, mask)
    non_green = cv2.bitwise_and(cv2.bitwise_not(green), (leaf > 0).astype(np.uint8) * 255)
    total = float((leaf > 0).sum())
    lesion_ratio = float((non_green > 0).sum()) / (total + 1e-5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_green, connectivity=8)
    if num_labels <= 1:
        count = 0.0
        mean_area = 0.0
        max_area = 0.0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        count = float(len(areas))
        mean_area = float(np.mean(areas))
        max_area = float(np.max(areas))
    return [lesion_ratio, count, mean_area, max_area], ["lesion_ratio", "lesion_count", "lesion_mean_area", "lesion_max_area"]

def _hough_veins(image_bgr, mask=None):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = cv2.bitwise_and(gray, mask)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=15, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 0.0], ["vein_density", "vein_length_mean", "vein_length_std", "vein_orient_0", "vein_orient_45", "vein_orient_90", "vein_orient_135"]
    lens = []
    angs = []
    for l in lines.reshape(-1, 4):
        dx = float(l[2] - l[0])
        dy = float(l[3] - l[1])
        lens.append(np.sqrt(dx * dx + dy * dy))
        a = np.degrees(np.arctan2(dy, dx))
        a = np.abs(((a + 180) % 180) - 90)
        angs.append(a)
    lens = np.array(lens, dtype=np.float32)
    angs = np.array(angs, dtype=np.float32)
    area = float((mask > 0).sum()) if mask is not None else float(image_bgr.shape[0] * image_bgr.shape[1])
    density = float(len(lens)) / (area + 1e-5)
    orient_bins = [0, 22.5, 67.5, 112.5, 180.0]
    hist = np.histogram(angs, bins=orient_bins)[0].astype(np.float32)
    hist = hist / (hist.sum() + 1e-5)
    return [density, float(np.mean(lens)), float(np.std(lens))] + [float(x) for x in hist], ["vein_density", "vein_length_mean", "vein_length_std", "vein_orient_0", "vein_orient_45", "vein_orient_90", "vein_orient_135"]

def _fractal_dimension(mask):
    if mask is None:
        return [0.0], ["fractal_dim"]
    m = (mask > 0).astype(np.uint8)
    sizes = np.array([2, 4, 8, 16, 32])
    h, w = m.shape
    ns = []
    for s in sizes:
        hh = int(np.ceil(h / s) * s)
        ww = int(np.ceil(w / s) * s)
        padded = np.zeros((hh, ww), dtype=np.uint8)
        padded[:h, :w] = m
        resh = padded.reshape(hh // s, s, ww // s, s).max(axis=(1, 3))
        ns.append(float(np.count_nonzero(resh)))
    ns = np.array(ns, dtype=np.float32)
    x = np.log(1.0 / sizes.astype(np.float32))
    y = np.log(ns + 1e-5)
    coeffs = np.polyfit(x, y, 1)
    return [float(-coeffs[0])], ["fractal_dim"]

def _curvature_features(mask):
    if mask is None:
        return [0.0, 0.0, 0.0], ["curv_mean", "curv_std", "curv_p95"]
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [0.0, 0.0, 0.0], ["curv_mean", "curv_std", "curv_p95"]
    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return [0.0, 0.0, 0.0], ["curv_mean", "curv_std", "curv_p95"]
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curv = np.abs(ddx * dy - ddy * dx) / (dx * dx + dy * dy + 1e-5) ** 1.5
    return [float(np.mean(curv)), float(np.std(curv)), float(np.percentile(curv, 95))], ["curv_mean", "curv_std", "curv_p95"]

def _slic_patchiness(image_bgr, mask=None):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    seg = slic(rgb, n_segments=100, compactness=10.0, start_label=1)
    g = image_bgr[:, :, 1].astype(np.float32)
    variances = []
    for label in np.unique(seg):
        sp = seg == label
        if mask is not None:
            sp = sp & (mask > 0)
        vals = g[sp]
        if vals.size == 0:
            continue
        variances.append(float(np.var(vals)))
    if len(variances) == 0:
        return [0.0, 0.0], ["slic_var_mean", "slic_var_std"]
    v = np.array(variances, dtype=np.float32)
    return [float(np.mean(v)), float(np.std(v))], ["slic_var_mean", "slic_var_std"]

def _fft_orientation(gray):
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    y = np.arange(-h // 2, h // 2)
    x = np.arange(-w // 2, w // 2)
    X, Y = np.meshgrid(x, y)
    ang = np.degrees(np.arctan2(Y, X))
    rad = np.sqrt(X * X + Y * Y)
    mask = rad > 5
    bins = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    hist = []
    for i in range(len(bins) - 1):
        m = (ang >= bins[i]) & (ang < bins[i + 1]) & mask
        hist.append(float(mag[m].sum()))
    hist = np.array(hist, dtype=np.float32)
    hist = hist / (hist.sum() + 1e-5)
    return [float(x) for x in hist], [f"fft_orient_{i}" for i in range(len(hist))]

def extract_features_advanced(image_bgr, mask=None):
    lab = _lab_stats(image_bgr, mask)
    exg, exg_names = _excess_indices(image_bgr, mask)
    lesion, lesion_names = _lesion_metrics(image_bgr, mask)
    vein, vein_names = _hough_veins(image_bgr, mask)
    fractal, fractal_names = _fractal_dimension(mask)
    curv, curv_names = _curvature_features(mask)
    slicv, slic_names = _slic_patchiness(image_bgr, mask)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ffto, ffto_names = _fft_orientation(gray)
    feats = lab + exg + lesion + vein + fractal + curv + slicv + ffto
    feats = [float(x) if np.isfinite(x) else 0.0 for x in feats]
    names = ["lab_l_mean", "lab_l_std", "lab_l_med", "lab_l_iqr", "lab_l_mad", "lab_a_mean", "lab_a_std", "lab_a_med", "lab_a_iqr", "lab_a_mad", "lab_b_mean", "lab_b_std", "lab_b_med", "lab_b_iqr", "lab_b_mad"] + exg_names + lesion_names + vein_names + fractal_names + curv_names + slic_names + ffto_names
    return feats, names
