import os
import time
import warnings
import cv2
import numpy as np
import pandas as pd

import cupy as cp
import cupyx.scipy.ndimage as cnd

warnings.filterwarnings("ignore")


# ============================================================
# Helpers
# ============================================================

EPS = 1e-10


def gpu_available():
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def to_gpu_bgr(img_cpu: np.ndarray) -> cp.ndarray:
    if img_cpu is None:
        raise ValueError("Failed to read image.")
    if img_cpu.ndim != 3 or img_cpu.shape[2] < 3:
        raise ValueError("Input must be a BGR image with at least 3 channels.")
    if img_cpu.shape[2] == 4:
        img_cpu = img_cpu[:, :, :3]
    return cp.asarray(img_cpu)


def to_cpu(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def resize_bgr_gpu(img_gpu: cp.ndarray, size):
    # OpenCV resize stays CPU for portability
    img_cpu = cp.asnumpy(img_gpu)
    out = cv2.resize(img_cpu, size)
    return cp.asarray(out)


def bgr_to_gray_gpu(img_gpu: cp.ndarray) -> cp.ndarray:
    # OpenCV grayscale equivalent
    b = img_gpu[:, :, 0].astype(cp.float32)
    g = img_gpu[:, :, 1].astype(cp.float32)
    r = img_gpu[:, :, 2].astype(cp.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return cp.clip(gray, 0, 255).astype(cp.uint8)


def bgr_to_hsv_gpu(img_gpu: cp.ndarray) -> cp.ndarray:
    """
    Match OpenCV HSV conventions approximately:
      H in [0,179], S in [0,255], V in [0,255]
    """
    img = img_gpu.astype(cp.float32) / 255.0
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    cmax = cp.maximum(cp.maximum(r, g), b)
    cmin = cp.minimum(cp.minimum(r, g), b)
    delta = cmax - cmin

    h = cp.zeros_like(cmax)

    mask = delta > 0
    rmax = (cmax == r) & mask
    gmax = (cmax == g) & mask
    bmax = (cmax == b) & mask

    h = cp.where(rmax, ((g - b) / (delta + EPS)) % 6, h)
    h = cp.where(gmax, ((b - r) / (delta + EPS)) + 2, h)
    h = cp.where(bmax, ((r - g) / (delta + EPS)) + 4, h)
    h = h * 30.0  # 0..180 approximate OpenCV scale

    s = cp.where(cmax > 0, delta / (cmax + EPS), 0.0) * 255.0
    v = cmax * 255.0

    hsv = cp.stack([
        cp.clip(h, 0, 179),
        cp.clip(s, 0, 255),
        cp.clip(v, 0, 255)
    ], axis=-1)

    return hsv.astype(cp.uint8)


def bgr_to_ycrcb_gpu(img_gpu: cp.ndarray) -> cp.ndarray:
    """
    Approximate OpenCV BGR -> YCrCb
    """
    b = img_gpu[:, :, 0].astype(cp.float32)
    g = img_gpu[:, :, 1].astype(cp.float32)
    r = img_gpu[:, :, 2].astype(cp.float32)

    y = 0.114 * b + 0.587 * g + 0.299 * r
    cb = 128 + (b - y) * 0.564
    cr = 128 + (r - y) * 0.713

    ycrcb = cp.stack([
        cp.clip(y, 0, 255),
        cp.clip(cr, 0, 255),
        cp.clip(cb, 0, 255),
    ], axis=-1)

    return ycrcb.astype(cp.float32)


def sobel_xy_gpu(gray_gpu: cp.ndarray):
    gray = gray_gpu.astype(cp.float32)
    gx = cnd.sobel(gray, axis=1, mode="reflect")
    gy = cnd.sobel(gray, axis=0, mode="reflect")
    return gx, gy


def cart_to_polar_gpu(gx: cp.ndarray, gy: cp.ndarray, angle_in_degrees=True):
    mag = cp.sqrt(gx * gx + gy * gy)
    ang = cp.arctan2(gy, gx)
    if angle_in_degrees:
        ang = cp.degrees(ang)
        ang = cp.mod(ang + 360.0, 360.0)
    return mag, ang


def dct_matrix(n=8, dtype=cp.float32):
    C = cp.zeros((n, n), dtype=dtype)
    factor = cp.pi / (2 * n)
    for k in range(n):
        alpha = cp.sqrt(1.0 / n) if k == 0 else cp.sqrt(2.0 / n)
        for i in range(n):
            C[k, i] = alpha * cp.cos((2 * i + 1) * k * factor)
    return C


_DCT8 = dct_matrix(8)


def dct2_8x8_gpu(x: cp.ndarray) -> cp.ndarray:
    # x shape: (8,8)
    return _DCT8 @ x @ _DCT8.T


def bincount2d_pairs_gpu(p: cp.ndarray, q: cp.ndarray, levels: int) -> cp.ndarray:
    idx = p.astype(cp.int32) * levels + q.astype(cp.int32)
    bc = cp.bincount(idx.ravel(), minlength=levels * levels)
    return bc.reshape(levels, levels).astype(cp.float64)


# ============================================================
# GPU feature functions
# ============================================================

def compute_glcm(img, distances=[1], angles=[0], levels=32):
    img_gpu = cp.asarray(img).astype(cp.uint8)

    img_q = (img_gpu.astype(cp.float32) * levels / 256.0).astype(cp.int32)
    img_q = cp.clip(img_q, 0, levels - 1)

    h, w = img_q.shape
    glcm = cp.zeros((levels, levels), dtype=cp.float64)

    for d in distances:
        for theta in angles:
            dx = int(round(d * np.cos(theta)))
            dy = int(round(d * np.sin(theta)))

            y0 = max(0, -dy)
            y1 = min(h, h - dy)
            x0 = max(0, -dx)
            x1 = min(w, w - dx)

            if y1 <= y0 or x1 <= x0:
                continue

            src = img_q[y0:y1, x0:x1]
            dst = img_q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            glcm += bincount2d_pairs_gpu(src, dst, levels)

    s = glcm.sum()
    if s > 0:
        glcm /= s

    return glcm


def quantize_image(img, bins_per_channel=8):
    img_gpu = cp.asarray(img)
    bins = cp.linspace(0, 256, bins_per_channel + 1, dtype=cp.int32)
    quantized = cp.digitize(img_gpu, bins) - 1
    return quantized


def color_correlogram(img, bins=32, distances=32):
    st = time.time()

    img_gpu = cp.asarray(img)
    img_small = resize_bgr_gpu(img_gpu, (64, 64))
    hsv = bgr_to_hsv_gpu(img_small)

    h_bins, s_bins, v_bins = 4, 4, 2
    idx = (
        (hsv[..., 0].astype(cp.int32) // (180 // h_bins)) * (s_bins * v_bins)
        + (hsv[..., 1].astype(cp.int32) // (256 // s_bins)) * v_bins
        + (hsv[..., 2].astype(cp.int32) // (256 // v_bins))
    )
    idx = cp.clip(idx, 0, bins - 1)

    h, w = idx.shape
    features = []

    for d in range(1, distances + 1):
        correlogram = cp.zeros(bins, dtype=cp.float32)
        count = cp.zeros(bins, dtype=cp.float32)

        # right neighbor
        if d < w:
            a = idx[:, :-d]
            b = idx[:, d:]
            matches = (a == b)
            if matches.size > 0:
                correlogram += cp.bincount(a[matches].ravel(), minlength=bins).astype(cp.float32)
                count += cp.bincount(a.ravel(), minlength=bins).astype(cp.float32)

        # down neighbor
        if d < h:
            a = idx[:-d, :]
            b = idx[d:, :]
            matches = (a == b)
            if matches.size > 0:
                correlogram += cp.bincount(a[matches].ravel(), minlength=bins).astype(cp.float32)
                count += cp.bincount(a.ravel(), minlength=bins).astype(cp.float32)

        correlogram = correlogram / (count + 1e-6)
        features.append(correlogram)

    t1 = time.time() - st
    return cp.asnumpy(cp.concatenate(features).astype(cp.float32)), t1


def tamura_features(img, blocks=6):
    st = time.time()

    img_gpu = cp.asarray(img)
    gray = bgr_to_gray_gpu(img_gpu)
    h, w = gray.shape
    fh, fw = h // blocks, w // blocks

    feats = []
    gx_all, gy_all = sobel_xy_gpu(gray)

    for i in range(3):
        for j in range(2):
            patch = gray[i * fh:(i + 1) * fh, j * fw:(j + 1) * fw].astype(cp.float32)
            gx = gx_all[i * fh:(i + 1) * fh, j * fw:(j + 1) * fw]
            gy = gy_all[i * fh:(i + 1) * fh, j * fw:(j + 1) * fw]

            var = cp.var(patch)
            contrast = cp.std(patch)
            ang = cp.mean(cp.arctan2(gy, gx))
            feats.extend([var, contrast, ang])

    feats = cp.asarray(feats, dtype=cp.float32)
    t1 = time.time() - st
    return cp.asnumpy(feats), t1


def edge_histogram(img, grid=4):
    st = time.time()

    img_gpu = cp.asarray(img)
    gray = bgr_to_gray_gpu(img_gpu).astype(cp.float32)
    h, w = gray.shape
    block_h, block_w = h // grid, w // grid

    h2 = block_h * grid
    w2 = block_w * grid
    gray = gray[:h2, :w2]

    filters = [
        cp.array([[-1,  1], [-1,  1]], dtype=cp.float32),  # vertical
        cp.array([[-1, -1], [ 1,  1]], dtype=cp.float32),  # horizontal
        cp.array([[ 1, -1], [-1,  1]], dtype=cp.float32),  # 45deg
        cp.array([[-1,  1], [ 1, -1]], dtype=cp.float32),  # 135deg
        cp.array([[ 1,  1], [ 1,  1]], dtype=cp.float32),  # non-directional
    ]

    edge_hist = cp.empty(grid * grid * 5, dtype=cp.float32)
    k = 0

    for f in filters:
        filtered = cnd.convolve(gray, f, mode="reflect")
        binary = (filtered > 0).astype(cp.float32)

        block_sums = (
            binary.reshape(grid, block_h, grid, block_w)
                  .transpose(0, 2, 1, 3)
                  .sum(axis=(2, 3))
        )

        vals = block_sums.reshape(-1)
        edge_hist[k:k + grid * grid] = vals
        k += grid * grid

    edge_hist = edge_hist.reshape(5, grid * grid).T.reshape(-1)
    edge_hist /= edge_hist.sum() + 1e-8

    t1 = time.time() - st
    return cp.asnumpy(edge_hist), t1


def jcd_descriptor(img, color_bins=64, edge_bins=272):
    st = time.time()

    img_gpu = cp.asarray(img)
    img_resized = resize_bgr_gpu(img_gpu, (128, 128))

    hsv = bgr_to_hsv_gpu(img_resized)

    hb = cp.clip(hsv[..., 0].astype(cp.int32) // (180 // 4), 0, 3)
    sb = cp.clip(hsv[..., 1].astype(cp.int32) // (256 // 4), 0, 3)
    vb = cp.clip(hsv[..., 2].astype(cp.int32) // (256 // 4), 0, 3)
    hist_idx = hb * 16 + sb * 4 + vb
    hist = cp.bincount(hist_idx.ravel(), minlength=64).astype(cp.float32)
    hist /= (hist.sum() + 1e-6)

    gray = bgr_to_gray_gpu(img_resized)
    gx, gy = sobel_xy_gpu(gray)
    _, ang = cart_to_polar_gpu(gx, gy, angle_in_degrees=True)
    bins_idx = (ang / 360.0 * edge_bins).astype(cp.int32) % edge_bins
    edge_hist = cp.bincount(bins_idx.ravel(), minlength=edge_bins).astype(cp.float32)
    edge_hist /= (edge_hist.sum() + 1e-6)

    feat = cp.concatenate([hist, edge_hist]).astype(cp.float32)
    t1 = time.time() - st
    return cp.asnumpy(feat), t1


def PHOG(img, bins=14, levels=5):
    st = time.time()

    img_gpu = cp.asarray(img)
    gray = bgr_to_gray_gpu(img_gpu)

    gx, gy = sobel_xy_gpu(gray)
    mag, ang = cart_to_polar_gpu(gx, gy, angle_in_degrees=False)
    ang = cp.mod(ang, cp.pi)

    bin_idx = cp.floor(ang * bins / cp.pi).astype(cp.int32)
    bin_idx = cp.clip(bin_idx, 0, bins - 1)

    h, w = gray.shape
    feats = []

    for l in range(levels):
        n_cells = 2 ** l
        ys = np.linspace(0, h, n_cells + 1, dtype=int)
        xs = np.linspace(0, w, n_cells + 1, dtype=int)

        for i in range(n_cells):
            y0, y1 = ys[i], ys[i + 1]
            for j in range(n_cells):
                x0, x1 = xs[j], xs[j + 1]

                patch_bins = bin_idx[y0:y1, x0:x1]
                patch_mag = mag[y0:y1, x0:x1]

                if patch_bins.size == 0:
                    hist = cp.zeros(bins, dtype=cp.float32)
                else:
                    hist = cp.bincount(
                        patch_bins.ravel(),
                        weights=patch_mag.ravel(),
                        minlength=bins
                    ).astype(cp.float32)
                    s = hist.sum()
                    if s > 0:
                        hist /= s

                feats.append(hist)

    feats = cp.concatenate(feats).astype(cp.float32)

    target = 630
    if feats.size >= target:
        feats = feats[:target]
    else:
        feats = cp.pad(feats, (0, target - feats.size))

    return cp.asnumpy(feats), time.time() - st


def _zigzag_scan(matrix):
    matrix = cp.asarray(matrix)
    h, w = matrix.shape
    result = []

    for s in range(h + w - 1):
        if s % 2 == 0:
            for i in range(min(s, h - 1), max(-1, s - w), -1):
                j = s - i
                if 0 <= i < h and 0 <= j < w:
                    result.append(matrix[i, j])
        else:
            for j in range(min(s, w - 1), max(-1, s - h), -1):
                i = s - j
                if 0 <= i < h and 0 <= j < w:
                    result.append(matrix[i, j])

    return cp.asarray(result, dtype=cp.float32)


def _block_average_channel(channel, blocks=8):
    channel = cp.asarray(channel).astype(cp.float32)
    h, w = channel.shape
    ys = np.linspace(0, h, blocks + 1, dtype=int)
    xs = np.linspace(0, w, blocks + 1, dtype=int)

    out = cp.zeros((blocks, blocks), dtype=cp.float32)

    for y in range(blocks):
        for x in range(blocks):
            block = channel[ys[y]:ys[y + 1], xs[x]:xs[x + 1]]
            out[y, x] = 0.0 if block.size == 0 else cp.mean(block)

    return out


def color_layout(img):
    st = time.time()

    img_gpu = cp.asarray(img)
    if img_gpu is None:
        raise ValueError("img is None")
    if img_gpu.ndim != 3 or img_gpu.shape[2] < 3:
        raise ValueError("img must be a color image with 3 channels")
    if img_gpu.shape[2] == 4:
        img_gpu = img_gpu[:, :, :3]

    ycrcb = bgr_to_ycrcb_gpu(img_gpu)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

    Y_blocks = _block_average_channel(Y, blocks=8)
    Cr_blocks = _block_average_channel(Cr, blocks=8)
    Cb_blocks = _block_average_channel(Cb, blocks=8)

    Y_dct = dct2_8x8_gpu(Y_blocks)
    Cr_dct = dct2_8x8_gpu(Cr_blocks)
    Cb_dct = dct2_8x8_gpu(Cb_blocks)

    Y_zigzag = _zigzag_scan(Y_dct)
    Cr_zigzag = _zigzag_scan(Cr_dct)
    Cb_zigzag = _zigzag_scan(Cb_dct)

    nY, nCr, nCb = 21, 6, 6
    features = cp.concatenate([
        Y_zigzag[:nY],
        Cr_zigzag[:nCr],
        Cb_zigzag[:nCb]
    ]).astype(cp.float32)

    t1 = time.time() - st
    return cp.asnumpy(features), t1


def weighted_correlation(glcm):
    glcm = cp.asarray(glcm)
    i, j = cp.indices(glcm.shape)
    mean_i = cp.sum(i * glcm)
    mean_j = cp.sum(j * glcm)
    std_i = cp.sqrt(cp.sum(glcm * (i - mean_i) ** 2))
    std_j = cp.sqrt(cp.sum(glcm * (j - mean_j) ** 2))
    corr = cp.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j + 1e-10)
    return corr


def haralick_features(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    st = time.time()

    img_gpu = cp.asarray(img)
    if img_gpu.ndim == 3:
        gray = bgr_to_gray_gpu(img_gpu)
    else:
        gray = img_gpu.astype(cp.uint8)

    features_list = []
    for theta in angles:
        glcm = compute_glcm(gray, distances=distances, angles=[theta], levels=levels)
        i, j = cp.indices(glcm.shape)

        contrast = cp.sum(glcm * (i - j) ** 2)
        energy = cp.sum(glcm ** 2)
        homogeneity = cp.sum(glcm / (1.0 + cp.abs(i - j)))
        correlation = weighted_correlation(glcm)
        entropy = -cp.sum(glcm * cp.log(glcm + 1e-10))
        dissimilarity = cp.sum(glcm * cp.abs(i - j))

        features_list.append(cp.stack([
            contrast, energy, homogeneity, correlation, entropy, dissimilarity
        ]))

    feat = cp.mean(cp.stack(features_list), axis=0)
    return cp.asnumpy(feat), time.time() - st


def haralick_features14(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
    st = time.time()

    img_gpu = cp.asarray(img)
    if img_gpu.ndim == 3:
        gray = bgr_to_gray_gpu(img_gpu)
    else:
        gray = img_gpu.astype(cp.uint8)

    eps = 1e-10
    features_list = []

    for theta in angles:
        glcm = compute_glcm(gray, distances=distances, angles=[theta], levels=levels)
        P = glcm.astype(cp.float64)
        P /= (P.sum() + eps)

        Ng = P.shape[0]
        i, j = cp.indices(P.shape)

        px = cp.sum(P, axis=1)
        py = cp.sum(P, axis=0)

        ux = cp.sum(i * P)
        uy = cp.sum(j * P)

        sigx = cp.sqrt(cp.sum((i - ux) ** 2 * P))
        sigy = cp.sqrt(cp.sum((j - uy) ** 2 * P))

        asm = cp.sum(P ** 2)
        contrast = cp.sum((i - j) ** 2 * P)
        correlation = cp.sum((i - ux) * (j - uy) * P) / (sigx * sigy + eps)
        variance = cp.sum((i - ux) ** 2 * P)
        homogeneity = cp.sum(P / (1 + (i - j) ** 2))
        entropy = -cp.sum(P * cp.log(P + eps))

        # GPU vectorized sum/diff distributions
        x_idx, y_idx = cp.indices((Ng, Ng))
        sum_idx = (x_idx + y_idx).ravel()
        diff_idx = cp.abs(x_idx - y_idx).ravel()
        p_sum = cp.bincount(sum_idx, weights=P.ravel(), minlength=2 * Ng).astype(cp.float64)
        p_diff = cp.bincount(diff_idx, weights=P.ravel(), minlength=Ng).astype(cp.float64)

        arr_sum = cp.arange(2 * Ng, dtype=cp.float64)
        sum_avg = cp.sum(arr_sum * p_sum)
        sum_entropy = -cp.sum(p_sum * cp.log(p_sum + eps))
        sum_var = cp.sum(((arr_sum - sum_entropy) ** 2) * p_sum)

        diff_var = cp.var(p_diff)
        diff_entropy = -cp.sum(p_diff * cp.log(p_diff + eps))

        HX = -cp.sum(px * cp.log(px + eps))
        HY = -cp.sum(py * cp.log(py + eps))
        HXY = entropy
        pxpy = px[:, None] * py[None, :]
        HXY1 = -cp.sum(P * cp.log(pxpy + eps))
        HXY2 = -cp.sum(pxpy * cp.log(pxpy + eps))

        imc1 = (HXY - HXY1) / (cp.maximum(HX, HY) + eps)
        imc2 = cp.sqrt(cp.maximum(0.0, 1 - cp.exp(-2 * (HXY2 - HXY))))

        max_prob = cp.max(P)

        features = cp.stack([
            asm,
            contrast,
            correlation,
            variance,
            homogeneity,
            sum_avg,
            sum_var,
            sum_entropy,
            entropy,
            diff_var,
            diff_entropy,
            imc1,
            imc2,
            max_prob
        ])

        features_list.append(features)

    feat = cp.mean(cp.stack(features_list), axis=0)
    return cp.asnumpy(feat), time.time() - st


# ============================================================
# Extraction
# ============================================================

def extract_lire(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    features = {}
    features["color_layout"] = color_layout(img)
    features["phog"] = PHOG(img)
    features["color_correlogram"] = color_correlogram(img)
    features["tamura"] = tamura_features(img)
    features["edge_histogram"] = edge_histogram(img)
    features["jcd"] = jcd_descriptor(img)

    return features


def _write_descriptor_csv(storage_path, filename, pindex, rows):
    df = pd.DataFrame(rows, index=pindex).rename_axis("img")
    df.to_csv(os.path.join(storage_path, filename), index=True)


def gpu_lires(dataset=None, paths=None, label=None, storage_path=None, batch_size=1000):
    if paths is None or len(paths) == 0:
        raise ValueError("paths must be a non-empty list")
    if storage_path is None:
        raise ValueError("storage_path must be provided")

    os.makedirs(storage_path, exist_ok=True)
    pindex = [os.path.basename(p) for p in paths]

    # Read once to avoid repeated disk IO
    images = []
    class_names = []
    file_names = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Could not read image: {p}")
        images.append(img)
        class_names.append(os.path.basename(os.path.dirname(p)))
        file_names.append(os.path.basename(p))

    descriptor_specs = [
        ("color_layout.csv", "color_layout", color_layout, 33),
        ("color_correlogram.csv", "color_correlogram", color_correlogram, 1024),
        ("tamura_features.csv", "tamura_features", tamura_features, 18),
        ("edge_histogram.csv", "edge_histogram", edge_histogram, 80),
        ("jcd_descriptor.csv", "jcd_descriptor", jcd_descriptor, 336),  # 64 + 272
        ("phog.csv", "phog", PHOG, 630),
        ("haralick_6.csv", "haralick_6", haralick_features, 6),
        ("haralick_14.csv", "haralick_14", haralick_features14, 14),
    ]

    for csv_name, label_name, fn, feat_len in descriptor_specs:
        total_time = 0.0
        rows = []

        for cls, fname, img in zip(class_names, file_names, images):
            feat, t1 = fn(img)
            total_time += t1
            rows.append([cls, fname, *feat.tolist()])

        _write_descriptor_csv(storage_path, csv_name, pindex, rows)

        fps = len(paths) / max(total_time, 1e-8)
        print(f"FPS for {label_name} is {fps:.4f}")