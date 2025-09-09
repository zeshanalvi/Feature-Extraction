import cv2, numpy as np, os
import pandas as pd
from skimage import feature
import warnings
warnings.filterwarnings('ignore')


#from skimage.feature import greycomatrix, greycoprops
#from skimage.feature.texture import greycomatrix, greycoprops

from skimage.feature import hog
from skimage.color import rgb2gray
#import mahotas
from image_read import Dataset
from skimage.feature import hog
from skimage import io, color


def compute_glcm(img, distances=[1], angles=[0], levels=256):
    h, w = img.shape
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    for d in distances:
        for theta in angles:
            dx, dy = int(np.round(np.cos(theta))), int(np.round(np.sin(theta)))
            for i in range(h - dy):
                for j in range(w - dx):
                    p = img[i, j]
                    q = img[i + dy, j + dx]
                    glcm[p, q] += 1
    return glcm

def quantize_image(img, bins_per_channel=8):
    """Reduce colors to a smaller palette."""
    bins = np.linspace(0, 256, bins_per_channel+1, dtype=np.int32)
    quantized = np.digitize(img, bins) - 1
    return quantized

import cv2
import numpy as np

def color_correlogram(img, bins=32, distances=32):
    """
    Extracts a color correlogram of size bins*distances (default: 1024).
    
    Args:
        img: input BGR image (OpenCV format)
        bins: number of quantized colors (default 32)
        distances: number of distances (default 32)

    Returns:
        feature vector of shape (bins * distances,) e.g. (1024,)
    """
    # Resize for consistency
    img_small = cv2.resize(img, (64, 64))

    # Quantize colors into bins using HSV (better for color perception)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = 4, 4, 2  # 4*4*2 = 32 bins
    quant = np.floor_divide(hsv[..., 0], 180 // h_bins) * (s_bins * v_bins) \
            + np.floor_divide(hsv[..., 1], 256 // s_bins) * v_bins \
            + np.floor_divide(hsv[..., 2], 256 // v_bins)
    idx = quant.astype(int)

    h, w = idx.shape
    features = []

    # For each distance d
    for d in range(1, distances+1):
        correlogram = np.zeros(bins, dtype=np.float32)
        count = np.zeros(bins, dtype=np.float32)

        for y in range(h):
            for x in range(w):
                color = idx[y, x]
                # Check right neighbor
                if x+d < w:
                    if idx[y, x+d] == color:
                        correlogram[color] += 1
                # Check down neighbor
                if y+d < h:
                    if idx[y+d, x] == color:
                        correlogram[color] += 1
                count[color] += 2  # two checks per pixel

        correlogram = correlogram / (count + 1e-6)  # normalize per color
        features.extend(correlogram)

    return np.array(features)  # (1024,)


def color_correlogram3(img, bins=32, distances=32):
    img_small = cv2.resize(img, (64, 64))
    quant = np.floor_divide(img_small, 256 // (bins // 3))
    idx = quant[..., 0]*(bins//3)**2 + quant[..., 1]*(bins//3) + quant[..., 2]

    h, w = idx.shape
    features = []
    for d in range(1, distances+1):  # 32 distances
        correlogram = np.zeros(bins)
        for y in range(h):
            for x in range(w):
                color = idx[y, x]
                if y+d < h and idx[y+d, x] == color: correlogram[color]+=1
                if x+d < w and idx[y, x+d] == color: correlogram[color]+=1
        features.extend(correlogram)
    return np.array(features)  # (1024,)
#def color_correlogram1(img, distances=[1, 3, 5], bins_per_channel=8):

def color_correlogram1(img, distances=[32], bins_per_channel=32):
    """
    Compute Color Correlogram of an image.
    
    Parameters:
    - img: BGR image
    - distances: list of pixel distances
    - bins_per_channel: quantization of colors
    
    Returns:
    - correlogram: flattened numpy array
    """
    h, w, c = img.shape
    q_img = quantize_image(img, bins_per_channel)
    num_colors = bins_per_channel ** 3
    correlogram = np.zeros((len(distances), num_colors))

    # Convert each pixel color to a single index
    color_idx = (q_img[:,:,0] * bins_per_channel * bins_per_channel +
                 q_img[:,:,1] * bins_per_channel +
                 q_img[:,:,2])

    coords = np.array([(y, x) for y in range(h) for x in range(w)])

    for d_index, d in enumerate(distances):
        for y, x in coords:
            center_color = color_idx[y, x]
            
            # 8 neighbors at distance d
            neighbors = [
                (y-d, x), (y+d, x), (y, x-d), (y, x+d),
                (y-d, x-d), (y-d, x+d), (y+d, x-d), (y+d, x+d)
            ]
            for ny, nx in neighbors:
                if 0 <= ny < h and 0 <= nx < w:
                    neigh_color = color_idx[ny, nx]
                    correlogram[d_index, center_color] += (center_color == neigh_color)

    # Normalize
    correlogram /= correlogram.sum(axis=1, keepdims=True) + 1e-8
    return correlogram.flatten()

def tamura_features(img, blocks=6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fh, fw = h // blocks, w // blocks

    feats = []
    for i in range(3):  # 3 regions vertically
        for j in range(2):  # 2 regions horizontally = 6 patches
            patch = gray[i*fh:(i+1)*fh, j*fw:(j+1)*fw]
            var = np.var(patch)
            contrast = patch.std()
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
            ang = np.arctan2(gy, gx).mean()
            feats.extend([var, contrast, ang])
    return np.array(feats)  # (18,)

def tamura_features2(img, distances=[1], angles=[0], levels=256):
    """
    Compute basic Tamura features: contrast, energy, homogeneity (from GLCM)
    img: grayscale image (0-255)
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        img = (color.rgb2gray(img) * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    glcm = compute_glcm(img, distances=distances, angles=angles, levels=levels)

    # Contrast
    i, j = np.indices(glcm.shape)
    contrast = np.sum(glcm * (i - j)**2)

    # Energy (angular second moment)
    energy = np.sum(glcm**2)

    # Homogeneity (inverse difference moment)
    homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))

    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity
    }

def tamura_features1(img, distances=[1], angles=[0]):
    """
    Approximate Tamura texture features using GLCM (coarseness, contrast, directionality).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm=compute_glcm(gray)
    #glcm = greycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').flatten()[0]
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()[0]
    energy = greycoprops(glcm, 'energy').flatten()[0]

    return np.array([contrast, homogeneity, energy])

def edge_histogram(img, grid=4):
    """
    Edge Histogram Descriptor (EHD) approximation.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block_h, block_w = h // grid, w // grid

    edge_hist = []

    # Edge filters
    filters = {
        "vertical": np.array([[-1, 1], [-1, 1]]),
        "horizontal": np.array([[-1, -1], [1, 1]]),
        "45deg": np.array([[1, -1], [-1, 1]]),
        "135deg": np.array([[-1, 1], [1, -1]]),
        "non-directional": np.array([[1, 1], [1, 1]])
    }

    for gy in range(grid):
        for gx in range(grid):
            block = gray[gy*block_h:(gy+1)*block_h, gx*block_w:(gx+1)*block_w]
            block_hist = []
            for f in filters.values():
                filtered = cv2.filter2D(block, -1, f)
                block_hist.append(np.sum(filtered > 0))
            edge_hist.extend(block_hist)

    edge_hist = np.array(edge_hist, dtype=np.float32)
    edge_hist /= edge_hist.sum() + 1e-8
    return edge_hist

def jcd_descriptor(img, color_bins=64, edge_bins=272):
    # Resize for consistency
    img_resized = cv2.resize(img, (128, 128))
    
    # ---- Color Histogram part ----
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 64 bins

    # ---- Edge Histogram part ----
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.int32(ang / 360 * edge_bins) % edge_bins
    edge_hist, _ = np.histogram(bins, bins=edge_bins, range=(0, edge_bins))
    edge_hist = edge_hist.astype("float32")
    edge_hist /= (edge_hist.sum() + 1e-6)

    # Concatenate → 64 + 274 = 338
    return np.hstack([hist, edge_hist])

def jcd_descriptor1(img):
    """
    Approximation of JCD (Joint Composite Descriptor).
    Combines color histogram + edge histogram + Tamura.
    """
    # Color histogram
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
    color_hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()

    # Combine with edge and texture
    edge_feat = edge_histogram(img)
    texture_feat = tamura_features(img)
    print(color_hist.shape, edge_feat.shape, texture_feat['contrast'])

    return color_hist #np.concatenate([color_hist, edge_feat, texture_feat])

def PHOG(img, bins=14, levels=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    phog_vec = []
    h, w = gray.shape

    for l in range(levels):
        step_y, step_x = h // (2**l), w // (2**l)
        for i in range(2**l):
            for j in range(2**l):
                patch = gray[i*step_y:(i+1)*step_y, j*step_x:(j+1)*step_x]
                if patch.size > 0:
                    feat = hog(patch, orientations=bins,
                               pixels_per_cell=(8,8), cells_per_block=(1,1),
                               feature_vector=True)
                    phog_vec.extend(feat[:bins])  
    return np.array(phog_vec[:630])  # (630,)

def PHOG1(img):
   # -------------------- SHAPE FEATURES --------------------
    # 5. PHOG (via HOG)
    phog_features, _ = hog(
        rgb2gray(img),
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        orientations=9,
        visualize=True,
        block_norm="L2-Hys"
    )
    return phog_features

def color_layout(img):
   # 2. Color Layout Descriptor (approx using mean color of blocks)
    h, w, _ = img.shape
    blocks = 4
    block_h, block_w = h // blocks, w // blocks
    color_layout = []
    for y in range(blocks):
        for x in range(blocks):
            block = img[y*block_h:(y+1)*block_h, x*block_w:(x+1)*block_w]
            color_layout.extend(cv2.mean(block)[:3])  # average BGR
    return np.array(color_layout)
   
def weighted_correlation(glcm):
    levels = glcm.shape[0]
    i, j = np.indices(glcm.shape)
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)
    std_i = np.sqrt(np.sum(glcm * (i - mean_i) ** 2))
    std_j = np.sqrt(np.sum(glcm * (j - mean_j) ** 2))
    corr = np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j + 1e-10)
    return corr

def haralick_features(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    if len(img.shape) == 3:
        img = (color.rgb2gray(img) * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    features_list = []
    for theta in angles:
        glcm = compute_glcm(img, distances=distances, angles=[theta], levels=levels)
        i, j = np.indices(glcm.shape)

        contrast = np.sum(glcm * (i - j) ** 2)
        energy = np.sum(glcm ** 2)
        homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))
        correlation = weighted_correlation(glcm)
        entropy = -np.sum(glcm * np.log(glcm + 1e-10))
        dissimilarity = np.sum(glcm * np.abs(i - j))

        features_list.append([contrast, energy, homogeneity, correlation, entropy, dissimilarity])
    
    return np.mean(features_list, axis=0)

def extract_lire(image_path):
    img = cv2.imread(image_path)
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}
    # -------------------- COLOR FEATURES --------------------
    # 1. Color Histogram
    #hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0,256, 0,256, 0,256])
    #hist = cv2.normalize(hist, hist).flatten()
    #features['color_histogram'] = hist

    features['color_layout'] = color_layout(img)

    # -------------------- TEXTURE FEATURES --------------------
    # 3. Haralick features (Tamura alternative)
    #features['haralick'] = haralick_features(img_gray)

    # 4. Local Binary Pattern (LBP)
    #lbp = mahotas.features.lbp(img_gray, radius=2, points=16)
    #features['lbp'] = lbp

    
    features['phog'] = PHOG(img)
    features['color_correlogram']=color_correlogram(img)
    features['tamura']=tamura_features(img)
    features['edge_histogram']=edge_histogram(img)
    features['jcd']=jcd_descriptor(img)

    return features

def get_lires(dataset,paths=None,label=None,storage_path=None,batch_size=1000):
  #paths,label,labels=gather_paths_all(jpg_path=data_path,num_classes=num_classes)
  


  pindex=[p.split("/")[-1] for p in paths]


  ary=np.zeros((len(paths),48),np.float64)
  for i,p in enumerate(paths):
     ary[i,:]=color_layout(cv2.imread(paths[i]))
  df = pd.DataFrame(ary, index=pindex).to_csv(storage_path+"color_layout.csv",index=True)

  ary=np.zeros((len(paths),630),np.float64)
  for i,p in enumerate(paths):
     ary[i,:]=PHOG(cv2.imread(paths[i]))
  df = pd.DataFrame(ary, index=pindex).to_csv(storage_path+"phog.csv",index=True)
  
  ary=np.zeros((len(paths),1024),np.float64)
  for i,p in enumerate(paths):
     ary[i,:]=color_correlogram(cv2.imread(paths[i]))
  df = pd.DataFrame(ary, index=pindex).to_csv(storage_path+"color_correlogram.csv",index=True)
  
  ary=np.zeros((len(paths),18),np.float64)
  for i,p in enumerate(paths):
     ary[i,:]=tamura_features(cv2.imread(paths[i]))
  df = pd.DataFrame(ary, index=pindex).to_csv(storage_path+"tamura_features.csv",index=True)
  
  ary=np.zeros((len(paths),80),np.float64)
  for i,p in enumerate(paths):
     ary[i,:]=edge_histogram(cv2.imread(paths[i]))
  df = pd.DataFrame(ary, index=pindex).to_csv(storage_path+"edge_histogram.csv",index=True)
  
  ary=np.zeros((len(paths),336),np.float64)
  for i,p in enumerate(paths):
     ary[i,:]=jcd_descriptor(cv2.imread(paths[i]))
  df = pd.DataFrame(ary, index=pindex).to_csv(storage_path+"jcd_descriptor.csv",index=True)


data_path_train="D:\\datasets\\Kvasirv3_dev\\dev\\"
num_classes=23
storage_path="D:\\datasets\\Kvasirv3_dev\\test\\"
kvasir=Dataset(num_classes=num_classes)
print(kvasir.label_map)
paths,label,labels=kvasir.gather_paths_all(jpg_path=data_path_train)
get_lires(kvasir,paths=paths,label=label,storage_path=storage_path,batch_size=2000)
