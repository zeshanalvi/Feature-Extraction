from .Lire import get_lires, color_correlogram, color_layout, compute_glcm, edge_histogram, extract_lire, tamura_features, jcd_descriptor, PHOG
from .LBP import get_lbps, get_lbp_single
from .image_read import Dataset
def extract_features_batch_old(img_path,labeled=False):
    dfs = [] 
    for img in img_path:
        print(img)
        dfs.append(extract_features_single(img_path=img,labeled=labeled))
    return pd.concat(dfs)

def extract_features_batch(img_path, labeled=False):
    dfs = []

    # If a single tensor or image path was passed, wrap it in a list for uniformity
    if isinstance(img_path, (torch.Tensor, np.ndarray, str)):
        img_path = [img_path]

    for img in img_path:
        # Handle tensor input
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()

            # Convert from (C,H,W) → (H,W,C)
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))

            # Scale to uint8 if in [0,1]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)

            # Give a fake name for consistency
            img_name = "tensor_input"

            dfs.append(extract_features_single(img_path=img_name, labeled=labeled, image_data=img_np))

        # Handle string path input
        elif isinstance(img, str):
            dfs.append(extract_features_single(img_path=img, labeled=labeled))

        else:
            raise TypeError(f"Unsupported input type: {type(img)}")

    return pd.concat(dfs, ignore_index=True)
        
def extract_features_single(img_path,labeled=False):
    """
    Extracts all features for a single image using the same structure
    as the batch feature generation code.
    Returns a pandas DataFrame (with class label and combined features).
    """

    # Extract class label from folder name
    if (labeled):
        label = img_path.split("/")[-2]
    else:
        label="NA"
    img_name = img_path.split("/")[-1]
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    total_time = 0.0
    features_list = []
    feature_names = []

    
    # ========== color_correlogram ==========
    feats, t2 = color_correlogram(img)
    features_list.extend(feats)
    feature_names.extend([f"{i}_0" for i in range(1, 1025)])  # 1024 features
    total_time += t2

    # ========== color_layout ==========
    feats, t1 = color_layout(img)
    features_list.extend(feats)
    feature_names.extend([f"{i}_1" for i in range(1, 49)])  # 48 features
    total_time += t1

    # ========== edge_histogram ==========
    feats, t4 = edge_histogram(img)
    features_list.extend(feats)
    feature_names.extend([f"{i}_2" for i in range(1, 81)])  # 80 features
    total_time += t4

    # ========== jcd_descriptor ==========
    feats, t5 = jcd_descriptor(img)
    features_list.extend(feats)
    feature_names.extend([f"{i}_3" for i in range(1, 337)])  # 336 features
    total_time += t5

    # ========== LBP ==========
    _, lbp_hist, _, _ = get_lbp_single(image=img,radius=1)
    features_list.extend(lbp_hist)
    feature_names.extend([f"{i}_4" for i in range(len(lbp_hist))])


    # ========== LBP ==========
    _, lbp_hist, _, _ = get_lbp_single(image=img,radius=2)
    features_list.extend(lbp_hist)
    feature_names.extend([f"{i}_5" for i in range(len(lbp_hist))])


    # ========== LBP ==========
    _, lbp_hist, _, _ = get_lbp_single(image=img,radius=3)
    features_list.extend(lbp_hist)
    feature_names.extend([f"{i}_6" for i in range(len(lbp_hist))])


    # ========== LBP ==========
    _, lbp_hist, _, _ = get_lbp_single(image=img,radius=4)
    features_list.extend(lbp_hist)
    feature_names.extend([f"{i}_7" for i in range(len(lbp_hist))])


    # ========== LBP ==========
    _, lbp_hist, _, _ = get_lbp_single(image=img,radius=5)
    features_list.extend(lbp_hist)
    feature_names.extend([f"{i}_8" for i in range(len(lbp_hist))])
    

    # ========== PHOG ==========
    feats, t6 = PHOG(img)
    features_list.extend(feats)
    feature_names.extend([f"{i}_9" for i in range(1, 631)])  # 630 features
    total_time += t6
    

    # ========== tamura_features ==========
    feats, t3 = tamura_features(img)
    features_list.extend(feats)
    feature_names.extend([f"{i}_10" for i in range(1, 19)])  # 18 features
    total_time += t3

    # Combine everything into DataFrame
    df = pd.DataFrame([[label] + features_list],
                      columns=["0"] + feature_names,
                      index=[img_name])
    df.index.name = "img"

    print(f"Total feature extraction time: {total_time:.4f} s")
    print(f"FPS: {1/total_time if total_time>0 else 0:.2f}")
    df.drop(columns={'0'},axis=1, inplace=True)
    df.rename(columns={'0_4': 'new_col_4', '0_5': 'new_col_5','0_6': 'new_col_6', '0_7': 'new_col_7', '0_8': 'new_col_8'}, inplace=True)
    return df
