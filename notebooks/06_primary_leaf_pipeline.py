import os
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.segmentation import segment_primary_leaf, save_outputs
from src.features import extract_features_extended

INPUT_DIR = ROOT / "Data_Balanced"
SEGMENTED_DIR = ROOT / "data/2_data_primary_leaf"
FEATURES_CSV = ROOT / "data/3_data_extract_features_primary/features_dataset.csv"
TARGET_SIZE = (224, 224)
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def segment_all():
    for class_folder in sorted(os.listdir(INPUT_DIR)):
        class_path = INPUT_DIR / class_folder
        if not class_path.is_dir():
            continue
        out_class = SEGMENTED_DIR / class_folder
        out_class.mkdir(parents=True, exist_ok=True)
        for img_file in tqdm(os.listdir(class_path), desc=f"Segmentando {class_folder}"):
            if not img_file.lower().endswith(VALID_EXTS):
                continue
            img_path = class_path / img_file
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            seg, mask, overlay = segment_primary_leaf(resized)
            stem = Path(img_file).stem
            save_outputs(seg, mask, overlay, out_class, stem)

def extract_all():
    rows = []
    names_ref = None
    for class_folder in sorted(os.listdir(SEGMENTED_DIR)):
        class_path = SEGMENTED_DIR / class_folder
        if not class_path.is_dir():
            continue
        for img_file in tqdm(os.listdir(class_path), desc=f"Features {class_folder}"):
            if "_primary.png" not in img_file:
                continue
            stem = img_file.replace("_primary.png", "")
            img_path = class_path / img_file
            mask_path = class_path / f"{stem}_mask.png"
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            mask = None
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), 0)
            feats, names = extract_features_extended(image, mask)
            if names_ref is None:
                names_ref = names
            row = {names_ref[i]: feats[i] for i in range(len(names_ref))}
            row["label"] = class_folder
            rows.append(row)
    df = pd.DataFrame(rows)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)

if __name__ == "__main__":
    segment_all()
    extract_all()
