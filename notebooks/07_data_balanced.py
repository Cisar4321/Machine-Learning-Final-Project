import os
from pathlib import Path
import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/1_data_original"
DST = ROOT / "Data_Balanced"
VALID = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=15,
    horizontal_flip=True,
    brightness_range=[0.6, 1.6],
    channel_shift_range=35
)

def count_class_images(path):
    return len([f for f in os.listdir(path) if f.lower().endswith(VALID)])

def copy_originals(src_class, dst_class):
    dst_class.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(src_class):
        if not f.lower().endswith(VALID):
            continue
        (dst_class / f).write_bytes((src_class / f).read_bytes())

def augment_to_target(src_class, dst_class, target_count, target_size=(224, 224)):
    current = count_class_images(dst_class)
    if current >= target_count:
        return
    files = [f for f in os.listdir(src_class) if f.lower().endswith(VALID)]
    i = 0
    while current < target_count and files:
        fname = files[i % len(files)]
        img_path = src_class / fname
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        batches = datagen.flow(x, batch_size=1, save_to_dir=str(dst_class), save_prefix="aug", save_format="jpg")
        next(batches)
        current += 1
        i += 1

def main():
    classes = [d for d in os.listdir(SRC) if (SRC / d).is_dir()]
    counts = {c: count_class_images(SRC / c) for c in classes}
    target = max(counts.values()) if counts else 0
    for c in classes:
        src_class = SRC / c
        dst_class = DST / c
        copy_originals(src_class, dst_class)
        augment_to_target(src_class, dst_class, target)

if __name__ == "__main__":
    main()
