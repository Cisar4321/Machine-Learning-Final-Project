import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

resize_dir = Path("data/2_data_resize")

class_vectors = {}
class_names = []

# 1. Promediar imágenes por clase
for class_name in sorted(os.listdir(resize_dir)):
    class_path = resize_dir / class_name
    if not class_path.is_dir():
        continue

    class_images = []

    for img_file in os.listdir(class_path):
        img_path = class_path / img_file
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Flatten para convertir en vector
        class_images.append(img.flatten())

    # Vector promedio representativo de la clase
    mean_vector = np.mean(np.array(class_images), axis=0)
    class_vectors[class_name] = mean_vector
    class_names.append(class_name)

# 2. Construir matriz de correlación entre clases
matrix = []
for c1 in class_names:
    row = []
    for c2 in class_names:
        corr = np.corrcoef(class_vectors[c1], class_vectors[c2])[0, 1]
        row.append(corr)
    matrix.append(row)

corr_matrix = np.array(matrix)

# 3. Graficar heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="viridis",
    fmt=".2f"
)
plt.title("Correlación entre clases de hojas de papa (promedios por clase)")
plt.show()
