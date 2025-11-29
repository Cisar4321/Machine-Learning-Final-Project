import streamlit as st
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("Proceso")

root = Path(__file__).resolve().parents[2]
data_dir = root / "data" / "1_data_original"
balanced_dir = root / "Data_Balanced"

st.subheader("1. Dataset y balanceo")
classes = [d.name for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []
orig_counts = []
for c in classes:
    p = data_dir / c
    orig_counts.append(len([f for f in os.listdir(p) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]))
df_counts = pd.DataFrame({"Clase": classes, "Original": orig_counts})
if balanced_dir.exists():
    bal_counts = []
    for c in classes:
        p = balanced_dir / c
        bal_counts.append(len([f for f in os.listdir(p) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]))
    df_counts["Balanceado"] = bal_counts
fig1, ax1 = plt.subplots(figsize=(8,4))
sns.barplot(df_counts.melt(id_vars=["Clase"], var_name="Tipo", value_name="Cantidad"), x="Clase", y="Cantidad", hue="Tipo", ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
st.pyplot(fig1)
st.write("""
El dataset se organiza por carpetas, cada una representa una clase de enfermedad. Se observa desbalanceo natural entre clases (por ejemplo, más 'Fungi' que 'Healthy'). Para mitigar el desbalance y mejorar la generalización, se generan nuevas imágenes por clase mediante aumentación y se guardan en 'Data_Balanced/'. Esto conserva la estructura por clases y facilita entrenamientos más estables.
""")

st.subheader("2. Preprocesamiento y segmentación de hoja primaria")
if classes:
    c = random.choice(classes)
    p = data_dir / c
    imgs = [f for f in os.listdir(p) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]
    if imgs:
        f = random.choice(imgs)
        img_path = p / f
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img_r, cv2.COLOR_RGB2HSV)
        m1 = cv2.inRange(hsv, np.array([20,40,20]), np.array([80,255,255]))
        m2 = cv2.inRange(hsv, np.array([30,10,40]), np.array([90,255,255]))
        m3 = cv2.inRange(hsv, np.array([14,30,80]), np.array([35,255,255]))
        mask = cv2.morphologyEx(cv2.bitwise_or(cv2.bitwise_or(m1,m2),m3), cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        mask = cv2.medianBlur(mask, 7)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            primary_mask = np.where(labels==idx, 255, 0).astype(np.uint8)
        else:
            primary_mask = mask
        overlay = cv2.addWeighted(cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR), 0.7, cv2.applyColorMap(cv2.normalize(primary_mask, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET), 0.3, 0)
        col = st.columns(2)
        col[0].image(img_r, caption="Original (redimensionada)", width=300)
        col[1].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Segmentación hoja primaria", width=300)
st.write("""
El preprocesamiento estandariza la resolución a 224×224 y aplica segmentación para aislar la hoja primaria. Se usan umbrales en espacio HSV para colores verde/amarillo, limpieza morfológica y componentes conectados; se selecciona el componente mayor como la hoja principal. Esto reduce la influencia de fondos y hojas secundarias antes de extraer características.
""")

st.subheader("3. Extracción de features")
features_primary = root / "data" / "3_data_extract_features_primary" / "features_dataset.csv"
features_orig = root / "data" / "3_data_extract_features" / "features_dataset.csv"
csv_path = features_primary if features_primary.exists() else features_orig if features_orig.exists() else None
if csv_path:
    df = pd.read_csv(csv_path)
    st.write(df.head())
    num_cols = [c for c in df.columns if c != "label"]
    sample_cols = num_cols[:8]
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(df[sample_cols], ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig2)
    fig3, ax3 = plt.subplots(figsize=(6,5))
    corr = df[num_cols[:20]].corr()
    sns.heatmap(corr, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)
st.write("""
Las características se agrupan en color (estadísticos por BGR/HSV, histogramas, colorfulness), textura (LBP multi‑escala, GLCM en varias distancias/ángulos, Gabor, HOG), forma (área, perímetro, circularidad, solidez, momentos de Hu), venación (longitud de esqueleto y ratio) y frecuencia (energía DCT). Este conjunto describe patrones cromáticos, de textura y morfológicos relevantes para el diagnóstico.
""")

st.subheader("4. Reducción PCA")
if csv_path:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values if "label" in df.columns else df.values
    y = df["label"].values if "label" in df.columns else None
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    fig4, ax4 = plt.subplots(figsize=(8,4))
    ax4.bar(range(1, len(evr)+1), evr)
    ax4.plot(range(1, len(evr)+1), np.cumsum(evr), marker='o', color='red')
    ax4.set_xlabel("Componente")
    ax4.set_ylabel("Varianza explicada")
    st.pyplot(fig4)
    pc1 = X_pca[:,0]
    pc2 = X_pca[:,1] if X_pca.shape[1] > 1 else np.zeros_like(pc1)
    fig5, ax5 = plt.subplots(figsize=(6,5))
    if y is not None:
        labels = pd.Series(y).astype(str)
        for cls in sorted(labels.unique()):
            m = labels == cls
            ax5.scatter(pc1[m], pc2[m], s=12, label=str(cls), alpha=0.7)
        ax5.legend()
    else:
        ax5.scatter(pc1, pc2, s=12, alpha=0.7)
    ax5.set_xlabel("PC1")
    ax5.set_ylabel("PC2")
    ax5.set_title("Proyección PCA (PC1 vs PC2)")
    st.pyplot(fig5)
st.write("""
PCA reduce la dimensionalidad manteniendo al menos el 95% de la varianza explicada. Esto ayuda a visualizar la separabilidad entre clases (PC1 vs PC2) y puede servir para acelerar modelos clásicos o detectar redundancias en las características.
""")

st.subheader("5. Entrenamiento y evaluación")
st.markdown("### SVM (RBF)")
st.write("""
Entrada: vectores de features numéricos. Preprocesamiento: imputación mediana y escalado estándar.
Hiperparámetros: C y gamma con validación estratificada (k-fold).
Evaluación: accuracy y F1 macro; matriz de confusión para interpretar confusiones entre clases.
Regularización: margen máximo del SVM controlado por C y suavidad del kernel por gamma.
""")
st.markdown("### MLP (Perceptrón Multicapa)")
st.write("""
Entrada: vectores de features normalizados. Arquitectura típica con una o más capas ocultas (por ejemplo 2–3 capas con activación ReLU) y capa de salida softmax para clasificación multiclase.
Hiperparámetros: número de neuronas por capa, tasa de aprendizaje y epochs; se ajustan con validación.
Evaluación: accuracy y F1 macro; curva de aprendizaje para monitorizar pérdida/precisión.
Regularización: L2 (weight decay) y Dropout en capas ocultas; early stopping sobre validación para evitar sobreajuste.
""")
st.markdown("### Random Forest")
st.write("""
Entrada: features crudos o escalados. Ensamble de árboles con votación.
Hiperparámetros: número de árboles, profundidad máxima, mínimo de muestras por hoja.
Evaluación: accuracy y F1 macro; importancia de variables para interpretación.
Regularización: limitación de profundidad y tamaños mínimos por división; bagging reduce varianza.
""")
st.markdown("### KNN")
st.write("""
Entrada: features escalados. Clasificación basada en vecinos más cercanos.
Hiperparámetros: número de vecinos y ponderación por distancia.
Evaluación: accuracy y F1 macro.
Regularización: elección de k adecuado y ponderación por distancia para reducir sensibilidad al ruido.
""")
st.markdown("### CNN (MobileNetV2 + cabeza densa)")
st.write("""
Entrada: imágenes 224×224 con rescale 1/255. Aumentación: rotación, desplazamiento, zoom, shear, flips, brillo y cambios de canal.
Arquitectura: MobileNetV2 preentrenado sin top, pooling global, capa densa de 512 ReLU y capa de salida.
Entrenamiento: fine‑tuning parcial del backbone y entrenamiento del bloque superior; validación interna (split de 20%).
Evaluación: precisión en validación, curvas de aprendizaje; distribución de probabilidades por clase.
Regularización: data augmentation, congelación parcial de capas y tasa de aprendizaje pequeña.
""")
st.markdown("### Protocolo de evaluación")
st.write("""
Validación estratificada para modelos clásicos, métricas macro por clase, matriz de confusión.
Para la CNN, división de entrenamiento/validación y seguimiento de pérdida y precisión, con análisis cualitativo por imagen.
""")

st.subheader("6. Flujo de trabajo")
st.write("""
1. Recolección de imágenes y organización por clase.
2. Redimensionamiento y segmentación de hoja primaria.
3. Extracción de features de color, textura, forma, venación y frecuencia.
4. Reducción de dimensionalidad (PCA) y preparación de datasets.
5. Entrenamiento de modelos clásicos y CNN, evaluación y comparación.
6. Despliegue en la aplicación para análisis y visualización.
""")

st.subheader("7. Resultados y evidencias")
imgs_dir = root / "app" / "img"
comp_path = imgs_dir / "Compración.jpeg"
svm_cm_path = imgs_dir / "MLP.jpeg"
cnn_loss_path = imgs_dir / "Loss.jpeg"
if comp_path.exists():
    st.image(str(comp_path), caption="Resultados comparativos de modelos", use_column_width=True)
if svm_cm_path.exists():
    st.image(str(svm_cm_path), caption="Matriz de confusión del modelo SVM", use_column_width=True)
if cnn_loss_path.exists():
    st.image(str(cnn_loss_path), caption="Curva de aprendizaje del CNN (Loss)", use_column_width=True)
