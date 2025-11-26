import streamlit as st
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

try:
    from tensorflow.keras.models import load_model as tf_load_model
except Exception:
    tf_load_model = None

# Cargar CSS externo
_here = Path(__file__).resolve()
_root = _here.parents[2]
_style_candidates = [
    _root / "app" / "styles.css",
    _here.parent.parent / "styles.css",
    Path("styles.css")
]
for _p in _style_candidates:
    if _p.exists():
        st.markdown(f"<style>{_p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
        break

st.title("🌿 Análisis de hojas de papa")
st.write("""
Sube una imagen de una hoja de papa para procesarla y analizar su estado.  
Actualmente, el modelo de diagnóstico aún no está integrado, pero puedes explorar información básica y análisis preliminares de la imagen.
""")

uploaded = st.file_uploader("Sube tu imagen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Mostrar la imagen
    image = Image.open(uploaded)
    st.image(image, caption="Imagen cargada", width=350)
    
    st.success("✅ La imagen se cargó correctamente")
    
    # Información preliminar de la imagen
    st.subheader("📊 Información de la imagen")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Formato:** {image.format}")
    col1.markdown(f"**Modo de color:** {image.mode}")
    col2.markdown(f"**Tamaño:** {image.size[0]} px ancho x {image.size[1]} px alto")
    
    st.markdown("---")
    
    # Análisis preliminar: histograma de colores
    st.subheader("🔍 Análisis preliminar de color")
    img_array = np.array(image)
    
    if img_array.ndim == 3:
        # Solo RGB
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_ratio = np.sum(g > r) / total_pixels * 100
        yellow_ratio = np.sum((r>100) & (g>100) & (b<100)) / total_pixels * 100
    else:
        green_ratio = yellow_ratio = 0
    
    st.write(f"- Aproximadamente {green_ratio:.1f}% de la hoja es verde (saludable).")
    st.write(f"- Aproximadamente {yellow_ratio:.1f}% de la hoja muestra amarillamiento (posible afectación).")
    
    # Mostrar histograma de colores
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(img_array.ravel(), bins=256, color='gray', alpha=0.7)
    ax.set_title("Histograma de intensidad de la imagen")
    ax.set_xlabel("Valor de píxel")
    ax.set_ylabel("Cantidad de píxeles")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("🧠 Diagnóstico preliminar (CNN)")
    model_path = _root / "models" / "potato_leaf_cnn.h5"
    if tf_load_model and model_path.exists():
        try:
            @st.cache_resource
            def _load_cnn(path):
                return tf_load_model(str(path))

            def _preprocess(img: Image.Image):
                img = img.convert("RGB")
                img = img.resize((224, 224))
                x = np.asarray(img, dtype=np.float32) / 255.0
                x = np.expand_dims(x, axis=0)
                return x

            model = _load_cnn(model_path)
            x = _preprocess(image)
            pred = model.predict(x)
            probs = np.squeeze(pred)

            classes_dir = _root / "data" / "2_data_resize"
            if classes_dir.exists():
                class_names = sorted([d.name for d in classes_dir.iterdir() if d.is_dir()])
            else:
                class_names = [f"Clase {i}" for i in range(len(probs))]

            top_idx = int(np.argmax(probs))
            st.success(f"Predicción: {class_names[top_idx]} ({probs[top_idx]*100:.1f}%)")
            st.write("Probabilidades:")
            for i, p in enumerate(probs):
                st.write(f"- {class_names[i]}: {p*100:.1f}%")
        except Exception as e:
            st.warning("No se pudo ejecutar la inferencia del modelo.")
    else:
        st.info("Modelo no disponible o TensorFlow no instalado.")

    st.subheader("📌 Siguiente pasos")
    st.write("""
    - Integrar el modelo de clasificación para detectar enfermedades específicas en hojas de papa.
    - Visualización de porcentaje de afectación y zonas afectadas (heatmaps).
    - Registro de historial de imágenes para seguimiento de evolución.
    - Comparación automática entre hojas sanas y afectadas.
    """)
    
    st.info("💡 Consejos: sube imágenes claras, con buena iluminación y enfoque en la hoja principal para obtener resultados óptimos cuando se integre el modelo.")
    
else:
    st.info("Espera a subir una imagen para comenzar el análisis.")
