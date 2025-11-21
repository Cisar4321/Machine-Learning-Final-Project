import streamlit as st
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

# Cargar CSS externo
with open("styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
