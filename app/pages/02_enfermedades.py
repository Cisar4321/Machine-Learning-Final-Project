import streamlit as st
import os
import base64
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns

# Cargar CSS externo
with open("styles.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# PATH CORRECTO
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "1_data_original"))

if not os.path.isdir(DATA_DIR):
    st.error(f"No se encontró la carpeta: {DATA_DIR}")
    st.stop()

# LISTA DE ENFERMEDADES
carpetas = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

st.write("Selecciona la enfermedad:")

# CSS para botones tipo card
st.markdown("""
<style>
.button-card {
    width: 120px;
    height: 80px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 12px;
    text-align: center;
    line-height: 1.2em;
    padding: 10px;
    font-weight: bold;
    word-wrap: break-word;
    cursor: pointer;
    transition: 0.3s;
}
.button-card:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Mostrar botones en 2 filas: 4 arriba, 3 abajo
selected = None
cols = st.columns(4)
for i, carpeta in enumerate(carpetas[:4]):
    if cols[i].button(carpeta):
        selected = carpeta

cols = st.columns(3)
for i, carpeta in enumerate(carpetas[4:7]):
    if cols[i].button(carpeta):
        selected = carpeta

if selected is None:
    st.stop()

opcion = selected

# --- Mostrar descripción si existe ---
info_path = os.path.join(DATA_DIR, opcion, "info.txt")
if os.path.exists(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        st.markdown(f"**Descripción de {opcion}:**  \n{f.read()}")

# --- Obtener todas las imágenes del dataset ---
ruta = os.path.join(DATA_DIR, opcion)
imgs = [
    os.path.join(ruta, f) for f in os.listdir(ruta)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
]

# --- Carrusel con máximo 6 imágenes ---
imgs64 = [base64.b64encode(open(i, "rb").read()).decode() for i in imgs[:6]]

html = f"""
<style>
#carousel {{
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: 15px;
  margin-top: 20px;
}}

.track {{
  display: flex;
  width: {len(imgs64)*100}%;
  transition: transform 0.8s ease;
}}

.track img {{
  width: {100/len(imgs64)}%;
  object-fit: cover;
  border-radius: 10px;
}}

.btn {{
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: rgba(0,0,0,0.5);
  color: white;
  border: none;
  padding: 10px 18px;
  cursor: pointer;
  font-size: 20px;
  border-radius: 8px;
  transition: 0.3s;
}}
.btn:hover {{
  background: rgba(0,0,0,0.7);
}}

#prev {{ left: 10px; }}
#next {{ right: 10px; }}
</style>

<div id="carousel">
  <div class="track" id="track">
"""

for img in imgs64:
    html += f'<img src="data:image/jpeg;base64,{img}">'

html += """
  </div>
  <button class="btn" id="prev">◀</button>
  <button class="btn" id="next">▶</button>
</div>

<script>
let index = 0;
const total = """ + str(len(imgs64)) + """;

document.getElementById("prev").onclick = () => move(-1);
document.getElementById("next").onclick = () => move(1);

function move(dir){
  index += dir;
  if(index < 0) index = 0;               
  if(index >= total) index = total - 1;  

  const shift = (100 / total) * index;
  document.getElementById("track").style.transform = 
      "translateX(" + (-shift) + "%)";
}
</script>
"""

st.components.v1.html(html, height=450)

# --- Miniaturas ---
st.write("Miniaturas del dataset:")
cols = st.columns(min(len(imgs), 6))
for i, img_path in enumerate(imgs[:6]):
    cols[i].image(img_path, width=100)

# --- Información completa de cada imagen ---
data = []
for img_path in imgs:
    with Image.open(img_path) as im:
        data.append({
            "Nombre": os.path.basename(img_path),
            "Formato": os.path.splitext(img_path)[1].lower(),
            "Ancho (px)": im.width,
            "Alto (px)": im.height
        })



# --- Gráficas globales por enfermedad (estéticas) ---
sns.set_theme(style="whitegrid")  # estilo limpio

totales = []
for c in carpetas:
    imgs_c = [f for f in os.listdir(os.path.join(DATA_DIR, c)) if f.lower().endswith((".png",".jpg",".jpeg",".webp"))]
    totales.append(len(imgs_c))

# Cantidad de imágenes por enfermedad
fig_tot, ax_tot = plt.subplots(figsize=(8,5))
bars = ax_tot.bar(carpetas, totales, color=sns.color_palette("viridis", len(carpetas)))
ax_tot.set_ylabel("Cantidad de imágenes")
ax_tot.set_title("Cantidad de imágenes por enfermedad", fontsize=14, weight='bold')
ax_tot.set_xticklabels(carpetas, rotation=30, ha='right')
for bar in bars:
    height = bar.get_height()
    ax_tot.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
st.pyplot(fig_tot)

# Porcentaje de imágenes por enfermedad
fig_pct, ax_pct = plt.subplots(figsize=(6,6))
colors = sns.color_palette("viridis", len(carpetas))
wedges, texts, autotexts = ax_pct.pie(
    totales, labels=carpetas, autopct=lambda p: f'{p:.1f}%', startangle=140,
    colors=colors, wedgeprops={'edgecolor':'white','linewidth':1.2}, textprops={'fontsize':10}
)
ax_pct.set_title("Porcentaje de imágenes por enfermedad", fontsize=14, weight='bold')
plt.setp(autotexts, color='white', weight='bold')
st.pyplot(fig_pct)
