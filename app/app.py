import streamlit as st
import time

# ============================================================
# ---------------------- ESTILOS ------------------------------
# ============================================================

import os
from pathlib import Path

_here = Path(__file__).resolve()
_root = _here.parent
_style_candidates = [
    _root / "styles.css",
    Path("styles.css")
]
for _p in _style_candidates:
    if _p.exists():
        st.markdown(f"<style>{_p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
        break

# ============================================================
# --------------------- TÍTULO PRINCIPAL ----------------------
# ============================================================

st.title("🥔🌿 Plataforma de Diagnóstico de Enfermedades en Papas")

st.markdown("""
Bienvenido a una plataforma creada para unir **agricultura**, **tecnología** y **machine learning**  
en un mismo espacio accesible, visual y fácil de explorar.

Aquí no solo podrás analizar enfermedades en hojas de papa; también podrás conocer datos curiosos,
entender cómo funciona la detección visual, aprender sobre los modelos involucrados y navegar entre
múltiples herramientas diseñadas para que descubras más en cada sección.

El menú lateral te permitirá desplazarte por todo el proyecto, pero antes, déjame contarte por qué
esta plataforma existe y qué puedes aprender dentro.
""")

# ============================================================
# -------------------------- SECCIÓN 1 ------------------------
# ============================================================

st.markdown('<div class="home-card">', unsafe_allow_html=True)
st.markdown("""
## ¿Cuál es la idea principal del proyecto?

La agricultura moderna necesita herramientas rápidas, confiables y accesibles para identificar
enfermedades en cultivos antes de que se propaguen. Este proyecto busca precisamente eso:
permitirte **subir una fotografía** de una hoja de papa y recibir un **diagnóstico automático**
basado en patrones visuales.

Modelos como los que detectan *Late Blight* o *Early Blight* se basan en señales que los humanos no
siempre perciben de inmediato: cambios en la textura, irregularidades en el color, bordes
desgastados, zonas inusualmente brillantes o sombreadas.

¿Te gustaría descubrir cómo una red neuronal detecta detalles invisibles para el ojo humano?
Aquí podrás explorarlo.

Este espacio está pensado tanto para estudiantes y desarrolladores como para agricultores que deseen
entender mejor lo que ocurre en sus cultivos desde una perspectiva científica y visual.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ------------------ SECCIÓN 2 (ROTADOR) ----------------------
# ============================================================

st.markdown('<div class="home-card">', unsafe_allow_html=True)
st.markdown("## Datos curiosos sobre la papa y sus enfermedades")

st.markdown("""
Las papas no solo son un alimento clave; representan un ecosistema frágil.
Aquí verás datos curiosos que cambiarán automáticamente cada pocos segundos:
""")

datos = [
    "La papa es la cuarta cosecha más importante del mundo.",
    "Existen más de 4,000 variedades de papa registradas oficialmente.",
    "El 'Late Blight' provocó la hambruna irlandesa del siglo XIX.",
    "Las enfermedades suelen aparecer primero en las hojas.",
    "La detección temprana puede salvar hasta 40% de la cosecha.",
    "Las hojas alteran su química antes de mostrar síntomas visibles."
]

# ---- Estado inicial ----
if "idx" not in st.session_state:
    st.session_state.idx = 0

# ---- Contenedor del texto ----
rotador = st.empty()

# ---- Mostrar el mensaje actual ----
rotador.markdown(
    f"""
    <div style="font-size:20px; font-weight:600; padding:12px 16px;
                border-left:6px solid #2b8a3e; background:#e9f7ef;
                border-radius:8px; margin-top:10px;">
        {datos[st.session_state.idx]}
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Control de tiempo ----
if "last_change" not in st.session_state:
    st.session_state.last_change = time.time()

# ---- Cambiar cada 5s sin recargar la página ----
if time.time() - st.session_state.last_change >= 5:
    st.session_state.idx = (st.session_state.idx + 1) % len(datos)
    st.session_state.last_change = time.time()
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# -------------------------- SECCIÓN 3 ------------------------
# ============================================================

st.markdown('<div class="home-card">', unsafe_allow_html=True)
st.markdown("""
## ¿Cómo funciona esta plataforma?

El proceso es simple y diseñado para cualquier usuario, incluso sin conocimientos técnicos:

1. En el menú lateral, selecciona **Images**.  
2. Sube una fotografía de la hoja de papa.  
3. La imagen se procesa, limpia y prepara para el modelo.  
4. Recibes un **diagnóstico preliminar** basado en patrones visuales detectados.

¿Quieres ver cómo se entrena una red neuronal?  
¿Te interesa comparar distintos algoritmos de clasificación?  
¿O entender cómo la visión artificial interpreta daños en una hoja?  

Las secciones de análisis y visualizaciones están hechas para explorar todo eso.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# -------------------------- SECCIÓN 4 ------------------------
# ============================================================

st.markdown('<div class="home-card">', unsafe_allow_html=True)
st.markdown("""
## ¿Por qué es útil este proyecto?

Las enfermedades agrícolas son un problema silencioso que avanza sin ser visto. Detectarlas a tiempo
puede marcar la diferencia entre conservar una cosecha o perderla por completo.

Esta plataforma ofrece:

- Diagnóstico rápido sin necesidad de un experto.
- Comparación visual entre múltiples enfermedades.
- Un entorno ideal para practicar clasificación e interpretación de imágenes.
- Posibilidad de integrar modelos reales de predicción en versiones futuras.

La pregunta es:  
**¿Qué tanto puede la inteligencia artificial ayudar a prevenir pérdidas agrícolas?**  
Explora este proyecto y empieza a descubrirlo.
""")
st.markdown('</div>', unsafe_allow_html=True)
