## Alcance
- Implementar segmentación para aislar la hoja primaria y generar imágenes derivadas en una carpeta nueva.
- Ampliar el conjunto de features (color, textura, forma, frecuencia, bordes/venación).
- Preparar notebooks/módulos listos para ejecutar, sin correr nada.

## Carpetas Nuevas
- `data/2_data_primary_leaf/` (derivados segmentados: `*_primary.png`, `*_mask.png`, `*_overlay.png`).
- `data/3_data_extract_features_primary/` (CSV con features extendidos).

## Segmentación Hoja Primaria
- Basado en tu pipeline actual (`01_resize.ipynb:85-93`, `164-190`):
  1) Preprocesado: resize a `224x224`, blur ligero, conversión a HSV.
  2) Máscara inicial: umbrales adaptativos en HSV (verde/amarillo) o clustering k‑means sobre HSV para separar hoja del fondo.
  3) Limpieza morfológica: `close` + `median blur`, como en `01_resize`.
  4) Componentes conectados: elegir el componente de mayor área como hoja primaria; descartar el resto.
  5) Refinamiento opcional: `GrabCut` con la máscara grande como semilla; recorte de bounding box ajustado.
  6) Outputs: imagen segmentada RGB, máscara binaria y overlay.

## Features Ampliados
- Color:
  - Estadísticos: media/mediana/IQR/MAD por canales BGR/HSV.
  - Histogramas con más bins (32) y uniformización por máscara.
  - Color moments (1ª–3ª orden) y métricas: colorfulness, ratios RG/GB/RB (ya tienes), saturación media.
- Textura:
  - LBP multi‑escala (R=1,2,3 ya; añadir `LBP var` y uniformidad).
  - GLCM extendido: promediar props (`contrast`, `homogeneity`, `energy`, `correlation`, `dissimilarity`, `ASM`) en todas distancias/ángulos (ya configurado en `02_extract_features.ipynb:111-115`).
  - Gabor bank (frecuencias y orientaciones) — medias/std de respuestas.
  - Laws texture energy y Tamura (coarseness, contrast, directionality).
- Forma:
  - Eccentricity, convex hull area/perimeter, convexity, compactness, perimeter/area, min enclosing circle.
  - Hu moments (7 ya), Zernike (si librería disponible), fractal dimension (box‑counting).
  - Curvatura de contorno: medias/std de kappa y número de picos.
- Frecuencia:
  - DCT: energía baja vs alta frecuencia, centro de masa espectral.
  - FFT PSD: momentos centrales y anisotropía direccional.
- Bordes/Venación:
  - Densidad de bordes Canny, HOG resumen.
  - Venación: skeletonization, longitud total/porcentaje sobre el área, número de intersecciones.

## Integración en Notebooks/Módulos
- Nuevo notebook `notebooks/01a_primary_leaf_segmentation.ipynb`:
  - Reutiliza y mejora la segmentación (hoja primaria). Guarda outputs en `data/2_data_primary_leaf/`.
- Nuevo notebook `notebooks/02_extract_features_primary.ipynb`:
  - Lee `data/2_data_primary_leaf/` y calcula features extendidos. Exporta a `data/3_data_extract_features_primary/features_dataset.csv`.
- Módulos reutilizables en `src/`:
  - `src/segmentation.py`: funciones `segment_primary_leaf(image)` y utilidades de máscara.
  - `src/features.py`: `extract_features_extended(image, mask)` (organizar por bloques y devolver vector + nombres de columnas).
- Mantener notebooks existentes como referencia; no se ejecuta nada.

## Validación (sin ejecutar)
- Notebook con una celda de verificación que:
  - Lista conteos por clase antes/después (esperado: 1 máscara principal por imagen).
  - Muestra `head()` del CSV de features extendidos (cuando se ejecute).

## Higiene y Derivados
- Añadir a `.gitignore` (cuando se confirme): `data/2_data_primary_leaf/` y `data/3_data_extract_features_primary/`.

## Consideraciones de Rendimiento
- Cachear funciones costosas si se integra en app.
- Escalar/estandarizar features y documentar dimensiones.

¿Confirmas que proceda con la creación de los notebooks y módulos propuestos (sin ejecutar) y la preparación de rutas/estructuras para que puedas probarlos cuando quieras?