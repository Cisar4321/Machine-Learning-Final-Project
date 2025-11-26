## Visión General
- Proyecto de clasificación/diagnóstico de enfermedades en hojas de papa con interfaz en `streamlit` y pipeline de preparación, extracción de características y entrenamiento.
- App multipágina para exploración del dataset y análisis básico de imágenes; el modelo aún no está integrado en la UI.

## Estructura del Repositorio
- `app/`: interfaz Streamlit.
  - `app/app.py` (landing y estilos; `app/app.py:15` título).
  - `app/pages/02_enfermedades.py` (explora `data/1_data_original`; carrusel; gráficas; `app/pages/02_enfermedades.py:13-21`).
  - `app/pages/03_visualizaciones.py` (plantilla de visualizaciones).
  - `app/pages/04_analysis.py` (uploader y análisis preliminar; `app/pages/04_analysis.py:17-23`).
  - `app/styles.css` (paleta y UI).
- `notebooks/`: pipeline end-to-end.
  - `01_resize.ipynb` (redimensionado y segmentación; genera `data/2_data_resize`).
  - `02_extract_features.ipynb` (features color/textura/forma; guarda CSV en `data/3_data_extract_features`; `02_extract_features.ipynb:56-58`).
  - `03_train_classical_models.ipynb` (modelos clásicos; guarda artefactos `.pkl`).
  - `04_train_cnn.ipynb` (CNN; guarda `models/potato_leaf_cnn.h5`).
  - `05_pre_cluster.ipynb` (clustering/curación; `data/4_features_embeddings`).
- `data/`:
  - `1_data_original/` (fuente cruda por clases).
  - `2_data_resize/` (derivados; usado por `src/cor.py:8`).
  - `3_data_extract_features/` (CSV de features).
  - `4_features_embeddings/` (embeddings, figuras y sospechosas).
- `models/`: artefactos entrenados (`*.pkl`, `*.h5`).
- `src/cor.py` (correlación entre clases con promedios de imágenes).
- `requirements.txt` (deps base ML/visión; `requirements.txt:1-7`).
- `.gitignore` (excluye entornos, caches y artefactos ML/derivados).

## Dependencias y Ejecución
- Dependencias en `requirements.txt`: `torch`, `torchvision`, `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `opencv-python`.
- Faltan en `requirements.txt` para la app: `streamlit`, `seaborn`, `Pillow`.
- Ejecución recomendada: `cd app` y `streamlit run app.py` (por uso de `open("styles.css")` con ruta relativa).

## Observaciones Clave
- La app aún no realiza inferencia con los modelos de `models/`.
- Los notebooks producen artefactos reproducibles y pesados; `.gitignore` los excluye correctamente.
- No hay `README` con instrucciones ni pruebas automatizadas.

## Próximos Pasos Propuestos
1. Completar `requirements.txt` con `streamlit`, `seaborn`, `Pillow` y fijar versiones mínimas.
2. Integrar inferencia en `app/pages/04_analysis.py`:
   - Cargar `models/best_model.pkl`, `scaler.pkl`, `label_encoder.pkl` y/o `potato_leaf_cnn.h5`.
   - Implementar preprocesamiento (resize/normalización) consistente con entrenamiento.
   - Mostrar predicción y probas, y preparar lugar para heatmaps (Grad-CAM si se usa la CNN).
3. Unificar rutas relativas/absolutas para evitar errores de CWD (helpers en `src/` para resolver paths).
4. Extraer funciones de notebooks a módulos `src/` reutilizables (preprocesamiento, feature extraction, inferencia), manteniendo notebooks como demostraciones.
5. Añadir un `README` con instalación, ejecución, y flujo de trabajo, y una guía breve para reproducir resultados.
6. Opcional: añadir tests básicos (forma de features, carga de modelos) y caching en Streamlit para evitar recomputación.

¿Confirmas que procedamos con estos pasos y la integración del modelo en la app?