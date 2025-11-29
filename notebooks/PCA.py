import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
primary_path = ROOT / "data/3_data_extract_features_primary/features_dataset.csv"
orig_path = ROOT / "data/3_data_extract_features/features_dataset.csv"
if primary_path.exists():
    data_path = primary_path
    out_path = ROOT / "data/3_data_extract_features_primary/features_dataset_pca.csv"
elif orig_path.exists():
    data_path = orig_path
    out_path = ROOT / "data/3_data_extract_features/features_dataset_pca.csv"
else:
    raise FileNotFoundError("No existe el CSV de features: primary ni original")
df = pd.read_csv(data_path)

if 'label' in df.columns:
    X = df.drop(columns=['label'])
    y = df['label']
else:
    X = df.copy()
    y = None

X = X.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=cols)
if y is not None:
    pca_df['label'] = y.values

print('Componentes:', pca.n_components_)
print('Varianza explicada total:', pca.explained_variance_ratio_.sum())
out_path.parent.mkdir(parents=True, exist_ok=True)
pca_df.to_csv(out_path, index=False)
print('Guardado en:', out_path)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
plt.figure(figsize=(8,4))
plt.bar(range(1, len(evr)+1), evr, alpha=0.7)
plt.plot(range(1, len(evr)+1), cum, marker='o', color='red')
plt.xlabel('Componente')
plt.ylabel('Varianza explicada')
plt.title('Scree plot y varianza acumulada')
plt.tight_layout()
plt.show()
pc1 = X_pca[:,0]
pc2 = X_pca[:,1] if X_pca.shape[1] > 1 else np.zeros_like(pc1)
plt.figure(figsize=(6,5))
if y is not None:
    classes = pd.Series(y).astype(str)
    for cls in sorted(classes.unique()):
        m = classes == cls
        plt.scatter(pc1[m], pc2[m], s=15, label=str(cls), alpha=0.7)
    plt.legend()
else:
    plt.scatter(pc1, pc2, s=15, alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Proyección PCA (PC1 vs PC2)')
plt.tight_layout()
plt.show()
