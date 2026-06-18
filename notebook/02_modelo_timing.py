import pandas as pd
import numpy as np
import pickle
import json
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
MODELS    = os.path.join(BASE, 'models') + os.sep
os.makedirs(MODELS, exist_ok=True)

# ── 1. Cargar datos ───────────────────────────────────────────────
print("Cargando datos...")
cols_necesarias = ['NIT', 'Fecha', 'Monto_Total_', 'Segmento', 'Subsegmento', 'CIIU_BUC']
df = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet', columns=cols_necesarias)
trm_usd_diario = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet')
df_sorted      = df.sort_values(['NIT', 'Fecha'])

# ── 2. Construir panel semanal ────────────────────────────────────
print("Construyendo panel...")
fecha_min = df['Fecha'].min()
fecha_max = df['Fecha'].max()

fechas_corte_global = pd.date_range(
    fecha_min, fecha_max - pd.Timedelta(days=7), freq='W-MON'
).values
n_fechas = len(fechas_corte_global)

resultados = []
for nit, grupo in df_sorted.groupby('NIT'):
    fechas_ops = grupo['Fecha'].values.astype('datetime64[ns]')
    montos     = grupo['Monto_Total_'].values

    idx_hasta_corte = np.searchsorted(fechas_ops, fechas_corte_global, side='right')
    idx_30          = np.searchsorted(fechas_ops, fechas_corte_global - np.timedelta64(30, 'D'), side='right')
    idx_90          = np.searchsorted(fechas_ops, fechas_corte_global - np.timedelta64(90, 'D'), side='right')
    idx_label       = np.searchsorted(fechas_ops, fechas_corte_global + np.timedelta64(7, 'D'),  side='right')

    dias_desde_ultima = np.full(n_fechas, np.nan)
    tiene_historia    = idx_hasta_corte > 0
    idx_prev          = np.clip(idx_hasta_corte - 1, 0, None)
    ultimas_fechas    = fechas_ops[idx_prev]
    dias_desde_ultima[tiene_historia] = (
        (fechas_corte_global[tiene_historia] - ultimas_fechas[tiene_historia])
        / np.timedelta64(1, 'D')
    )

    freq_30d = idx_hasta_corte - idx_30
    freq_90d = idx_hasta_corte - idx_90

    cum_montos   = np.concatenate([[0], np.cumsum(montos)])
    suma_90      = cum_montos[idx_hasta_corte] - cum_montos[idx_90]
    monto_prom_90d = np.divide(suma_90, freq_90d,
                                out=np.full(n_fechas, np.nan),
                                where=freq_90d > 0)

    label = (idx_label > idx_hasta_corte).astype(int)

    resultados.append(pd.DataFrame({
        'NIT':                nit,
        'fecha_corte':        fechas_corte_global,
        'dias_desde_ultima_op': dias_desde_ultima,
        'freq_30d':           freq_30d,
        'freq_90d':           freq_90d,
        'monto_prom_90d':     monto_prom_90d,
        'label':              label,
    }))

panel = pd.concat(resultados, ignore_index=True)

# ── 3. Agregar variables estáticas y calendario ───────────────────
clientes_estatico = (df[['NIT', 'Segmento', 'Subsegmento', 'CIIU_BUC']]
                     .drop_duplicates(subset='NIT'))
panel = panel.merge(clientes_estatico, on='NIT', how='left')
panel['dia_mes']    = panel['fecha_corte'].dt.day
panel['mes']        = panel['fecha_corte'].dt.month
panel['semana_año'] = panel['fecha_corte'].dt.strftime('%V').astype(int)

# ── 4. Guardar panel para que lo use el script de producto ────────
panel.to_parquet(PROCESSED + 'panel_semanal.parquet', index=False)
print(f" Panel guardado: {panel.shape}")

# ── 5. Reducir memoria antes del split ───────────────────────────
# Convertir columnas numéricas a tipos más livianos
for col in ['freq_30d', 'freq_90d', 'dia_mes', 'mes', 'semana_año', 'label']:
    panel[col] = panel[col].astype('int16')
panel['dias_desde_ultima_op'] = panel['dias_desde_ultima_op'].astype('float32')
panel['monto_prom_90d']       = panel['monto_prom_90d'].astype('float32')

print(f"Memoria panel: {panel.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── 6. Train / Test split SIN .copy() ────────────────────────────
FECHA_CORTE_TRAIN = pd.Timestamp('2025-01-01')
mask_train = panel['fecha_corte'] < FECHA_CORTE_TRAIN

features     = ['dias_desde_ultima_op', 'freq_30d', 'freq_90d', 'monto_prom_90d',
                'CIIU_BUC', 'Segmento', 'Subsegmento', 'dia_mes', 'mes', 'semana_año']
cat_features = ['CIIU_BUC', 'Segmento', 'Subsegmento']

# Extraer directamente X e y sin guardar train/test completos
X_train = panel.loc[mask_train,  features].copy()
y_train = panel.loc[mask_train,  'label']
X_test  = panel.loc[~mask_train, features].copy()
y_test  = panel.loc[~mask_train, 'label']

for col in cat_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col]  = X_test[col].astype('category')

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
# ── 7. Entrenar ───────────────────────────────────────────────────
print("Entrenando modelo timing...")

# Liberar memoria antes de entrenar
import gc
del panel
gc.collect()

modelo = lgb.LGBMClassifier(
    objective='binary',
    is_unbalance=True,
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    # Parámetros para reducir uso de RAM
    max_bin=63,          # por defecto 255 — reduce memoria ~4x
    num_leaves=31,       # por defecto 31, dejarlo así
    min_data_in_leaf=50, # evita hojas muy pequeñas
    force_col_wise=True  # más eficiente en memoria que row-wise
)

modelo.fit(X_train, y_train, categorical_feature=cat_features)
# ── 8. Evaluar ─────────────────────────────────────────────────
print("\nEvaluando modelo timing...")
proba_test = modelo.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
print(f"AUC: {auc:.4f}")

# ── 9. Guardar modelo y metadata ──────────────────────────────
with open(MODELS + 'modelo_timing_v1.pkl', 'wb') as f:
    pickle.dump(modelo, f)

metadata = {
    'features': features,
    'cat_features': cat_features,
    'fecha_corte_train': str(FECHA_CORTE_TRAIN.date()),
    'auc_test': round(auc, 4),
    'n_train': len(X_train),
    'n_test': len(X_test),
}
with open(MODELS + 'modelo_timing_v1_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n Guardado: {MODELS}modelo_timing_v1.pkl")
print(f" Guardado: {MODELS}modelo_timing_v1_metadata.json")