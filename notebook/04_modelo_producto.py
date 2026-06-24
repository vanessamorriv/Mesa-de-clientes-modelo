import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
import os
import gc
from sklearn.preprocessing import LabelEncoder

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
MODELS    = os.path.join(BASE, 'models') + os.sep
os.makedirs(MODELS, exist_ok=True)

PRODUCTOS_VALIDOS = ['SPOT', 'FORWARD', 'FIX', 'NEXT DAY', 'OTROS']

def agrupar_producto(p):
    p = str(p).strip().upper()
    if p in {'OPCIONES', 'SWAPS'}:
        return 'OTROS'
    return p if p in PRODUCTOS_VALIDOS else 'OTROS'

features = [
    'dias_desde_ultima_op', 'freq_30d', 'freq_90d', 'monto_prom_90d',
    'CIIU_BUC', 'Segmento', 'Subsegmento',
    'dia_semana', 'dia_mes', 'mes',
    'TRM_USD', 'variacion_trm_7d',
    'hist_90d_SPOT', 'hist_90d_FORWARD', 'hist_90d_FIX',
    'hist_90d_NEXT_DAY', 'hist_90d_OTROS'
]
cat_features = ['CIIU_BUC', 'Segmento', 'Subsegmento']

PARAMS_LGB = {
    'objective'       : 'multiclass',
    'num_class'       : 5,
    'learning_rate'   : 0.05,
    'max_bin'         : 63,
    'num_leaves'      : 31,
    'min_data_in_leaf': 20,
    'force_col_wise'  : True,
    'verbose'         : -1,
    'seed'            : 42,
}

# ── 1. Cargar datos ───────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet',
                     columns=['Fecha', 'NIT', 'Producto', 'Monto_Total_',
                               'CIIU_BUC', 'Segmento', 'Subsegmento', 'TRM_USD'])
df['Fecha'] = pd.to_datetime(df['Fecha']).astype('datetime64[ns]')
df['NIT']   = df['NIT'].astype(str)

trm = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet',
                      columns=['Fecha', 'variacion_trm_7d'])
trm['Fecha'] = pd.to_datetime(trm['Fecha']).astype('datetime64[ns]')
df = df.merge(trm, on='Fecha', how='left')
df['variacion_trm_7d'] = df['variacion_trm_7d'].fillna(0)

# ── 2. Target ─────────────────────────────────────────────────────
df['Producto_grupo'] = df['Producto'].apply(agrupar_producto)
df = df[df['Producto_grupo'].isin(PRODUCTOS_VALIDOS)].copy()

le = LabelEncoder()
df['label'] = le.fit_transform(df['Producto_grupo'])
print(f"Clases: {list(le.classes_)}")
print(f"Registros: {len(df):,}")

# ── 3. Features históricas por grupo con searchsorted ────────────
print("Construyendo features históricas...")
df_sorted = df.sort_values(['NIT', 'Fecha']).reset_index(drop=True)

# Convertir fechas a int64 para operar con numpy
fechas_ns  = df_sorted['Fecha'].values.astype(np.int64)
montos     = df_sorted['Monto_Total_'].values
productos  = df_sorted['Producto_grupo'].values
nits       = df_sorted['NIT'].values

# Límites de cada grupo NIT
_, group_start = np.unique(nits, return_index=True)
group_end = np.append(group_start[1:], len(df_sorted))

D30 = int(pd.Timedelta(days=30).value)
D90 = int(pd.Timedelta(days=90).value)

freq30_arr    = np.zeros(len(df_sorted), dtype=np.int32)
freq90_arr    = np.zeros(len(df_sorted), dtype=np.int32)
monto90_arr   = np.zeros(len(df_sorted), dtype=np.float64)
dias_ult_arr  = np.full(len(df_sorted), 999, dtype=np.int32)
hist_arrs     = {p: np.zeros(len(df_sorted), dtype=np.int32) for p in PRODUCTOS_VALIDOS}

print(f"  Procesando {len(group_start):,} NITs...")
for k, (gs, ge) in enumerate(zip(group_start, group_end)):
    if k % 2000 == 0:
        print(f"    NIT {k:,}/{len(group_start):,}...")

    f = fechas_ns[gs:ge]   # fechas del grupo, ya ordenadas
    m = montos[gs:ge]
    p = productos[gs:ge]
    n = ge - gs

    for i in range(n):
        fi = f[i]

        # Índice donde empieza la ventana 30d y 90d (antes de fi)
        ini30 = np.searchsorted(f[:i], fi - D30, side='left')
        ini90 = np.searchsorted(f[:i], fi - D90, side='left')

        freq30_arr[gs + i] = i - ini30
        freq90_arr[gs + i] = i - ini90

        if i > 0:
            w90 = m[ini90:i]
            monto90_arr[gs + i]  = w90.mean() if len(w90) > 0 else 0
            dias_ult_arr[gs + i] = int((fi - f[i-1]) / int(pd.Timedelta(days=1).value))

        for prod in PRODUCTOS_VALIDOS:
            hist_arrs[prod][gs + i] = (p[ini90:i] == prod).sum()

df_sorted['freq_30d']            = freq30_arr
df_sorted['freq_90d']            = freq90_arr
df_sorted['monto_prom_90d']      = monto90_arr
df_sorted['dias_desde_ultima_op']= dias_ult_arr
for prod in PRODUCTOS_VALIDOS:
    col = f'hist_90d_{prod.replace(" ", "_")}'
    df_sorted[col] = hist_arrs[prod]

df_sorted['dia_semana'] = df_sorted['Fecha'].dt.weekday
df_sorted['dia_mes']    = df_sorted['Fecha'].dt.day
df_sorted['mes']        = df_sorted['Fecha'].dt.month

print(f"  Listo. Shape: {df_sorted.shape}")
gc.collect()

# ── 4. Soft class weights ─────────────────────────────────────────
conteos = df_sorted['label'].value_counts().sort_index()
pesos   = (1 / conteos) / (1 / conteos).sum()
sample_weights = df_sorted['label'].map(pesos).values
print(f"\nDistribución clases:\n{conteos}")

# ── 5. Categoricas ────────────────────────────────────────────────
for col in cat_features:
    df_sorted[col] = df_sorted[col].astype('category')

# ── 6. Entrenar ───────────────────────────────────────────────────
print("\nEntrenando modelo de producto...")
train_set = lgb.Dataset(
    df_sorted[features],
    label=df_sorted['label'],
    categorical_feature=cat_features,
    weight=sample_weights,
    free_raw_data=True
)
modelo = lgb.train(PARAMS_LGB, train_set, num_boost_round=300)

del train_set
gc.collect()

# ── 7. Guardar ────────────────────────────────────────────────────
print("\nGuardando modelo...")
with open(MODELS + 'modelo_producto_v4.pkl', 'wb') as f:
    pickle.dump(modelo, f)

with open(MODELS + 'label_encoder_producto.pkl', 'wb') as f:
    pickle.dump(le, f)

metadata = {
    'features'           : features,
    'cat_features'       : cat_features,
    'clases'             : list(le.classes_),
    'fecha_entrenamiento': str(df_sorted['Fecha'].max().date()),
    'n_registros'        : len(df_sorted),
}
with open(MODELS + 'modelo_producto_v4_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nGuardado: modelo_producto_v4.pkl")
print(f"Guardado: label_encoder_producto.pkl")
print(f"Guardado: modelo_producto_v4_metadata.json")
print(f"Fecha máxima: {df_sorted['Fecha'].max().date()}")
print(f"Registros: {len(df_sorted):,}")