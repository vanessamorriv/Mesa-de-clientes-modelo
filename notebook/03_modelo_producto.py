import pandas as pd
import numpy as np
import pickle
import json
import os
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
MODELS    = os.path.join(BASE, 'models') + os.sep
os.makedirs(MODELS, exist_ok=True)

# ── 1. Cargar datos ───────────────────────────────────────────────
print("Cargando datos...")
df             = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet')
panel          = pd.read_parquet(PROCESSED + 'panel_semanal.parquet')
trm_usd_diario = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet')

# ── 1b. Filtrar solo operaciones en USD ───────────────────────────
df = df[df['Moneda'].str.contains('USD', na=False)].copy()
print(f"Operaciones en USD: {df.shape[0]}")

# ── 2. Agrupar productos ──────────────────────────────────────────
def agrupar_producto(p):
    return 'OTROS' if p in ('OPCIONES', 'SWAPS') else p

ops = df[['NIT', 'Fecha', 'Producto']].sort_values(['NIT', 'Fecha'])
ops['Producto_grupo'] = ops['Producto'].apply(agrupar_producto)

# ── 3. Construir target: producto de la siguiente operación ───────
print("Construyendo target de producto...")
panel_pos = panel[panel['label'] == 1].copy().sort_values(['NIT', 'fecha_corte'])

resultados_producto = []
for nit, grp_panel in panel_pos.groupby('NIT'):
    grp_ops = ops[ops['NIT'] == nit]
    for _, row in grp_panel.iterrows():
        ops_ventana = grp_ops[
            (grp_ops['Fecha'] > row['fecha_corte']) &
            (grp_ops['Fecha'] <= row['fecha_corte'] + pd.Timedelta(days=7))
        ]
        prod = ops_ventana.iloc[0]['Producto_grupo'] if len(ops_ventana) > 0 else None
        resultados_producto.append(prod)

panel_pos['producto_siguiente'] = resultados_producto
panel_producto = panel_pos[panel_pos['producto_siguiente'].notna()].copy()
print(f" Panel producto: {panel_producto.shape}")
print(panel_producto['producto_siguiente'].value_counts())

# ── 4. Función vectorizada para contar historial 90d ─────────────
productos = ['SPOT', 'FORWARD', 'NEXT DAY', 'FIX', 'OTROS']

def calcular_historial_90d(panel_df, ops_hist, productos):
    """
    Para cada fila de panel_df, cuenta cuántas ops de cada producto
    ocurrieron en los 90 días anteriores a fecha_corte.
    Vectorizado por NIT para mayor velocidad.
    """
    ops_by_nit = ops_hist.groupby('NIT')
    result = {f'hist_90d_{p}': np.zeros(len(panel_df), dtype=np.int16) for p in productos}
    nits = panel_df['NIT'].values
    fechas = panel_df['fecha_corte'].values.astype('datetime64[ns]')
    ventana = np.timedelta64(90, 'D')

    nit_to_idx = {}
    for i, nit in enumerate(nits):
        if nit not in nit_to_idx:
            nit_to_idx[nit] = []
        nit_to_idx[nit].append(i)

    total_nits = len(nit_to_idx)
    for k, (nit, indices) in enumerate(nit_to_idx.items()):
        if k % 500 == 0:
            print(f"  Historial: {k}/{total_nits} NITs")
        if nit not in ops_by_nit.groups:
            continue
        grp = ops_by_nit.get_group(nit)
        fechas_nit = fechas[indices]
        for prod in productos:
            ops_prod = grp[grp['Producto_grupo'] == prod]['Fecha'].values.astype('datetime64[ns]')
            if len(ops_prod) == 0:
                continue
            # Para cada fecha_corte del NIT, contar ops en (fc-90d, fc]
            counts = np.array([
                np.sum((ops_prod > (fc - ventana)) & (ops_prod <= fc))
                for fc in fechas_nit
            ], dtype=np.int16)
            for idx, c in zip(indices, counts):
                result[f'hist_90d_{prod}'][idx] = c

    return pd.DataFrame(result, index=panel_df.index)

# ── 5. Historial para panel_producto (entrenamiento) ─────────────
print("\nConstruyendo historial para entrenamiento...")
ops_hist = df[['NIT', 'Fecha', 'Producto']].copy()
ops_hist['Producto_grupo'] = ops_hist['Producto'].apply(agrupar_producto)

hist_train = calcular_historial_90d(panel_producto, ops_hist, productos)
for col in hist_train.columns:
    panel_producto[col] = hist_train[col]

# ── 5b. Historial para panel COMPLETO + TRM ───────────────────────
print("\nConstruyendo historial para panel completo (tarda más)...")
hist_completo = calcular_historial_90d(panel, ops_hist, productos)
panel_completo = panel.copy()
for col in hist_completo.columns:
    panel_completo[col] = hist_completo[col]

# Agregar TRM al panel completo
panel_completo = panel_completo.merge(
    trm_usd_diario[['Fecha', 'TRM_USD', 'variacion_trm_7d']],
    left_on='fecha_corte', right_on='Fecha', how='left'
).drop(columns='Fecha')

panel_completo.to_parquet(PROCESSED + 'panel_semanal_completo.parquet', index=False)
print(f" Panel completo guardado: {panel_completo.shape}")

# ── 6. Agregar TRM a panel_producto ──────────────────────────────
panel_producto = panel_producto.merge(
    trm_usd_diario[['Fecha', 'TRM_USD', 'variacion_trm_7d']],
    left_on='fecha_corte', right_on='Fecha', how='left'
).drop(columns='Fecha')

# ── 7. Encode del target ──────────────────────────────────────────
le = LabelEncoder()
panel_producto['label_prod'] = le.fit_transform(panel_producto['producto_siguiente'])
print(f"\nClases: {list(le.classes_)}")

# ── 8. Train / Test split ─────────────────────────────────────────
FECHA_CORTE_TRAIN = pd.Timestamp('2025-01-01')
train_p = panel_producto[panel_producto['fecha_corte'] < FECHA_CORTE_TRAIN].copy()
test_p  = panel_producto[panel_producto['fecha_corte'] >= FECHA_CORTE_TRAIN].copy()

features_prod = ['dias_desde_ultima_op', 'freq_30d', 'freq_90d', 'monto_prom_90d',
                 'CIIU_BUC', 'Segmento', 'Subsegmento', 'dia_mes', 'mes', 'semana_año',
                 'hist_90d_SPOT', 'hist_90d_FORWARD', 'hist_90d_NEXT DAY',
                 'hist_90d_FIX', 'hist_90d_OTROS',
                 'TRM_USD', 'variacion_trm_7d']
cat_features_prod = ['CIIU_BUC', 'Segmento', 'Subsegmento']

for col in cat_features_prod:
    train_p[col] = train_p[col].astype('category')
    test_p[col]  = test_p[col].astype('category')

# ── 9. Entrenar con pesos balanceados ─────────────────────────────
print("Entrenando modelo producto...")
# Clases (orden alfabético de LabelEncoder): 0=FIX, 1=FORWARD, 2=NEXT DAY, 3=OTROS, 4=SPOT
# FORWARD, NEXT DAY y SPOT reciben el mismo peso para no favorecer
# artificialmente a una sobre otra dentro de ese grupo.
pesos_suaves = {0: 3.0, 1: 1.0, 2: 1.0, 3: 4.0, 4: 1.0}

modelo = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=5,
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    class_weight=pesos_suaves
)
modelo.fit(train_p[features_prod], train_p['label_prod'],
           categorical_feature=cat_features_prod)

# ── 10. Evaluar ───────────────────────────────────────────────────
pred = modelo.predict(test_p[features_prod])
print("\n── Resultados ──")
print(classification_report(test_p['label_prod'], pred, target_names=le.classes_))

# ── 11. Guardar ───────────────────────────────────────────────────
with open(MODELS + 'modelo_producto_v4.pkl', 'wb') as f:
    pickle.dump(modelo, f)

with open(MODELS + 'label_encoder_producto.pkl', 'wb') as f:
    pickle.dump(le, f)

metadata = {
    'clases':            list(le.classes_),
    'features':          features_prod,
    'cat_features':      cat_features_prod,
    'fecha_corte_train': '2025-01-01',
    'balanceo':          'FIX=3, FORWARD=1, NEXT DAY=1, OTROS=4, SPOT=1',
    'moneda':            'Solo operaciones en USD',
    'trm':               'TRM_USD + variacion_trm_7d incluidas',
}
with open(MODELS + 'modelo_producto_v4_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n Guardado: {MODELS}modelo_producto_v4.pkl")
print(f" Guardado: {MODELS}label_encoder_producto.pkl")
print(f" Guardado: {MODELS}modelo_producto_v4_metadata.json")
