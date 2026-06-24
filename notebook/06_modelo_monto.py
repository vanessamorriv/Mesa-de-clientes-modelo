import pandas as pd
import numpy as np
import pickle
import json
import os
import gc
from catboost import CatBoostRegressor

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
MODELS    = os.path.join(BASE, 'models') + os.sep
os.makedirs(MODELS, exist_ok=True)

NUM_FEATURES = [
    'anio', 'mes', 'dia', 'dia_semana', 'semana_anio', 'trimestre',
    'es_inicio_mes', 'es_fin_mes',
    'TRM_USD', 'TRM_EUR', 'variacion_trm_7d',
    'cli_count_prev', 'cli_mean_prev',
    'cli_prod_count_prev', 'cli_prod_mean_prev',
    'cli_prod_mon_count_prev', 'cli_prod_mon_mean_prev',
    'prod_count_prev', 'prod_mean_prev',
    'prod_mon_count_prev', 'prod_mon_mean_prev',
    'dias_desde_ultima_operacion',
]
CAT_FEATURES = ['NIT', 'Producto_grupo', 'Moneda', 'Lado',
                'CIIU_BUC', 'Segmento', 'Subsegmento']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

PRODUCTOS_VALIDOS = ['SPOT', 'FORWARD', 'FIX', 'NEXT DAY', 'OTROS']
MIN_TRAIN_PRODUCTO = 200

# ── 1. Cargar (se agrega Monto_Total_ para poder winsorizar) ──────
print("Cargando dataset de monto...")
df = pd.read_parquet(PROCESSED + 'dataset_monto.parquet',
                     columns=ALL_FEATURES + ['log_monto', 'Monto_Total_', 'Fecha'])
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values('Fecha').reset_index(drop=True)

for col in CAT_FEATURES:
    df[col] = df[col].astype(str)
for col in NUM_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"Registros: {len(df):,}  |  Rango: {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")

X = df[ALL_FEATURES]
y = df['log_monto']

# ── 2. Winsorizing global (solo se usa para entrenar P50 por producto) ──
p01 = df['Monto_Total_'].quantile(0.01)
p99 = df['Monto_Total_'].quantile(0.99)
df['log_monto_clip'] = np.log1p(df['Monto_Total_'].clip(lower=p01, upper=p99))
print(f"P01 monto: ${p01:,.0f} | P99 monto: ${p99:,.0f}")

def entrenar_catboost(X_tr, y_tr, quantile, cat_feats, verbose=0):
    modelo = CatBoostRegressor(
        loss_function=f'Quantile:alpha={quantile}',
        iterations=500, learning_rate=0.05, depth=6,
        cat_features=cat_feats, random_seed=42, verbose=verbose,
    )
    modelo.fit(X_tr, y_tr)
    return modelo

# ── 3. P10 y P90 — modelos GLOBALES, sin cambios respecto al original ──
modelos_globales = {}
for q in [0.1, 0.9]:
    nombre = f'monto_p{int(q*100)}'
    print(f"\nEntrenando CatBoost Quantile {int(q*100)} (global)...")
    modelo = entrenar_catboost(X, y, q, CAT_FEATURES, verbose=100)
    modelos_globales[nombre] = modelo
    with open(MODELS + f'{nombre}.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    print(f"  Guardado: {nombre}.pkl")

# ── 4. P50 global — se entrena igual, queda como FALLBACK ──────────
print(f"\nEntrenando CatBoost Quantile 50 (global, fallback)...")
modelo_p50_global = entrenar_catboost(X, y, 0.5, CAT_FEATURES, verbose=100)
with open(MODELS + 'monto_p50_global.pkl', 'wb') as f:
    pickle.dump(modelo_p50_global, f)
print("  Guardado: monto_p50_global.pkl")

# ── 5. P50 por producto (HÍBRIDO) — entrenado con target winsorizado ──
cat_feats_sin_prod = [c for c in CAT_FEATURES if c != 'Producto_grupo']
feats_prod          = [f for f in ALL_FEATURES if f != 'Producto_grupo']

productos_con_modelo_propio = []
for producto in PRODUCTOS_VALIDOS:
    mask = df['Producto_grupo'] == producto
    n    = mask.sum()
    nombre_archivo = f"monto_p50_{producto.replace(' ', '_')}"

    if n < MIN_TRAIN_PRODUCTO:
        print(f"\n  {producto}: pocos datos ({n}) — usará el modelo global como fallback")
        continue

    print(f"\nEntrenando CatBoost Quantile 50 — producto: {producto} ({n:,} registros)...")
    X_prod = df.loc[mask, feats_prod]
    y_prod = df.loc[mask, 'log_monto_clip']
    modelo_prod = entrenar_catboost(X_prod, y_prod, 0.5, cat_feats_sin_prod, verbose=100)

    with open(MODELS + f'{nombre_archivo}.pkl', 'wb') as f:
        pickle.dump(modelo_prod, f)
    print(f"  Guardado: {nombre_archivo}.pkl")
    productos_con_modelo_propio.append(producto)

# ── 6. Metadata ───────────────────────────────────────────────────
metadata = {
    'features_num': NUM_FEATURES,
    'features_cat': CAT_FEATURES,
    'features_cat_sin_producto': cat_feats_sin_prod,
    'target': 'log_monto',
    'target_p50_por_producto': 'log_monto_clip',
    'p01_winsorizing': float(p01),
    'p99_winsorizing': float(p99),
    'productos_validos': PRODUCTOS_VALIDOS,
    'productos_con_modelo_propio': productos_con_modelo_propio,
    'nota': (
        'Esquema HÍBRIDO: monto_p10.pkl y monto_p90.pkl son modelos GLOBALES '
        '(todas las operaciones, sin winsorizar). monto_p50 se predice POR PRODUCTO: '
        'usar monto_p50_{producto}.pkl si el producto está en productos_con_modelo_propio, '
        'si no, usar monto_p50_global.pkl como fallback. '
        'Todas las predicciones están en log(1+monto); aplicar expm1() para obtener el monto real.'
    ),
    'fecha_entrenamiento': str(df['Fecha'].max().date()),
    'n_registros': len(df),
}
with open(MODELS + 'modelo_monto_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nEntrenamiento completo (esquema HÍBRIDO).")
print(f"Modelos guardados: monto_p10.pkl, monto_p90.pkl (globales)")
print(f"                    monto_p50_global.pkl (fallback)")
print(f"                    monto_p50_{{producto}}.pkl para: {productos_con_modelo_propio}")
gc.collect()