import pandas as pd
import numpy as np
import os
import gc

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
os.makedirs(PROCESSED, exist_ok=True)

PRODUCTOS_VALIDOS = ['SPOT', 'FORWARD', 'FIX', 'NEXT DAY', 'OTROS']

def agrupar_producto(p):
    p = str(p).strip().upper()
    if p in {'OPCIONES', 'SWAPS'}:
        return 'OTROS'
    return p if p in PRODUCTOS_VALIDOS else 'OTROS'

# ── 1. Cargar consolidado ─────────────────────────────────────────
print("Cargando dataset consolidado...")
df = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet',
                     columns=['Fecha', 'NIT', 'Producto', 'Moneda', 'Lado',
                               'Monto_Total_', 'CIIU_BUC', 'Segmento',
                               'Subsegmento', 'TRM_USD', 'TRM_EUR',
                               'TRM_aplicable'])
df['Fecha'] = pd.to_datetime(df['Fecha']).astype('datetime64[ns]')

trm = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet',
                      columns=['Fecha', 'variacion_trm_7d'])
trm['Fecha'] = pd.to_datetime(trm['Fecha']).astype('datetime64[ns]')
df = df.merge(trm, on='Fecha', how='left')
df['variacion_trm_7d'] = df['variacion_trm_7d'].fillna(0)

df = df[df['Monto_Total_'] > 0].copy()
df['NIT']            = df['NIT'].astype(str)
df['Producto_grupo'] = df['Producto'].apply(agrupar_producto)
df['log_monto']      = np.log1p(df['Monto_Total_'])

print(f"Registros cargados: {len(df):,}")

# ── 2. Features de calendario ─────────────────────────────────────
df['anio']          = df['Fecha'].dt.year
df['mes']           = df['Fecha'].dt.month
df['dia']           = df['Fecha'].dt.day
df['dia_semana']    = df['Fecha'].dt.dayofweek
df['semana_anio']   = df['Fecha'].dt.isocalendar().week.astype(int)
df['trimestre']     = df['Fecha'].dt.quarter
df['es_inicio_mes'] = (df['Fecha'].dt.day <= 5).astype(int)
df['es_fin_mes']    = (df['Fecha'].dt.day >= 25).astype(int)

# ── 3. Features históricas acumuladas (sin fuga) ──────────────────
print("Construyendo features históricas por cliente/producto/moneda...")
df = df.sort_values(['NIT', 'Fecha']).reset_index(drop=True)

for keys, prefix in [
    (['NIT'],                             'cli'),
    (['NIT', 'Producto_grupo'],           'cli_prod'),
    (['NIT', 'Producto_grupo', 'Moneda'], 'cli_prod_mon'),
    (['Producto_grupo'],                  'prod'),
    (['Producto_grupo', 'Moneda'],        'prod_mon'),
]:
    grp        = df.groupby(keys, sort=False)
    count_prev = grp.cumcount()
    cumsum_prev = grp['log_monto'].cumsum() - df['log_monto']
    df[f'{prefix}_count_prev'] = count_prev
    df[f'{prefix}_mean_prev']  = cumsum_prev / np.maximum(count_prev, 1)
    df[f'{prefix}_mean_prev']  = df[f'{prefix}_mean_prev'].where(count_prev > 0, np.nan)

# Días desde última operación del cliente
df['dias_desde_ultima_operacion'] = (
    df.groupby('NIT', sort=False)['Fecha']
      .diff()
      .dt.days
      .fillna(999)
      .astype(int)
)

# ── 4. Imputación ─────────────────────────────────────────────────
num_cols = [c for c in df.columns
            if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and c not in ['Monto_Total_', 'log_monto']]
for col in num_cols:
    med = df[col].median()
    df[col] = df[col].fillna(med if not np.isnan(med) else 0)

cat_cols = ['NIT', 'Producto_grupo', 'Moneda', 'Lado',
            'CIIU_BUC', 'Segmento', 'Subsegmento']
for col in cat_cols:
    df[col] = df[col].astype(str).fillna('Sin informacion')

# ── 5. Guardar ────────────────────────────────────────────────────
out = PROCESSED + 'dataset_monto.parquet'
df.to_parquet(out, index=False)
print(f"\nGuardado: {out}")
print(f"Shape final: {df.shape}")
print(f"Rango fechas: {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")
gc.collect()