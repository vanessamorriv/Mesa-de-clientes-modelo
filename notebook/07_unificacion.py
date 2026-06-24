import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
import os
import gc
from workalendar.america import Colombia

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
MODELS    = os.path.join(BASE, 'models') + os.sep
OUTPUT    = os.path.join(BASE, 'outputs') + os.sep
os.makedirs(OUTPUT, exist_ok=True)

CUPO_DIARIO        = 40
UMBRAL_HIPERACTIVO = 45
PRODUCTOS_VALIDOS  = ['SPOT', 'FORWARD', 'FIX', 'NEXT DAY', 'OTROS']

def agrupar_producto(p):
    p = str(p).strip().upper()
    return 'OTROS' if p in {'OPCIONES', 'SWAPS'} or p not in PRODUCTOS_VALIDOS else p

# ── 1. Cargar datos base ──────────────────────────────────────────
print("Cargando datos...")
df_ops = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet',
                         columns=['Fecha', 'NIT', 'Monto_Total_',
                                  'Producto', 'Moneda', 'Lado',
                                  'CIIU_BUC', 'Segmento', 'Subsegmento'])
df_ops['Fecha'] = pd.to_datetime(df_ops['Fecha']).astype('datetime64[ns]')
df_ops['NIT']   = df_ops['NIT'].astype(str)
df_ops['Producto_grupo'] = df_ops['Producto'].apply(agrupar_producto)

trm = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet',
                      columns=['Fecha', 'TRM_USD', 'variacion_trm_7d'])
trm['Fecha'] = pd.to_datetime(trm['Fecha']).astype('datetime64[ns]')

traders = pd.read_parquet(PROCESSED + 'traders.parquet')

# ── 2. Fechas ─────────────────────────────────────────────────────
print("Calculando fechas...")
cal      = Colombia()
festivos = [pd.Timestamp(d) for y in range(2020, 2028)
            for d, _ in cal.holidays(y)]

FECHA_ANCLA = df_ops['Fecha'].max()
todos_habiles = pd.DatetimeIndex(
    pd.bdate_range(FECHA_ANCLA, FECHA_ANCLA + pd.Timedelta(days=10),
                   freq='C', holidays=festivos))
DIA_PREDICCION = todos_habiles[todos_habiles > FECHA_ANCLA][0]

print(f"Fecha ancla:     {FECHA_ANCLA.date()}")
print(f"Día predicción:  {DIA_PREDICCION.date()}")

trm_ancla = trm[trm['Fecha'] <= FECHA_ANCLA].sort_values('Fecha').iloc[-1]

# ── 3. Construir features de timing (MEJORADO: + 3 features nuevas) ────
print("\nConstruyendo features de timing...")
attrs = (df_ops[['NIT', 'Segmento', 'Subsegmento', 'CIIU_BUC']]
         .drop_duplicates('NIT')
         .set_index('NIT'))

nits_activos = df_ops['NIT'].unique()

w30 = df_ops[df_ops['Fecha'] >= FECHA_ANCLA - pd.Timedelta(days=30)]
w90 = df_ops[df_ops['Fecha'] >= FECHA_ANCLA - pd.Timedelta(days=90)]

freq30  = w30.groupby('NIT').size()
freq90  = w90.groupby('NIT').size()
monto90 = w90.groupby('NIT')['Monto_Total_'].mean()
ult_op  = df_ops.groupby('NIT')['Fecha'].max()

# Features nuevas de timing: promedio_dias_entre_ops, dia_favorito_cliente
gap_medio = (df_ops.sort_values(['NIT', 'Fecha'])
                    .groupby('NIT')['Fecha']
                    .apply(lambda x: x.diff().dt.days.mean())
                    .fillna(999)
                    .rename('promedio_dias_entre_ops'))

dia_fav = (df_ops.copy()
                  .assign(dia=df_ops['Fecha'].dt.weekday)
                  .groupby(['NIT', 'dia'])
                  .size()
                  .reset_index(name='n')
                  .sort_values('n', ascending=False)
                  .drop_duplicates('NIT')
                  .set_index('NIT')['dia']
                  .rename('dia_favorito_cliente'))

base = pd.DataFrame({'NIT': nits_activos})
base['freq_30d']             = base['NIT'].map(freq30).fillna(0).astype(int)
base['freq_90d']             = base['NIT'].map(freq90).fillna(0).astype(int)
base = base[base['freq_90d'] < UMBRAL_HIPERACTIVO].copy()
base['monto_prom_90d']       = base['NIT'].map(monto90).fillna(0)
base['dias_desde_ultima_op'] = (FECHA_ANCLA - base['NIT'].map(ult_op)).dt.days.fillna(999).astype(int)
base['dia_semana']           = DIA_PREDICCION.weekday()
base['dia_mes']              = DIA_PREDICCION.day
base['mes']                  = DIA_PREDICCION.month
base['TRM_USD']              = trm_ancla['TRM_USD']
base['variacion_trm_7d']     = trm_ancla['variacion_trm_7d']
base['promedio_dias_entre_ops'] = base['NIT'].map(gap_medio).fillna(999)
base['dia_favorito_cliente']    = base['NIT'].map(dia_fav).fillna(-1).astype(int)
base['es_dia_favorito']         = (base['dia_semana'] == base['dia_favorito_cliente']).astype(int)
base = base.merge(attrs.reset_index(), on='NIT', how='left')

cat_features_timing = ['CIIU_BUC', 'Segmento', 'Subsegmento']
for col in cat_features_timing:
    base[col] = base[col].astype('category')

# ── 4. Predecir timing (lee features dinámicamente desde metadata) ─────
print("Prediciendo timing...")
modelo_timing = lgb.Booster(model_file=MODELS + 'modelo_timing_final.txt')

with open(MODELS + 'modelo_timing_metadata.json') as f:
    meta_timing = json.load(f)
features_timing = meta_timing['features']

base['prob_opera'] = modelo_timing.predict(base[features_timing])

# Priorizar por segmento con cupo diario
demanda = base.groupby('Segmento', observed=True)['prob_opera'].sum()
props   = demanda / demanda.sum()
cupos   = (props * CUPO_DIARIO).round().astype(int).clip(lower=1)

lista = []
for seg, cupo in cupos.items():
    sub = base[base['Segmento'] == seg].sort_values('prob_opera', ascending=False)
    lista.append(sub.head(cupo))
clientes = pd.concat(lista, ignore_index=True)
print(f"Clientes priorizados: {len(clientes)}")

# ── 5. Construir features de producto ────────────────────────────
print("\nConstruyendo features de producto...")
with open(MODELS + 'modelo_producto_v4.pkl', 'rb') as f:
    modelo_producto = pickle.load(f)
with open(MODELS + 'label_encoder_producto.pkl', 'rb') as f:
    le_producto = pickle.load(f)
with open(MODELS + 'modelo_producto_v4_metadata.json') as f:
    meta_producto = json.load(f)

prod_input = clientes[['NIT', 'freq_30d', 'freq_90d', 'monto_prom_90d',
                        'dias_desde_ultima_op', 'CIIU_BUC', 'Segmento',
                        'Subsegmento', 'TRM_USD', 'variacion_trm_7d']].copy()

prod_input['dia_semana'] = DIA_PREDICCION.weekday()
prod_input['dia_mes']    = DIA_PREDICCION.day
prod_input['mes']        = DIA_PREDICCION.month

ops_90d = df_ops[df_ops['Fecha'] >= FECHA_ANCLA - pd.Timedelta(days=90)]
for p in PRODUCTOS_VALIDOS:
    col     = f'hist_90d_{p.replace(" ", "_")}'
    conteos = ops_90d[ops_90d['Producto_grupo'] == p].groupby('NIT').size()
    prod_input[col] = prod_input['NIT'].map(conteos).fillna(0).astype(int)

features_prod  = meta_producto['features']
cat_feats_prod = meta_producto['cat_features']

for col in cat_feats_prod:
    if col not in prod_input.columns:
        prod_input[col] = 'Sin informacion'
    prod_input[col] = prod_input[col].astype(str).astype('category')

for col in [c for c in features_prod if c not in cat_feats_prod]:
    if col not in prod_input.columns:
        prod_input[col] = 0
    prod_input[col] = pd.to_numeric(prod_input[col], errors='coerce').fillna(0)

proba_producto   = modelo_producto.predict(prod_input[features_prod])
pred_encoded     = np.argmax(proba_producto, axis=1)
clientes['producto_predicho']      = le_producto.inverse_transform(pred_encoded)
clientes['prob_producto']          = proba_producto.max(axis=1)
print(f"Distribución producto predicho:\n{clientes['producto_predicho'].value_counts()}")

# ── 6. Construir features de monto ───────────────────────────────
print("\nConstruyendo features de monto...")
with open(MODELS + 'modelo_monto_metadata.json') as f:
    meta_monto = json.load(f)

num_feats_monto = meta_monto['features_num']
cat_feats_monto = meta_monto['features_cat']
all_feats_monto = num_feats_monto + cat_feats_monto
productos_con_modelo_propio = meta_monto.get('productos_con_modelo_propio', [])

monto_input = clientes.copy()
monto_input['Producto_grupo'] = monto_input['producto_predicho'].apply(agrupar_producto)

moneda_freq = (ops_90d.groupby(['NIT', 'Moneda'])
                       .size()
                       .reset_index(name='n')
                       .sort_values('n', ascending=False)
                       .drop_duplicates('NIT')
                       .set_index('NIT')['Moneda'])
lado_freq   = (ops_90d.groupby(['NIT', 'Lado'])
                       .size()
                       .reset_index(name='n')
                       .sort_values('n', ascending=False)
                       .drop_duplicates('NIT')
                       .set_index('NIT')['Lado'])

monto_input['Moneda'] = monto_input['NIT'].map(moneda_freq).fillna('USD/COP')
monto_input['Lado']   = monto_input['NIT'].map(lado_freq).fillna('Sin informacion')

monto_input['anio']          = DIA_PREDICCION.year
monto_input['mes']           = DIA_PREDICCION.month
monto_input['dia']           = DIA_PREDICCION.day
monto_input['dia_semana']    = DIA_PREDICCION.weekday()
monto_input['semana_anio']   = DIA_PREDICCION.isocalendar()[1]
monto_input['trimestre']     = (DIA_PREDICCION.month - 1) // 3 + 1
monto_input['es_inicio_mes'] = int(DIA_PREDICCION.day <= 5)
monto_input['es_fin_mes']    = int(DIA_PREDICCION.day >= 25)
monto_input['TRM_EUR']       = trm_ancla.get('TRM_EUR', np.nan)

df_monto = pd.read_parquet(PROCESSED + 'dataset_monto.parquet',
                            columns=['NIT', 'Fecha',
                                     'cli_count_prev', 'cli_mean_prev',
                                     'cli_prod_count_prev', 'cli_prod_mean_prev',
                                     'cli_prod_mon_count_prev', 'cli_prod_mon_mean_prev',
                                     'prod_count_prev', 'prod_mean_prev',
                                     'prod_mon_count_prev', 'prod_mon_mean_prev',
                                     'dias_desde_ultima_operacion'])
df_monto['Fecha'] = pd.to_datetime(df_monto['Fecha'])
df_monto['NIT']   = df_monto['NIT'].astype(str)

ultimas = df_monto.sort_values('Fecha').groupby('NIT').last().reset_index()

hist_cols = ['cli_count_prev', 'cli_mean_prev',
             'cli_prod_count_prev', 'cli_prod_mean_prev',
             'cli_prod_mon_count_prev', 'cli_prod_mon_mean_prev',
             'prod_count_prev', 'prod_mean_prev',
             'prod_mon_count_prev', 'prod_mon_mean_prev',
             'dias_desde_ultima_operacion']

monto_input = monto_input.merge(ultimas[['NIT'] + hist_cols],
                                 on='NIT', how='left')
for col in hist_cols:
    monto_input[col] = monto_input[col].fillna(0)

for col in cat_feats_monto:
    if col not in monto_input.columns:
        monto_input[col] = 'Sin informacion'
    monto_input[col] = monto_input[col].astype(str)

for col in num_feats_monto:
    if col not in monto_input.columns:
        monto_input[col] = 0
    monto_input[col] = pd.to_numeric(monto_input[col], errors='coerce').fillna(0)

# ── 6b. Predecir monto — P10/P90 globales, P50 HÍBRIDO por producto ────
print("\nPrediciendo monto (esquema HÍBRIDO)...")

# P10 y P90: modelos globales, sin cambios
for q_name in ['monto_p10', 'monto_p90']:
    path = MODELS + f'{q_name}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            modelo_q = pickle.load(f)
        preds_log = modelo_q.predict(monto_input[all_feats_monto])
        clientes[q_name] = np.expm1(preds_log).clip(min=0)
    else:
        print(f"  AVISO: {q_name}.pkl no encontrado")
        clientes[q_name] = np.nan

# P50: por producto, con fallback al modelo global
feats_prod_monto    = [f for f in all_feats_monto if f != 'Producto_grupo']
preds_p50           = np.full(len(monto_input), np.nan)

# Cargar fallback global una sola vez
modelo_p50_global = None
path_global = MODELS + 'monto_p50_global.pkl'
if os.path.exists(path_global):
    with open(path_global, 'rb') as f:
        modelo_p50_global = pickle.load(f)
else:
    print("  AVISO: monto_p50_global.pkl no encontrado (no habrá fallback)")

for producto in PRODUCTOS_VALIDOS:
    mask = (monto_input['Producto_grupo'] == producto).values
    if mask.sum() == 0:
        continue

    nombre_archivo = f"monto_p50_{producto.replace(' ', '_')}"
    path_prod = MODELS + f'{nombre_archivo}.pkl'

    if producto in productos_con_modelo_propio and os.path.exists(path_prod):
        with open(path_prod, 'rb') as f:
            modelo_prod = pickle.load(f)
        preds_log = modelo_prod.predict(monto_input.loc[mask, feats_prod_monto])
        preds_p50[mask] = np.expm1(preds_log).clip(min=0)
    elif modelo_p50_global is not None:
        preds_log = modelo_p50_global.predict(monto_input.loc[mask, all_feats_monto])
        preds_p50[mask] = np.expm1(preds_log).clip(min=0)
    else:
        print(f"  AVISO: sin modelo disponible para producto {producto}")

clientes['monto_p50'] = preds_p50

# ── 7. Salida unificada ───────────────────────────────────────────
print("\nGenerando salida unificada...")
traders['NIT'] = traders['NIT'].astype(str)
clientes = clientes.merge(traders, on='NIT', how='left')
clientes['Trader']        = clientes['Trader'].fillna('No asignado')
clientes['fecha_ancla']   = FECHA_ANCLA
clientes['dia_llamada']   = DIA_PREDICCION

cols_salida = ['NIT', 'Trader', 'Segmento', 'Subsegmento',
               'prob_opera', 'producto_predicho',
               'monto_p10', 'monto_p50', 'monto_p90',
               'freq_30d', 'freq_90d', 'dias_desde_ultima_op',
               'fecha_ancla', 'dia_llamada']

salida = clientes[cols_salida].sort_values('prob_opera', ascending=False).reset_index(drop=True)

out_csv     = OUTPUT + f'lista_unificada_{DIA_PREDICCION.date()}.csv'
out_parquet = OUTPUT + 'lista_unificada_ultima.parquet'
salida.to_csv(out_csv, index=False)
salida.to_parquet(out_parquet, index=False)

print(f"\n{'='*55}")
print(f"SALIDA UNIFICADA — {DIA_PREDICCION.date()}")
print(f"{'='*55}")
print(f"Clientes priorizados:    {len(salida)}")
print(f"Con trader asignado:     {(salida['Trader'] != 'No asignado').sum()}")
print(f"\nDistribución producto:")
print(salida['producto_predicho'].value_counts().to_string())
print(f"\nMonto P50 mediano:       ${salida['monto_p50'].median():,.0f}")
print(f"Monto P50 promedio:      ${salida['monto_p50'].mean():,.0f}")
print(f"\nGuardado: {out_csv}")

del df_ops, df_monto, w30, w90, ops_90d
gc.collect()