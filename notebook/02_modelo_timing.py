import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from workalendar.america import Colombia

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
MODELS    = os.path.join(BASE, 'models') + os.sep

CUPO_DIARIO = 40
UMBRAL_HIPERACTIVO = 45

features = ['dias_desde_ultima_op', 'freq_30d', 'freq_90d', 'monto_prom_90d',
            'CIIU_BUC', 'Segmento', 'Subsegmento',
            'dia_semana', 'dia_mes', 'mes',
            'TRM_USD', 'variacion_trm_7d']
cat_features = ['CIIU_BUC', 'Segmento', 'Subsegmento']

# ── 1. Cargar panel completo (para entrenar) ────────────────────────────
print("Cargando panel diario...")
panel = pd.read_parquet(PROCESSED + 'panel_diario_completo.parquet')
panel['fecha_corte'] = pd.to_datetime(panel['fecha_corte'])
for col in cat_features:
    panel[col] = panel[col].astype('category')

FECHA_ANCLA = panel['fecha_corte'].max()
print(f"Fecha ancla (último dato real disponible): {FECHA_ANCLA.date()}")

# ── 2. Calendario hábil colombiano — primera semana de junio ────────────
print("Construyendo calendario para primera semana de junio...")
cal = Colombia()
festivos_2026 = [pd.Timestamp(d) for d, _ in cal.holidays(2026)]

dias_junio = pd.bdate_range('2026-06-01', '2026-06-10', freq='C',
                             holidays=festivos_2026)
dias_junio = pd.DatetimeIndex(dias_junio)[:5]
print(f"Días hábiles primera semana de junio: {[d.date() for d in dias_junio]}")

# ── 3. Construir features "como si hoy fuera el día ancla" ──────────────
df_ops = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet',
                          columns=['Fecha', 'NIT', 'Monto_Total_'])
df_ops['Fecha'] = pd.to_datetime(df_ops['Fecha'])
df_ops = df_ops.sort_values('Fecha').reset_index(drop=True)

trm = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet')
trm['Fecha'] = pd.to_datetime(trm['Fecha'])
trm_ancla_row = trm[trm['Fecha'] == FECHA_ANCLA]
if len(trm_ancla_row) == 0:
    trm_ancla_row = trm[trm['Fecha'] <= FECHA_ANCLA].sort_values('Fecha').iloc[[-1]]
trm_ancla = trm_ancla_row.iloc[0]
print(f"TRM usada (fecha {trm_ancla['Fecha'].date()}): {trm_ancla['TRM_USD']}")

attrs = panel[['NIT', 'Segmento', 'Subsegmento', 'CIIU_BUC']].drop_duplicates(subset='NIT')
nits_activos = df_ops['NIT'].unique()

hace_30d = FECHA_ANCLA - pd.Timedelta(days=30)
hace_90d = FECHA_ANCLA - pd.Timedelta(days=90)

w30 = df_ops[(df_ops['Fecha'] >= hace_30d) & (df_ops['Fecha'] <= FECHA_ANCLA)]
w90 = df_ops[(df_ops['Fecha'] >= hace_90d) & (df_ops['Fecha'] <= FECHA_ANCLA)]

freq30    = w30.groupby('NIT').size()
freq90    = w90.groupby('NIT').size()
monto90   = w90.groupby('NIT')['Monto_Total_'].mean()
ultima_op = df_ops.groupby('NIT')['Fecha'].max()

base = pd.DataFrame({'NIT': nits_activos})
base['freq_30d'] = base['NIT'].map(freq30).fillna(0).astype(int)
base['freq_90d'] = base['NIT'].map(freq90).fillna(0).astype(int)
base = base[base['freq_90d'] < UMBRAL_HIPERACTIVO].copy()
base['monto_prom_90d'] = base['NIT'].map(monto90).fillna(0)
base['ultima_op'] = base['NIT'].map(ultima_op)
base['dias_desde_ultima_op'] = (FECHA_ANCLA - base['ultima_op']).dt.days.fillna(999).astype(int)
base = base.drop(columns=['ultima_op'])
base = base.merge(attrs, on='NIT', how='left')

# Una fila por cliente por cada día hábil de la primera semana de junio
bloques_junio = []
for fecha in dias_junio:
    b = base.copy()
    b['fecha_corte']      = fecha
    b['dia_semana']       = fecha.weekday()
    b['dia_mes']          = fecha.day
    b['mes']              = fecha.month
    b['TRM_USD']          = trm_ancla['TRM_USD']
    b['variacion_trm_7d'] = trm_ancla['variacion_trm_7d']
    bloques_junio.append(b)

panel_junio = pd.concat(bloques_junio, ignore_index=True)
for col in cat_features:
    panel_junio[col] = panel_junio[col].astype('category')

print(f"Filas construidas para primera semana de junio: {panel_junio.shape}")

del df_ops, w30, w90
gc.collect()

# ── 4. Entrenar modelo final con TODO el panel disponible (hasta mayo) ──
print("\nEntrenando modelo final con todos los datos disponibles (hasta mayo 2026)...")

PARAMS_LGB = {
    'objective': 'binary', 'is_unbalance': True, 'learning_rate': 0.05,
    'max_bin': 63, 'num_leaves': 31, 'min_data_in_leaf': 50,
    'force_col_wise': True, 'verbose': -1, 'seed': 42,
}

train_set = lgb.Dataset(
    panel[features], label=panel['label'],
    categorical_feature=cat_features, free_raw_data=True
)
modelo_final = lgb.train(PARAMS_LGB, train_set, num_boost_round=300)

del train_set, panel
gc.collect()

# ── 5. Predecir primera semana de junio ──────────────────────────────────
proba_junio = modelo_final.predict(panel_junio[features])
panel_junio['prob_opera_manana'] = proba_junio

# ── 6. Priorizar por día (cupo por demanda, igual que siempre) ──────────
lista_junio = []
for fecha in sorted(panel_junio['fecha_corte'].unique()):
    grupo   = panel_junio[panel_junio['fecha_corte'] == fecha].copy()
    demanda = grupo.groupby('Segmento', observed=True)['prob_opera_manana'].sum()
    props   = demanda / demanda.sum()
    cupos   = (props * CUPO_DIARIO).round().astype(int).clip(lower=1)

    for seg, cupo in cupos.items():
        sub = grupo[grupo['Segmento'] == seg].sort_values(
              'prob_opera_manana', ascending=False)
        lista_junio.append(sub.head(cupo))

lista_junio = pd.concat(lista_junio, ignore_index=True)
lista_junio['fecha_prediccion'] = lista_junio['fecha_corte']
lista_junio['dia_llamada']      = lista_junio['fecha_corte'] + pd.Timedelta(days=1)

cols_out = ['fecha_corte', 'dia_llamada', 'NIT', 'Segmento', 'prob_opera_manana']
lista_junio[cols_out].to_csv(MODELS + 'predicciones_primera_semana_junio_2026.csv', index=False)

print(f"\nPredicciones primera semana de junio 2026:")
print(f"  Días con lista:     {lista_junio['fecha_corte'].nunique()}")
print(f"  Clientes por día:   {lista_junio.groupby('fecha_corte').size().mean():.0f} promedio")
print(f"  Total predicciones: {len(lista_junio)}")
print(f"\nGuardado: {MODELS}predicciones_primera_semana_junio_2026.csv")