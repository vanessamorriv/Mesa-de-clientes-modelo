import pandas as pd
import pickle
import os
from supabase import create_client

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, '..', 'models') + os.sep
PROCESSED = os.path.join(BASE, '..', 'data', 'processed') + os.sep

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_KEY']
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Cargar modelos ────────────────────────────────────────────────
with open(MODELS + 'modelo_timing_v1.pkl', 'rb') as f:
    modelo_timing = pickle.load(f)

with open(MODELS + 'modelo_producto_v4.pkl', 'rb') as f:
    modelo_producto = pickle.load(f)

with open(MODELS + 'label_encoder_producto.pkl', 'rb') as f:
    le = pickle.load(f)

# ── Cargar panel COMPLETO (con historial + TRM) ───────────────────
panel = pd.read_parquet(PROCESSED + 'panel_semanal_completo.parquet')  # <-- cambio
fecha_mas_reciente = panel['fecha_corte'].max()
snapshot_hoy = panel[panel['fecha_corte'] == fecha_mas_reciente].copy()

# ── Predecir timing ───────────────────────────────────────────────
features_timing = ['dias_desde_ultima_op', 'freq_30d', 'freq_90d', 'monto_prom_90d',
                   'CIIU_BUC', 'Segmento', 'Subsegmento', 'dia_mes', 'mes', 'semana_año']

for col in ['CIIU_BUC', 'Segmento', 'Subsegmento']:
    snapshot_hoy[col] = snapshot_hoy[col].astype('category')

snapshot_hoy['prob_opera_7d'] = modelo_timing.predict_proba(
    snapshot_hoy[features_timing])[:, 1]

# ── Predecir producto (solo para clientes activos) ────────────────
features_prod = ['dias_desde_ultima_op', 'freq_30d', 'freq_90d', 'monto_prom_90d',
                 'CIIU_BUC', 'Segmento', 'Subsegmento', 'dia_mes', 'mes', 'semana_año',
                 'hist_90d_SPOT', 'hist_90d_FORWARD', 'hist_90d_NEXT DAY',
                 'hist_90d_FIX', 'hist_90d_OTROS',
                 'TRM_USD', 'variacion_trm_7d']

snapshot_hoy['producto_predicho'] = None
mask_activos = snapshot_hoy['prob_opera_7d'] > 0.3

if mask_activos.sum() > 0:
    pred_encoded = modelo_producto.predict(snapshot_hoy.loc[mask_activos, features_prod])
    snapshot_hoy.loc[mask_activos, 'producto_predicho'] = le.inverse_transform(pred_encoded)
    print(f"Productos predichos para {mask_activos.sum()} clientes activos")

# ── Score combinado ───────────────────────────────────────────────
snapshot_hoy['score_prioridad'] = snapshot_hoy['prob_opera_7d']

# ── Preparar y subir ──────────────────────────────────────────────
registros = snapshot_hoy[['NIT', 'prob_opera_7d', 'producto_predicho', 'score_prioridad']].rename(
    columns={'NIT': 'nit'}
)
registros['fecha_prediccion'] = str(fecha_mas_reciente.date())

data = registros.to_dict('records')
for i in range(0, len(data), 500):
    supabase.table('predicciones').upsert(
        data[i:i+500],
        on_conflict='nit,fecha_prediccion'
    ).execute()
    print(f"Subido lote {i}–{i+500}")

print(f"\nTotal subido: {len(data)} predicciones")