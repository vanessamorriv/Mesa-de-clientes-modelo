import pandas as pd
import numpy as np
import os
from workalendar.america import Colombia

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW       = os.path.join(BASE, 'data', 'raw') + os.sep
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
os.makedirs(PROCESSED, exist_ok=True)

UMBRAL_HIPERACTIVO = 45
FECHA_INICIO_PANEL = pd.Timestamp('2023-01-01')

# ── 1. Cargar datos ───────────────────────────────────────────────────
print("Cargando datos...")
df       = pd.read_parquet(PROCESSED + 'dataset_consolidado.parquet',
                            columns=['Fecha', 'NIT', 'Monto_Total_'])
trm      = pd.read_parquet(PROCESSED + 'trm_usd_diario.parquet')
clientes = pd.read_excel(RAW + 'clientes.xlsx',
                          usecols=['ID', 'Segmento', 'Subsegmento', 'CIIU_BUC'])

df['Fecha']  = pd.to_datetime(df['Fecha'])
trm['Fecha'] = pd.to_datetime(trm['Fecha'])

# ── 2. Calendario hábil colombiano ────────────────────────────────────
print("Construyendo calendario...")
cal = Colombia()
festivos = [pd.Timestamp(d) for y in range(2020, 2027)
            for d, _ in cal.holidays(y)]
dias_habiles = pd.DatetimeIndex(
    pd.bdate_range('2020-01-02', '2026-05-29', freq='C', holidays=festivos))
dias_panel = dias_habiles[dias_habiles >= FECHA_INICIO_PANEL]
print(f"Días a procesar: {len(dias_panel)}")

# ── 3. Atributos estáticos de clientes ───────────────────────────────
attrs = (clientes.rename(columns={'ID': 'NIT'})
                 .drop_duplicates(subset='NIT')
                 .set_index('NIT'))

nits_activos = df['NIT'].unique()
print(f"NITs activos: {len(nits_activos)}")

# ── 4. Pre-computar operaciones indexadas por fecha y NIT ─────────────
# Ordenar una sola vez — no volver a filtrar el df completo en el loop
df = df.sort_values('Fecha').reset_index(drop=True)
df['Fecha_ord'] = df['Fecha'].astype(np.int64)  # para búsquedas rápidas

# TRM indexado por fecha
trm_idx = trm.set_index('Fecha')[['TRM_USD', 'variacion_trm_7d']]

# ── 5. Construir panel vectorizado ────────────────────────────────────
print("Construyendo panel diario (vectorizado)...")
bloques = []

for i, fecha_corte in enumerate(dias_panel):
    if i % 100 == 0:
        print(f"  {fecha_corte.date()} ({i}/{len(dias_panel)})...")

    hace_30d = fecha_corte - pd.Timedelta(days=30)
    hace_90d = fecha_corte - pd.Timedelta(days=90)

    # Filtrar una sola vez con searchsorted (mucho más rápido)
    idx_corte = df['Fecha'].searchsorted(fecha_corte, side='left')
    idx_30d   = df['Fecha'].searchsorted(hace_30d,    side='left')
    idx_90d   = df['Fecha'].searchsorted(hace_90d,    side='left')

    hist  = df.iloc[:idx_corte]          # toda la historia hasta ayer
    w30   = df.iloc[idx_30d:idx_corte]   # últimos 30 días
    w90   = df.iloc[idx_90d:idx_corte]   # últimos 90 días

    # Aggregations vectorizadas
    freq30    = w30.groupby('NIT').size()
    freq90    = w90.groupby('NIT').size()
    monto90   = w90.groupby('NIT')['Monto_Total_'].mean()
    ultima_op = hist.groupby('NIT')['Fecha'].max()

    # Label: ¿operó el siguiente día hábil?
    if i + 1 >= len(dias_panel):
        continue
    dia_sig      = dias_panel[i + 1]
    nits_manana  = set(df.loc[df['Fecha'] == dia_sig, 'NIT'].unique())

    # TRM del día
    trm_row  = trm_idx.loc[fecha_corte] if fecha_corte in trm_idx.index \
               else pd.Series({'TRM_USD': np.nan, 'variacion_trm_7d': np.nan})

    # Construir bloque del día como DataFrame de una vez
    base = pd.DataFrame({'NIT': nits_activos})
    base['freq_30d']      = base['NIT'].map(freq30).fillna(0).astype(int)
    base['freq_90d']      = base['NIT'].map(freq90).fillna(0).astype(int)

    # Filtrar hiperactivos
    base = base[base['freq_90d'] < UMBRAL_HIPERACTIVO].copy()

    base['monto_prom_90d']      = base['NIT'].map(monto90).fillna(0)
    base['ultima_op']           = base['NIT'].map(ultima_op)
    base['dias_desde_ultima_op']= (fecha_corte - base['ultima_op']).dt.days.fillna(999).astype(int)
    base['fecha_corte']         = fecha_corte
    base['dia_semana']          = fecha_corte.weekday()
    base['dia_mes']             = fecha_corte.day
    base['mes']                 = fecha_corte.month
    base['TRM_USD']             = trm_row['TRM_USD']
    base['variacion_trm_7d']    = trm_row['variacion_trm_7d']
    base['label']               = base['NIT'].isin(nits_manana).astype(int)

    base = base.drop(columns=['ultima_op'])
    bloques.append(base)

# ── 6. Concatenar y hacer merge con atributos ─────────────────────────
print("Concatenando bloques...")
panel = pd.concat(bloques, ignore_index=True)

print("Mergeando atributos de clientes...")
panel = panel.join(attrs, on='NIT', how='left')

# ── 7. Validaciones ───────────────────────────────────────────────────
print("\n── Validaciones ──")
print(f"Shape panel diario:    {panel.shape}")
print(f"Rango fechas:          {panel['fecha_corte'].min().date()} → {panel['fecha_corte'].max().date()}")
print(f"Tasa positivos global: {panel['label'].mean():.4f} ({panel['label'].mean()*100:.2f}%)")
print(f"Días únicos:           {panel['fecha_corte'].nunique()}")
print(f"NITs únicos:           {panel['NIT'].nunique()}")
print(f"Nulos en Segmento:     {panel['Segmento'].isna().sum()}")

# ── 8. Guardar ────────────────────────────────────────────────────────
out = PROCESSED + 'panel_diario_completo.parquet'
panel.to_parquet(out, index=False)
print(f"\nGuardado: {out}")