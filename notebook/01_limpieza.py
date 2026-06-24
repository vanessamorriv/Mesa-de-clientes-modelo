import pandas as pd
import numpy as np
import os

# Rutas absolutas relativas a la ubicación del script
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW       = os.path.join(BASE, 'data', 'raw') + os.sep
PROCESSED = os.path.join(BASE, 'data', 'processed') + os.sep
os.makedirs(PROCESSED, exist_ok=True)

# ── 1. Cargar archivos ────────────────────────────────────────────
print("Cargando archivos...")
operaciones = pd.read_excel(RAW + 'operaciones.xlsx')
clientes    = pd.read_excel(RAW + 'clientes.xlsx')
trm_usd     = pd.read_csv(RAW + 'trm_usd.csv', sep=None, engine='python')
trm_eur     = pd.read_excel(RAW + 'trm_eur.xlsx', skiprows=[1])
ciiu_desc   = pd.read_excel(RAW + 'ciiu.xlsx')
traders     = pd.read_excel(RAW + 'traiders.xlsx')

# ── 2. Limpiar fechas de operaciones ─────────────────────────────
operaciones['Fecha'] = (pd.to_timedelta(operaciones['Fecha'], unit='D')
                        + pd.Timestamp('1899-12-30'))

# ── 3. Limpiar TRM USD ───────────────────────────────────────────
def parse_numero_es(x):
    if pd.isna(x):
        return None
    x = str(x).strip().replace('$', '').replace(' ', '')
    if x in ('', '-', 'nan'):
        return None
    x = x.replace('.', '').replace(',', '.')
    return float(x)

trm_usd['VALOR']         = trm_usd['VALOR'].apply(parse_numero_es)
trm_usd['VIGENCIADESDE'] = pd.to_datetime(trm_usd['VIGENCIADESDE'], dayfirst=True)
trm_usd['VIGENCIAHASTA'] = pd.to_datetime(trm_usd['VIGENCIAHASTA'], dayfirst=True)

# Expandir TRM USD a una fila por día
filas_expandidas = []
for row in trm_usd.itertuples():
    fechas = pd.date_range(row.VIGENCIADESDE, row.VIGENCIAHASTA)
    filas_expandidas.append(pd.DataFrame({'Fecha': fechas, 'TRM_USD': row.VALOR}))
trm_usd_diario = pd.concat(filas_expandidas, ignore_index=True)

# Agregar variación 7 días para uso posterior
trm_usd_diario = trm_usd_diario.sort_values('Fecha').copy()
trm_usd_diario['variacion_trm_7d'] = (trm_usd_diario['TRM_USD']
                                       .pct_change(periods=7).round(6))

# ── 4. Limpiar TRM EUR ───────────────────────────────────────────
trm_eur = trm_eur[['Fecha', 'Euro - COP/EUR - Tasa media(Dato diario)']].copy()
trm_eur.columns = ['Fecha', 'TRM_EUR']
trm_eur['Fecha']   = pd.to_datetime(trm_eur['Fecha'], dayfirst=True, errors='coerce')
trm_eur['TRM_EUR'] = trm_eur['TRM_EUR'].apply(parse_numero_es)
trm_eur = trm_eur.dropna(subset=['Fecha'])

# ── 4b. Limpiar base de traders (NIT -> Trader asignado) ──────────
print("Limpiando base de traders...")
traders = traders.rename(columns={'ID': 'NIT'})
traders = traders[['NIT', 'Trader']].drop_duplicates(subset='NIT')
traders['Trader'] = traders['Trader'].replace({'Sin informacion': 'No asignado'})
print(f"  Clientes con trader: {traders['NIT'].nunique()}")
print(f"  Traders/canales únicos: {traders['Trader'].nunique()}")

# ── 5. Merge principal ───────────────────────────────────────────
print("Haciendo merge...")
df = operaciones.merge(clientes.drop(columns=['IDE']),
                       left_on='NIT', right_on='ID', how='left')
df = df.merge(ciiu_desc,
              left_on='CIIU_BUC', right_on='COD_ACT_CIIU_NOCLI', how='left')
df = df.merge(trm_usd_diario[['Fecha', 'TRM_USD']], on='Fecha', how='left')
df = df.merge(trm_eur, on='Fecha', how='left')
df = df.merge(traders, on='NIT', how='left')
df['Trader'] = df['Trader'].fillna('No asignado')

# ── 6. Limpieza de columnas duplicadas ───────────────────────────
df = df.drop(columns=['TipoIDC_y', 'ID', 'COD_ACT_CIIU_NOCLI'])
df = df.rename(columns={'TipoIDC_x': 'TipoIDC'})

df['TRM_aplicable'] = df.apply(
    lambda r: r['TRM_USD'] if 'USD' in str(r['Moneda'])
              else (r['TRM_EUR'] if 'EUR' in str(r['Moneda']) else None),
    axis=1
)

# ── 7. Validaciones de calidad ───────────────────────────────────
print("\n── Validaciones ──")
assert df['Segmento'].isna().mean() == 0,   "❌ Clientes sin match"
assert df['TRM_USD'].isna().mean()  == 0,   "❌ Fechas sin TRM USD"
print(f"Shape final:          {df.shape}")
print(f"Rango fechas:         {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")
print(f"Sin match cliente:    {df['Segmento'].isna().sum()}")
print(f"Sin TRM USD:          {df['TRM_USD'].isna().sum()}")
print(f"   Sin TRM aplicable:   {df['TRM_aplicable'].isna().sum()} "
      f"({df['TRM_aplicable'].isna().mean()*100:.2f}%) — monedas distintas USD/EUR")
print(f"Sin trader asignado:  {(df['Trader'] == 'No asignado').sum()} "
      f"({(df['Trader'] == 'No asignado').mean()*100:.2f}%)")

# ── 8. Guardar ───────────────────────────────────────────────────
out = PROCESSED + 'dataset_consolidado.parquet'
df.to_parquet(out, index=False)

# Guardar también trm_usd_diario para que lo usen los siguientes scripts
trm_usd_diario.to_parquet(PROCESSED + 'trm_usd_diario.parquet', index=False)

# Guardar tabla NIT -> Trader por separado, para que 02_modelo_timing.py
# la use al final sin tener que tocar el dataset_consolidado completo
traders.to_parquet(PROCESSED + 'traders.parquet', index=False)

print(f"\n Guardado: {out}")
print(" Guardado: ../data/processed/trm_usd_diario.parquet")
print(" Guardado: ../data/processed/traders.parquet")