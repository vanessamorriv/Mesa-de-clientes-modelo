import pandas as pd
import os
from supabase import create_client

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW  = os.path.join(BASE, '..', 'data', 'raw') + os.sep

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_KEY']
supabase     = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 1. Subir CIIU ─────────────────────────────────────────────────
print("Subiendo CIIU...")
df_ciiu = pd.read_excel(DATA_RAW + 'ciiu.xlsx')
df_ciiu.columns = [c.strip().lower() for c in df_ciiu.columns]
df_ciiu = df_ciiu.rename(columns={'cod_act_ciiu_nocli': 'cod_act_ciiu_nocli', 'des_ciiu': 'des_ciiu'})
df_ciiu['cod_act_ciiu_nocli'] = df_ciiu['cod_act_ciiu_nocli'].astype(str).str.strip()
df_ciiu = df_ciiu.dropna(subset=['cod_act_ciiu_nocli'])

data = df_ciiu.to_dict('records')
for i in range(0, len(data), 500):
    supabase.table('ciiu').upsert(data[i:i+500], on_conflict='cod_act_ciiu_nocli').execute()
    print(f"  CIIU lote {i}–{i+500}")
print(f"  Total CIIU: {len(data)} registros")

# ── 2. Subir CLIENTES ─────────────────────────────────────────────
print("\nSubiendo clientes...")
df_clientes = pd.read_excel(DATA_RAW + 'clientes.xlsx')

# Renombrar columnas explícitamente al formato de la tabla
df_clientes = df_clientes.rename(columns={
    'ID':          'id',
    'TipoID':      'tipo_id',
    'TipoIDC':     'tipo_idc',
    'Segmento':    'segmento',
    'Subsegmento': 'subsegmento',
    'Cod_Cartera': 'cod_cartera',
    'CIIU_BUC':    'ciiu_buc',
    'IDE':         'ide',
})

df_clientes['id'] = df_clientes['id'].astype(str).str.strip()
df_clientes = df_clientes.dropna(subset=['id'])
df_clientes = df_clientes.where(pd.notna(df_clientes), None)

data = df_clientes.to_dict('records')
for i in range(0, len(data), 500):
    supabase.table('clientes').upsert(data[i:i+500], on_conflict='id').execute()
    print(f"  Clientes lote {i}–{i+500}")
print(f"  Total clientes: {len(data)} registros")

# Reemplazar NaN por None para que Supabase los acepte
df_clientes = df_clientes.where(pd.notna(df_clientes), None)

data = df_clientes.to_dict('records')
for i in range(0, len(data), 500):
    supabase.table('clientes').upsert(data[i:i+500], on_conflict='id').execute()
    print(f"  Clientes lote {i}–{i+500}")
print(f"  Total clientes: {len(data)} registros")

# ── 3. Subir OPERACIONES ──────────────────────────────────────────
print("\nSubiendo operaciones...")
df_ops = pd.read_excel(DATA_RAW + 'operaciones.xlsx')

df_ops = df_ops.rename(columns={
    'Fecha':         'fecha',
    'TipoIDC':       'tipo_idc',
    'NIT':           'nit',
    'Producto':      'producto',
    'Lado':          'lado',
    'Entidad':       'entidad',
    'Moneda':        'moneda',
    'Monto_Total_':  'monto_total',
    'Monto_Entidad': 'monto_entidad',
    'Monto_Mercado': 'monto_mercado',
})

# Convertir fecha
if pd.api.types.is_numeric_dtype(df_ops['fecha']):
    df_ops['fecha'] = pd.to_datetime(
        df_ops['fecha'], origin='1899-12-30', unit='D', errors='coerce'
    )
else:
    df_ops['fecha'] = pd.to_datetime(df_ops['fecha'], errors='coerce')

df_ops['fecha'] = df_ops['fecha'].dt.strftime('%Y-%m-%d')
df_ops['nit']   = df_ops['nit'].astype(str).str.strip()
df_ops = df_ops.dropna(subset=['fecha', 'nit'])
df_ops = df_ops.where(pd.notna(df_ops), None)

data = df_ops.to_dict('records')
for i in range(0, len(data), 500):
    supabase.table('operaciones').upsert(
        data[i:i+500],
        on_conflict='fecha,nit,producto,lado,entidad,moneda,monto_total'
    ).execute()
    print(f"  Operaciones lote {i}–{i+500}")
print(f"  Total operaciones: {len(data)} registros")

print("\n¡Carga inicial completa!")