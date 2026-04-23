"""
Urban Mobility Audit — Medellín
Writes all results to audit_results.txt (UTF-8).
"""

import pandas as pd
import numpy as np
import os
import sys

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\WinterOS\Documents\Dashboard\reto4"
MASTER = os.path.join(BASE, "data", "processed", "master.csv")
ANOM   = os.path.join(BASE, "data", "processed", "anomalies.csv")
OUT    = os.path.join(BASE, "audit_results.txt")

# ── helpers ───────────────────────────────────────────────────────────────────
SEP = "=" * 80

def hdr(title):
    return f"\n{SEP}\n  {title}\n{SEP}\n"

def fmt_df(df, float_fmt="{:,.2f}"):
    """Return a string table with formatted floats and integer thousands-sep."""
    df = df.copy()
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].map(lambda x: float_fmt.format(x) if pd.notna(x) else "N/A")
    for col in df.select_dtypes(include="int").columns:
        df[col] = df[col].map(lambda x: f"{x:,}")
    return df.to_string(index=False)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading master.csv …", flush=True)
df = pd.read_csv(MASTER, low_memory=False)
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns", flush=True)

results = []

# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL STATS — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
results.append(hdr("OVERVIEW — Basic Dataset Statistics"))

n_total       = len(df)
# fecha_trafico may have NaN floats mixed with strings — convert safely
df["fecha_trafico"] = df["fecha_trafico"].astype(str).replace("nan", pd.NA)
fecha_series  = df["fecha_trafico"].dropna()
fecha_min     = fecha_series.min() if len(fecha_series) else "N/A"
fecha_max     = fecha_series.max() if len(fecha_series) else "N/A"
n_corredores  = df["corredor"].nunique()
n_comunas     = df["nombre_comuna"].nunique()
n_weekend     = df["es_fin_semana"].sum()
n_weekday     = (~df["es_fin_semana"]).sum()
n_hora_pico   = df["es_hora_pico"].sum()

results.append(f"Total registros        : {n_total:,}")
results.append(f"Rango fechas           : {fecha_min}  →  {fecha_max}")
results.append(f"Corredores únicos      : {n_corredores}")
results.append(f"Comunas únicas         : {n_comunas}")
results.append(f"Registros entre semana : {n_weekday:,}  ({n_weekday/n_total*100:.1f}%)")
results.append(f"Registros fin de semana: {n_weekend:,}  ({n_weekend/n_total*100:.1f}%)")
results.append(f"Registros hora pico    : {n_hora_pico:,}  ({n_hora_pico/n_total*100:.1f}%)")

# ICV global summary
icv_stats = df["icv"].describe()
results.append(f"\nICV global  —  mean={icv_stats['mean']:.2f}  std={icv_stats['std']:.2f}"
               f"  min={icv_stats['min']:.2f}  p25={icv_stats['25%']:.2f}"
               f"  p50={icv_stats['50%']:.2f}  p75={icv_stats['75%']:.2f}"
               f"  max={icv_stats['max']:.2f}")

# Velocidad global summary
v_stats = df["velocidad_km_h"].describe()
results.append(f"Velocidad   —  mean={v_stats['mean']:.1f}  std={v_stats['std']:.1f}"
               f"  min={v_stats['min']:.0f}  max={v_stats['max']:.0f} km/h")

# ── Velocidad media por franja_horaria ────────────────────────────────────────
results.append(hdr("VELOCIDAD MEDIA POR FRANJA HORARIA"))
franja_vel = (df.groupby("franja_horaria", observed=True)["velocidad_km_h"]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "velocidad_media", "std": "velocidad_std", "count": "n_registros"})
               .sort_values("velocidad_media"))
franja_vel["icv_medio"] = df.groupby("franja_horaria", observed=True)["icv"].mean()
results.append(fmt_df(franja_vel.reset_index()))

# ── ICV por día de semana ─────────────────────────────────────────────────────
results.append(hdr("ICV MEDIO POR DIA DE SEMANA (dia_num 1=Lunes … 7=Domingo)"))
dia_labels = {1: "Lunes", 2: "Martes", 3: "Miercoles", 4: "Jueves",
              5: "Viernes", 6: "Sabado", 7: "Domingo"}
icv_dia = (df.groupby("dia_num")["icv"]
             .agg(["mean", "std", "count"])
             .rename(columns={"mean": "icv_medio", "std": "icv_std", "count": "n_registros"})
             .reset_index())
icv_dia["dia"] = icv_dia["dia_num"].map(dia_labels)
icv_dia["velocidad_media"] = df.groupby("dia_num")["velocidad_km_h"].mean().values
icv_dia = icv_dia[["dia_num", "dia", "icv_medio", "icv_std", "velocidad_media", "n_registros"]]
results.append(fmt_df(icv_dia))

# ── Top 5 horas más críticas ───────────────────────────────────────────────────
results.append(hdr("TOP 5 HORAS MAS CRITICAS (mayor ICV medio)"))
hora_icv = (df.groupby("hora")["icv"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "icv_medio", "count": "n_registros"})
              .sort_values("icv_medio", ascending=False)
              .head(5)
              .reset_index())
hora_icv["velocidad_media"] = df.groupby("hora")["velocidad_km_h"].mean().reindex(hora_icv["hora"]).values
hora_icv["intensidad_media"] = df.groupby("hora")["intensidad"].mean().reindex(hora_icv["hora"]).values
results.append(fmt_df(hora_icv))

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 1 — Top 10 corredores críticos (entre semana)
# ══════════════════════════════════════════════════════════════════════════════
results.append(hdr("PREGUNTA 1 — Top 10 Corredores Criticos (Entre Semana)"))
results.append("Filtro: es_fin_semana == False | Ranking: icv_medio DESC\n")

wd = df[~df["es_fin_semana"]].copy()

grp1 = wd.groupby("corredor", observed=True).agg(
    icv_medio        = ("icv",            "mean"),
    icv_maximo       = ("icv",            "max"),
    velocidad_media  = ("velocidad_km_h", "mean"),
    intensidad_media = ("intensidad",     "mean"),
    ocupacion_media  = ("ocupacion",      "mean"),
    n_registros      = ("icv",            "count"),
).reset_index()

# % de tiempo en hora pico
hp_frac = (wd.groupby("corredor", observed=True)["es_hora_pico"]
             .mean()
             .rename("pct_hora_pico") * 100)
grp1 = grp1.merge(hp_frac, on="corredor", how="left")

top10 = grp1.sort_values("icv_medio", ascending=False).head(10).reset_index(drop=True)
top10.index += 1   # rank starts at 1

results.append("Rank | Corredor | ICV_medio | ICV_max | Vel_media | Int_media | Ocup_media | n_reg | %HoraPico\n")
for rank, row in top10.iterrows():
    results.append(
        f"  {rank:2d}  {row['corredor']:<45s}  "
        f"ICV={row['icv_medio']:6.2f}  ICVmax={row['icv_maximo']:6.2f}  "
        f"vel={row['velocidad_media']:5.1f}km/h  int={row['intensidad_media']:7.1f}  "
        f"ocup={row['ocupacion_media']:5.1f}  n={row['n_registros']:,}  "
        f"pico={row['pct_hora_pico']:5.1f}%"
    )

results.append("\n--- Tabla completa ---")
results.append(fmt_df(top10.reset_index().rename(columns={"index": "rank"})))

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 2 — Comunas con mayor presión vehicular en hora pico
# ══════════════════════════════════════════════════════════════════════════════
results.append(hdr("PREGUNTA 2 — Comunas con Mayor Presion Vehicular en Hora Pico"))
results.append("Filtro: es_hora_pico == True | Ranking: icv_medio DESC (todas las comunas)\n")

hp = df[df["es_hora_pico"]].copy()

grp2 = hp.groupby("nombre_comuna", observed=True).agg(
    icv_medio        = ("icv",            "mean"),
    icv_maximo       = ("icv",            "max"),
    velocidad_media  = ("velocidad_km_h", "mean"),
    intensidad_media = ("intensidad",     "mean"),
    n_registros      = ("icv",            "count"),
    poblacion_2019   = ("poblacion_2019", "first"),
).reset_index()

# vehicles per 1000 inhabitants
grp2["vehiculos_por_1000hab"] = np.where(
    grp2["poblacion_2019"] > 0,
    grp2["intensidad_media"] / grp2["poblacion_2019"] * 1000,
    np.nan
)

grp2 = grp2.sort_values("icv_medio", ascending=False).reset_index(drop=True)
grp2.index += 1

results.append("Rank | Comuna | ICV_medio | ICV_max | Vel_media | Int_media | n_reg | Pob_2019 | Veh/1000hab\n")
for rank, row in grp2.iterrows():
    pob  = f"{row['poblacion_2019']:,.0f}" if pd.notna(row['poblacion_2019']) else "N/A"
    vph  = f"{row['vehiculos_por_1000hab']:,.2f}" if pd.notna(row['vehiculos_por_1000hab']) else "N/A"
    results.append(
        f"  {rank:2d}  {row['nombre_comuna']:<35s}  "
        f"ICV={row['icv_medio']:6.2f}  ICVmax={row['icv_maximo']:6.2f}  "
        f"vel={row['velocidad_media']:5.1f}km/h  int={row['intensidad_media']:7.1f}  "
        f"n={row['n_registros']:,}  pob={pob}  veh/1000={vph}"
    )

results.append("\n--- Tabla completa ---")
results.append(fmt_df(grp2.reset_index().rename(columns={"index": "rank"})))

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 3 — Zonas baja velocidad + alto flujo (hotspots)
# ══════════════════════════════════════════════════════════════════════════════
results.append(hdr("PREGUNTA 3 — Zonas Baja Velocidad + Alto Flujo (Hotspots)"))

p30_vel = df["velocidad_km_h"].quantile(0.30)
p70_int = df["intensidad"].quantile(0.70)

results.append(f"Umbral baja velocidad  : <= p30 = {p30_vel:.1f} km/h")
results.append(f"Umbral alto flujo      : >= p70 = {p70_int:.1f} veh/h\n")

mask_bv = df["velocidad_km_h"] <= p30_vel
mask_af = df["intensidad"]     >= p70_int
hotspots_df = df[mask_bv & mask_af].copy()

results.append(f"Registros que cumplen AMBAS condiciones: {len(hotspots_df):,}  "
               f"({len(hotspots_df)/len(df)*100:.2f}% del total)\n")

grp3 = hotspots_df.groupby(["corredor", "nombre_comuna"], observed=True).agg(
    n_eventos        = ("icv",            "count"),
    velocidad_media  = ("velocidad_km_h", "mean"),
    intensidad_media = ("intensidad",     "mean"),
    icv_medio        = ("icv",            "mean"),
    hora_pico_pct    = ("es_hora_pico",   "mean"),
).reset_index()
grp3["hora_pico_pct"] *= 100

top20_hs = grp3.sort_values("n_eventos", ascending=False).head(20).reset_index(drop=True)
top20_hs.index += 1

results.append("Top 20 Hotspots por numero de eventos:\n")
results.append("Rank | Corredor | Comuna | n_eventos | Vel_media | Int_media | ICV_medio | %HoraPico\n")
for rank, row in top20_hs.iterrows():
    results.append(
        f"  {rank:2d}  {row['corredor']:<45s}  {row['nombre_comuna']:<30s}  "
        f"n={row['n_eventos']:,}  vel={row['velocidad_media']:5.1f}km/h  "
        f"int={row['intensidad_media']:7.1f}  ICV={row['icv_medio']:6.2f}  "
        f"pico={row['hora_pico_pct']:5.1f}%"
    )

results.append("\n--- Tabla completa ---")
results.append(fmt_df(top20_hs.reset_index().rename(columns={"index": "rank"})))

# Distribution by franja_horaria
results.append("\n--- Distribucion de hotspot eventos por Franja Horaria ---")
franja_hs = (hotspots_df.groupby("franja_horaria", observed=True).agg(
    n_eventos        = ("icv",            "count"),
    velocidad_media  = ("velocidad_km_h", "mean"),
    intensidad_media = ("intensidad",     "mean"),
    icv_medio        = ("icv",            "mean"),
).reset_index()
.sort_values("n_eventos", ascending=False))
franja_hs["pct_del_total"] = franja_hs["n_eventos"] / franja_hs["n_eventos"].sum() * 100
results.append(fmt_df(franja_hs))

# ══════════════════════════════════════════════════════════════════════════════
# QUESTION 4 — Zonas prioritarias para intervención
# ══════════════════════════════════════════════════════════════════════════════
results.append(hdr("PREGUNTA 4 — Zonas Prioritarias para Intervencion"))

# Build a per-corredor summary from the full dataset (weekday)
p80_int = wd["intensidad"].quantile(0.80)
results.append(f"Umbral p80 intensidad (entre semana): {p80_int:.1f} veh/h")

# Use grp1 (weekday aggregates) enriched with nombre_comuna + hotspot flag
# We need nombre_comuna → take the most frequent value per corredor from weekday data
def safe_mode(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) > 0 else np.nan

comuna_map = (wd.groupby("corredor", observed=True)["nombre_comuna"].agg(safe_mode))
pop_map    = (wd.groupby("corredor", observed=True)["poblacion_2019"]
                .agg(lambda s: s.dropna().mean()))

q4 = grp1.copy()   # already computed for weekdays
q4["nombre_comuna"]  = q4["corredor"].map(comuna_map)
q4["poblacion_2019"] = q4["corredor"].map(pop_map)

# alta poblacion: top tercile
p67_pop = q4["poblacion_2019"].quantile(0.67)
results.append(f"Umbral alta poblacion (p67): {p67_pop:,.0f} hab")

hotspot_corredores = set(grp3["corredor"].unique())

cat_a = (q4["icv_medio"] >= 50) & (q4["velocidad_media"] < 25)
cat_b = (q4["intensidad_media"] >= p80_int) & (q4["icv_medio"] >= 45)
cat_c = q4["corredor"].isin(hotspot_corredores) & (q4["poblacion_2019"] >= p67_pop)
n_cats = cat_a.astype(int) + cat_b.astype(int) + cat_c.astype(int)

cat_multi = n_cats >= 2

q4["cat_semaforos"]    = cat_a & ~cat_multi
q4["cat_rutas"]        = cat_b & ~cat_multi
q4["cat_sostenible"]   = cat_c & ~cat_multi
q4["cat_multiple"]     = cat_multi

# A corridor can be in "multiple" and also flagged; exclusive assignment:
# priority: multiple > semaforos > rutas > sostenible
def assign_cat(row):
    nc = (int(row["icv_medio"] >= 50 and row["velocidad_media"] < 25) +
          int(row["intensidad_media"] >= p80_int and row["icv_medio"] >= 45) +
          int(row["corredor"] in hotspot_corredores and row["poblacion_2019"] >= p67_pop))
    if nc >= 2:
        return "Multiples Intervenciones"
    elif row["icv_medio"] >= 50 and row["velocidad_media"] < 25:
        return "Gestion Semafórica"
    elif row["intensidad_media"] >= p80_int and row["icv_medio"] >= 45:
        return "Rutas Alternas"
    elif row["corredor"] in hotspot_corredores and row["poblacion_2019"] >= p67_pop:
        return "Transporte Sostenible"
    else:
        return "Sin clasificar"

q4["intervencion"] = q4.apply(assign_cat, axis=1)

cat_counts = q4["intervencion"].value_counts()
results.append(f"\nResumen de categorias de intervencion:\n")
for cat, cnt in cat_counts.items():
    results.append(f"  {cat:<30s}: {cnt:>4d} corredores  ({cnt/len(q4)*100:.1f}%)")

for cat in ["Gestion Semafórica", "Rutas Alternas", "Transporte Sostenible", "Multiples Intervenciones"]:
    subset = q4[q4["intervencion"] == cat].sort_values("icv_medio", ascending=False)
    results.append(f"\n--- {cat} ({len(subset)} corredores) ---")
    results.append(f"  {'Corredor':<45s}  {'ICV_med':>7s}  {'Vel_med':>7s}  {'Int_med':>8s}  {'Pob_2019':>10s}  {'%HoraPico':>9s}")
    for _, row in subset.iterrows():
        pob = f"{row['poblacion_2019']:>10,.0f}" if pd.notna(row['poblacion_2019']) else "       N/A"
        results.append(
            f"  {row['corredor']:<45s}  {row['icv_medio']:>7.2f}  {row['velocidad_media']:>7.1f}  "
            f"{row['intensidad_media']:>8.1f}  {pob}  {row['pct_hora_pico']:>9.1f}%"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
results.append(hdr("ANALISIS DE ANOMALIAS (anomalies.csv)"))

if os.path.exists(ANOM):
    print("Loading anomalies.csv …", flush=True)
    adf = pd.read_csv(ANOM, low_memory=False)
    print(f"  Loaded {len(adf):,} rows", flush=True)

    results.append(f"Total registros con anomalia: {len(adf):,}")

    # Severity breakdown
    if "severity_level" in adf.columns:
        sev = adf["severity_level"].value_counts()
        results.append("\nSeverity breakdown:")
        for sev_label, cnt in sev.items():
            results.append(f"  {sev_label:<15s}: {cnt:,}  ({cnt/len(adf)*100:.1f}%)")

    # Anomaly type breakdown
    anom_cols = ["z_score_anomaly", "iqr_anomaly", "speed_drop_anomaly",
                 "isolation_forest_anomaly", "dbscan_anomaly"]
    existing_anom_cols = [c for c in anom_cols if c in adf.columns]
    if existing_anom_cols:
        results.append("\nAnomaly type counts:")
        for col in existing_anom_cols:
            cnt = adf[col].sum()
            results.append(f"  {col:<30s}: {cnt:,}  ({cnt/len(adf)*100:.1f}%)")

    # Top corredores by anomaly count
    results.append("\nTop 20 Corredores por numero de anomalias:")
    top_corr_anom = (adf.groupby("corredor", observed=True)
                       .agg(
                           n_anomalias       = ("corredor",       "count"),
                           icv_medio         = ("icv",            "mean"),
                           velocidad_media   = ("velocidad_km_h", "mean"),
                           intensidad_media  = ("intensidad",     "mean"),
                           anomaly_score_med = ("anomaly_score",  "mean"),
                       )
                       .sort_values("n_anomalias", ascending=False)
                       .head(20)
                       .reset_index())
    if "severity_level" in adf.columns:
        sev_map = (adf.groupby(["corredor", "severity_level"], observed=True)
                     .size()
                     .unstack(fill_value=0))
        top_corr_anom = top_corr_anom.merge(sev_map, on="corredor", how="left")

    results.append(fmt_df(top_corr_anom))

    # Top comunas by anomaly count
    results.append("\nTop Comunas por numero de anomalias:")
    top_com_anom = (adf.groupby("nombre_comuna", observed=True)
                      .agg(
                          n_anomalias      = ("nombre_comuna", "count"),
                          icv_medio        = ("icv",           "mean"),
                          velocidad_media  = ("velocidad_km_h","mean"),
                      )
                      .sort_values("n_anomalias", ascending=False)
                      .reset_index())
    results.append(fmt_df(top_com_anom))

    # Franja horaria de anomalias
    results.append("\nDistribucion de anomalias por Franja Horaria:")
    fh_anom = (adf.groupby("franja_horaria", observed=True)
                  .agg(n_anomalias=("corredor", "count"),
                       icv_medio=("icv", "mean"))
                  .sort_values("n_anomalias", ascending=False)
                  .reset_index())
    results.append(fmt_df(fh_anom))

else:
    results.append("anomalies.csv no encontrado en data/processed/")

# ══════════════════════════════════════════════════════════════════════════════
# WRITE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
header_block = [
    "=" * 80,
    " AUDITORIA CUANTITATIVA — MOVILIDAD URBANA MEDELLIN",
    f" Fecha de ejecucion : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f" Archivo fuente     : {MASTER}",
    f" Total registros    : {n_total:,}",
    "=" * 80,
]

full_output = "\n".join(header_block) + "\n" + "\n".join(results)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\nOutput written to: {OUT}")
print(f"File size: {os.path.getsize(OUT):,} bytes")
print("AUDIT COMPLETE")
