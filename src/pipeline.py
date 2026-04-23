"""
pipeline.py — ETL + features + ICV + rankings + hotspots para Movilidad Medellín.

Única función pública: run(config) -> dict
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from unidecode import unidecode

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CORRECCIONES DE CARACTERES UNICODE CORRUPTOS
# El archivo fuente velocidad_*.csv contiene bytes U+FFFD (EF BF BD)
# en lugar de caracteres acentuados.  Cada acento → 1-3 grupos de FFFD.
# Estas correcciones se aplican ANTES de eliminar los FFFD residuales.
# ═══════════════════════════════════════════════════════════════════

_FFFD = "\uFFFD"

# Lista de (patrón regex, reemplazo correcto)
_UNICODE_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(_FFFD + r"+a\b"),            "ía"),        # Vía  → V + ía
    (re.compile(r"\bV" + _FFFD + r"+a"),    "Vía"),       # Vía  (contexto completo)
    (re.compile(r"T" + _FFFD + r"+nel"),    "Túnel"),     # Túnel
    (re.compile(r"Crist" + _FFFD + r"+bal"),"Cristóbal"), # Cristóbal
    (re.compile(r"r" + _FFFD + r"+o"),      "río"),       # río
    (re.compile(_FFFD + r"+XITO"),          "ÉXITO"),     # ÉXITO
    (re.compile(r"DO" + _FFFD + r"+A"),     "DOÑA"),      # DOÑA
]


def _fix_unicode_corruption(text: str) -> str:
    """Restaura caracteres acentuados que fueron convertidos a U+FFFD."""
    if _FFFD not in text:
        return text
    for pattern, replacement in _UNICODE_FIXES:
        text = pattern.sub(replacement, text)
    # Eliminar cualquier FFFD residual tras las correcciones conocidas
    text = text.replace(_FFFD, "")
    return text


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CARGA
# ═══════════════════════════════════════════════════════════════════

_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]


def _safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {path} con ningún encoding conocido.")


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 2 — LIMPIEZA POR DATASET
# ═══════════════════════════════════════════════════════════════════


def _to_snake(col: str) -> str:
    """snake_case ASCII sin tildes."""
    col = unidecode(str(col)).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake(c) for c in df.columns]
    return df


def _clean_velocidad(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # Parsear fechas
    for col in ("fecha_trafico", "fecha"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Coordenadas proyectadas MAGNA-SIRGAS: coma decimal → punto
    for col in ("longitud", "latitud"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Numéricos
    for col in ("velocidad_km_h", "intensidad", "ocupacion", "hora", "dia_num", "mes_num", "ano"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "hora" in df.columns:
        # hora puede venir como "14:00:00" → extraer primer token
        if df["hora"].dtype == object:
            df["hora"] = df["hora"].astype(str).str.split(":").str[0]
            df["hora"] = pd.to_numeric(df["hora"], errors="coerce")

    # Eliminar duplicados exactos
    df.drop_duplicates(inplace=True)

    # Eliminar filas con valores de velocidad fuera de rango físico (notebook: 1-100)
    df = df.dropna(subset=["velocidad_km_h", "intensidad", "hora", "corredor"])
    df = df[
        (df["velocidad_km_h"] >= 1) & (df["velocidad_km_h"] <= 100) &
        (df["intensidad"] >= 0) &
        (df["hora"] >= 0) & (df["hora"] <= 23)
    ]

    # Limpiar strings y corregir caracteres U+FFFD corruptos
    for col in ("corredor", "nombre_comuna", "carril"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().apply(_fix_unicode_corruption)

    return df.reset_index(drop=True)


def _clean_aforos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # Remover comillas escapadas (artefactos Excel)
    df = df.replace(r'^\s*"*\s*$', np.nan, regex=True)
    df = df.replace('""', np.nan)

    # Parsear fecha desde formato DDMMYY en columna 'fecha'
    if "fecha" in df.columns:
        def _parse_ddmmyy(val):
            s = str(val).strip().zfill(6)
            try:
                return pd.to_datetime(s, format="%d%m%y", errors="coerce")
            except Exception:
                return pd.NaT
        df["fecha"] = df["fecha"].apply(_parse_ddmmyy)

    # Renombrar coordenadas a lon/lat (ya son WGS84)
    if "coordenadax" in df.columns:
        df.rename(columns={"coordenadax": "lon"}, inplace=True)
    if "coordenaday" in df.columns:
        df.rename(columns={"coordenaday": "lat"}, inplace=True)

    for col in ("lon", "lat"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Columnas de conteo vehicular → int
    count_cols = [
        "autos", "buses", "camiones", "motos", "bicicletas",
        "autos_hora", "buses_hora", "camiones_hora", "motos_hora",
        "bicicletas_hora", "volumen_total_hora", "vehiculo_equivalente",
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "nombre_comuna" in df.columns:
        df["nombre_comuna"] = df["nombre_comuna"].astype(str).str.strip()

    return df


def _clean_simm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # Parsear campo location (JSON-string) → lat, lon
    if "location" in df.columns:
        def _parse_loc(loc):
            if pd.isna(loc):
                return np.nan, np.nan
            try:
                lon_m = re.search(r"'lon'\s*:\s*'([^']+)'", str(loc))
                lat_m = re.search(r"'lat'\s*:\s*'([^']+)'", str(loc))
                lon = float(lon_m.group(1)) if lon_m else np.nan
                lat = float(lat_m.group(1)) if lat_m else np.nan
                return lon, lat
            except Exception:
                return np.nan, np.nan

        coords = df["location"].apply(_parse_loc)
        df["lon"] = coords.apply(lambda t: t[0])
        df["lat"] = coords.apply(lambda t: t[1])

    # Datetime UTC → naive
    if "fechahora" in df.columns:
        df["fechahora"] = pd.to_datetime(df["fechahora"], utc=True, errors="coerce")
        df["fechahora"] = df["fechahora"].dt.tz_localize(None)

    # Renombrar columnas con notación punto → snake_case (ya manejado por _norm_cols)
    # Numéricos
    for col in ("velocidad", "intensidad", "ocupacion", "headway", "registros"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "corredor" in df.columns:
        df["corredor"] = df["corredor"].astype(str).str.strip()

    return df


def _clean_pasajeros(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # Detectar columna año (puede aparecer con artefacto de encoding)
    year_col = next(
        (c for c in df.columns
         if 1 <= len(c) <= 6 and "a" in c and "o" in c and c not in ("orden",)),
        None,
    )
    if year_col and year_col != "ano":
        df.rename(columns={year_col: "ano"}, inplace=True)

    for col in ("ano", "semestre", "num_mes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Columnas de pasajeros: remover separador de miles (punto español)
    pax_cols = [c for c in df.columns if "pax_mov" in c]

    def _rm_thousands(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace('"', "").replace("'", "")
        if "," in s:
            # formato decimal español: 492,314 → 492.314
            s = s.replace(".", "").replace(",", ".")
        elif "." in s:
            parts = s.split(".")
            # punto = miles si hay exactamente 3 dígitos tras el punto
            if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isdigit():
                s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return np.nan

    for col in pax_cols:
        df[col] = df[col].apply(_rm_thousands)

    # Total pasajeros por fila
    df["total_pax"] = df[pax_cols].sum(axis=1, min_count=1)

    # Eliminar filas con valores negativos en pasajeros
    df = df[df["total_pax"].isna() | (df["total_pax"] >= 0)]

    # Construir columna de fecha (año + mes donde exista)
    if "ano" in df.columns:
        if "num_mes" in df.columns:
            df["fecha_periodo"] = pd.to_datetime(
                df["ano"].astype(str).str.zfill(4) + "-" +
                df["num_mes"].fillna(1).astype(int).astype(str).str.zfill(2) + "-01",
                errors="coerce",
            )
        else:
            df["fecha_periodo"] = pd.to_datetime(df["ano"].astype(str) + "-01-01", errors="coerce")

    return df


def _clean_poblacion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)

    # Eliminar filas agregadas
    if "grupo_edad" in df.columns:
        df["grupo_edad"] = df["grupo_edad"].astype(str).str.strip()
        df = df[df["grupo_edad"].str.lower() == "total"].copy()

    # Eliminar filas de totales
    if "codigo" in df.columns:
        mask_agg = df["codigo"].astype(str).str.strip().isin(
            ["Suma Comunas", "Total Medellín", "Total", "Suma Corregimientos"]
        )
        df = df[~mask_agg].copy()
        df["nombre_area"] = df["codigo"].astype(str).str.strip()

    for col in ("total_2019", "hombres_2019", "mujeres_2019"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["total_2019"])
    return df.reset_index(drop=True)


def _clean_ciclorrutas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _norm_cols(df)
    for col in ("nombre", "estado"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════


def _build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Agrega features temporales sobre el dataframe velocidad limpio."""
    df = df.copy()

    # hora: ya existe como int en el dataset
    if "hora" in df.columns:
        df["hora"] = df["hora"].astype(int)

    # dia_num: ya existe. 1=Lun … 7=Dom (dataset Medellín)
    if "dia_num" in df.columns:
        df["dia_num"] = pd.to_numeric(df["dia_num"], errors="coerce").fillna(0).astype(int)
        df["es_fin_semana"] = df["dia_num"] >= 6  # Sáb=6, Dom=7

    # mes_num: ya existe
    if "mes_num" in df.columns:
        df["mes_num"] = pd.to_numeric(df["mes_num"], errors="coerce").fillna(0).astype(int)

    # franja_horaria desde config
    franjas: dict = config["eta"]["franjas"]
    hora_a_franja: dict = {}
    for nombre_franja, horas in franjas.items():
        for h in horas:
            hora_a_franja[h] = nombre_franja

    def _get_franja(h):
        try:
            return hora_a_franja.get(int(h), "desconocida")
        except (TypeError, ValueError):
            return "desconocida"

    if "hora" in df.columns:
        df["franja_horaria"] = df["hora"].apply(_get_franja)

    # es_hora_pico
    pico = set(config["eta"]["hora_pico_manana"] + config["eta"]["hora_pico_tarde"])
    if "hora" in df.columns:
        df["es_hora_pico"] = df["hora"].isin(pico)

    return df


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 4 — INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════


def _norm_name(name: str) -> str:
    """Normaliza nombre de área: lower, sin tildes, sin guiones, sin espacios extra."""
    if pd.isna(name):
        return ""
    s = unidecode(str(name)).lower()
    s = re.sub(r"[-–—]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_master(
    velocidad: pd.DataFrame,
    aforos: pd.DataFrame,
    simm: pd.DataFrame,
    pasajeros: pd.DataFrame,
    poblacion: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    master = _build_features(velocidad, config)

    # ── Join con poblacion por nombre_comuna (fuzzy: lower, sin tildes, sin guiones) ──
    if not poblacion.empty and "nombre_area" in poblacion.columns and "total_2019" in poblacion.columns:
        pop_map = {_norm_name(k): v for k, v in
                   zip(poblacion["nombre_area"], poblacion["total_2019"])}

        def _lookup_pop(name):
            key = _norm_name(name)
            if key in pop_map:
                return pop_map[key]
            # Búsqueda parcial
            for k, v in pop_map.items():
                if key and k and (key in k or k in key):
                    return v
            return np.nan

        if "nombre_comuna" in master.columns:
            master["poblacion_2019"] = master["nombre_comuna"].apply(_lookup_pop)
        else:
            master["poblacion_2019"] = np.nan
    else:
        master["poblacion_2019"] = np.nan

    # ── Join con aforos agregado (mean volumen_total_hora por nombre_comuna) ──
    if not aforos.empty and "nombre_comuna" in aforos.columns and "volumen_total_hora" in aforos.columns:
        aforos_agg = (
            aforos.groupby("nombre_comuna")["volumen_total_hora"]
            .mean()
            .reset_index()
            .rename(columns={"volumen_total_hora": "aforo_medio"})
        )
        aforos_agg["_key"] = aforos_agg["nombre_comuna"].apply(_norm_name)
        master["_key"] = master["nombre_comuna"].apply(_norm_name) if "nombre_comuna" in master.columns else ""
        master = master.merge(aforos_agg[["_key", "aforo_medio"]], on="_key", how="left")
        master.drop(columns=["_key"], inplace=True)
    else:
        master["aforo_medio"] = np.nan

    # ── Join con simm agregado (mean velocidad e intensidad por corredor) ──
    if not simm.empty and "corredor" in simm.columns:
        agg_cols = {"velocidad": "simm_vel_media", "intensidad": "simm_int_media"}
        simm_agg_map = {}
        for src, dst in agg_cols.items():
            if src in simm.columns:
                simm_agg_map[src] = dst

        if simm_agg_map:
            simm_agg = (
                simm.groupby("corredor")
                .agg(**{v: (k, "mean") for k, v in simm_agg_map.items()})
                .reset_index()
            )
            # Añadir lat/lon WGS84 de SIMM para el mapa del dashboard
            if "lat" in simm.columns and "lon" in simm.columns:
                simm_coords = (
                    simm.groupby("corredor")[["lat", "lon"]]
                    .mean()
                    .rename(columns={"lat": "simm_lat", "lon": "simm_lon"})
                    .reset_index()
                )
                simm_agg = simm_agg.merge(simm_coords, on="corredor", how="left")

            master = master.merge(simm_agg, on="corredor", how="left")
    else:
        for col in ("simm_vel_media", "simm_int_media", "simm_lat", "simm_lon"):
            master[col] = np.nan

    return master.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 5 — ICV
# ═══════════════════════════════════════════════════════════════════


def _compute_icv(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["icv"]
    w_vel = cfg["peso_velocidad"]    # 0.40
    w_int = cfg["peso_intensidad"]   # 0.35
    w_ocu = cfg["peso_ocupacion"]    # 0.25
    w_pop = cfg["peso_poblacion"]    # 0.15
    escala = cfg["escala"]           # 100

    df = df.copy()

    def _minmax(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    n_vel = _minmax(df["velocidad_km_h"])

    n_int = _minmax(df["intensidad"])

    if "ocupacion" in df.columns:
        ocu = df["ocupacion"].fillna(df["ocupacion"].median() if df["ocupacion"].notna().any() else 0)
        n_ocu = _minmax(ocu)
    else:
        n_ocu = pd.Series(0.0, index=df.index)

    if "poblacion_2019" in df.columns:
        pop = df["poblacion_2019"].fillna(df["poblacion_2019"].median() if df["poblacion_2019"].notna().any() else 0)
        n_pop = _minmax(pop)
    else:
        n_pop = pd.Series(0.0, index=df.index)

    icv_base = w_vel * (1 - n_vel) + w_int * n_int + w_ocu * n_ocu
    amplificador = 1 + w_pop * n_pop
    df["icv"] = (escala * icv_base * amplificador).clip(0, escala).round(2)

    return df


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN 6 — RANKINGS Y HOTSPOTS
# ═══════════════════════════════════════════════════════════════════


def _rank_corredores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["rankings"]
    if cfg["solo_dias_habiles"] and "es_fin_semana" in df.columns:
        df = df[~df["es_fin_semana"]]

    if df.empty or "corredor" not in df.columns:
        return pd.DataFrame()

    grp = (
        df.groupby("corredor")
        .agg(
            icv_medio=("icv", "mean"),
            icv_maximo=("icv", "max"),
            n_registros=("icv", "count"),
        )
        .reset_index()
        .nlargest(cfg["top_corredores"], "icv_medio")
        .reset_index(drop=True)
    )
    grp["rank"] = range(1, len(grp) + 1)
    return grp[["rank", "corredor", "icv_medio", "icv_maximo", "n_registros"]]


def _clean_corridor_label(value: object) -> str:
    if pd.isna(value):
        return "DESCONOCIDO"

    text = str(value).lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"^[\d\s\-.]+", "", text)
    return text.strip().upper() or "DESCONOCIDO"


def _rank_corredores_ivc(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config.get("ivc", {})
    top_n = cfg.get("top_corredores", 15)
    capacidad_base = cfg.get("capacidad_base", 2500)
    velocidad_libre = cfg.get("velocidad_libre", 60)

    required_cols = {"corredor", "intensidad", "velocidad_km_h"}
    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df_ivc = df.copy()
    df_ivc["intensidad"] = pd.to_numeric(df_ivc["intensidad"], errors="coerce")
    df_ivc["velocidad_km_h"] = pd.to_numeric(df_ivc["velocidad_km_h"], errors="coerce")
    df_ivc = df_ivc.dropna(subset=["corredor", "intensidad", "velocidad_km_h"])
    df_ivc = df_ivc[
        (df_ivc["intensidad"] >= 0)
        & (df_ivc["velocidad_km_h"] > 0)
        & (df_ivc["velocidad_km_h"] < 120)
    ].copy()

    if df_ivc.empty:
        return pd.DataFrame()

    df_ivc["corredor"] = df_ivc["corredor"].apply(_clean_corridor_label)
    df_ivc["ivc"] = df_ivc["intensidad"] / capacidad_base
    df_ivc["icv_simple"] = (1 - (df_ivc["velocidad_km_h"] / velocidad_libre)).clip(0, 1)

    grp = (
        df_ivc.groupby("corredor", as_index=False)
        .agg(
            intensidad_media=("intensidad", "mean"),
            velocidad_media=("velocidad_km_h", "mean"),
            ivc_medio=("ivc", "mean"),
            icv_simple_medio=("icv_simple", "mean"),
            n_registros=("ivc", "count"),
        )
        .sort_values(["ivc_medio", "corredor"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    grp["rank"] = range(1, len(grp) + 1)
    return grp[
        [
            "rank",
            "corredor",
            "ivc_medio",
            "velocidad_media",
            "intensidad_media",
            "icv_simple_medio",
            "n_registros",
        ]
    ]


def _rank_comunas(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["rankings"]
    if "es_hora_pico" in df.columns:
        df = df[df["es_hora_pico"]]

    if df.empty or "nombre_comuna" not in df.columns:
        return pd.DataFrame()

    grp = (
        df.groupby("nombre_comuna")
        .agg(icv_medio=("icv", "mean"))
        .reset_index()
        .nlargest(cfg["top_comunas"], "icv_medio")
        .reset_index(drop=True)
    )
    grp["rank"] = range(1, len(grp) + 1)
    return grp[["rank", "nombre_comuna", "icv_medio"]]


def _identify_hotspots(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["rankings"]
    vel_pct = cfg["hotspot_velocidad_pct"] / 100  # 0.30
    int_pct = cfg["hotspot_intensidad_pct"] / 100  # 0.70

    vel_thr = df["velocidad_km_h"].quantile(vel_pct)
    int_thr = df["intensidad"].quantile(int_pct)

    return df[
        (df["velocidad_km_h"] <= vel_thr) & (df["intensidad"] >= int_thr)
    ].copy()


# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN PÚBLICA
# ═══════════════════════════════════════════════════════════════════


def run(config: dict) -> dict:
    """
    Ejecuta el pipeline completo y retorna todos los dataframes procesados.

    Retorna:
        master      — DataFrame completo con ICV (row-level)
        corredores  — Ranking top corredores críticos
        comunas     — Ranking top comunas en hora pico
        hotspots    — Zonas con velocidad <= P30 e intensidad >= P70
        pasajeros   — Dataset de pasajeros limpio
        ciclorrutas — Dataset de ciclorrutas limpio
    """
    raw_dir = Path(config["paths"]["raw"])
    proc_dir = Path(config["paths"]["processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── Carga ────────────────────────────────────────────────────────
    def _load(key: str) -> pd.DataFrame:
        path = raw_dir / config["files"][key]
        if not path.exists():
            print(f"[WARN] No encontrado: {path}")
            return pd.DataFrame()
        df = _safe_read_csv(str(path))
        print(f"  [{key}] cargado: {df.shape}")
        return df

    print("=== CARGA ===")
    raw_vel = _load("velocidad")
    raw_aforos = _load("aforos")
    raw_simm = _load("simm")
    raw_pax = _load("pasajeros")
    raw_pop = _load("poblacion")
    raw_cic = _load("ciclorrutas")

    # ── ETL ──────────────────────────────────────────────────────────
    print("=== ETL ===")
    vel = _clean_velocidad(raw_vel)
    aforos = _clean_aforos(raw_aforos)
    simm = _clean_simm(raw_simm)
    pax = _clean_pasajeros(raw_pax)
    pop = _clean_poblacion(raw_pop)
    cic = _clean_ciclorrutas(raw_cic)

    print(f"  velocidad limpio: {vel.shape}")
    print(f"  aforos limpio:    {aforos.shape}")
    print(f"  simm limpio:      {simm.shape}")
    print(f"  pasajeros limpio: {pax.shape}")
    print(f"  poblacion limpio: {pop.shape}")
    print(f"  ciclorrutas:      {cic.shape}")

    # ── Integración ──────────────────────────────────────────────────
    print("=== INTEGRACION ===")
    master = _build_master(vel, aforos, simm, pax, pop, config)

    # ── ICV ──────────────────────────────────────────────────────────
    print("=== ICV ===")
    master = _compute_icv(master, config)
    print(f"  ICV calculado. master shape: {master.shape}")
    print(f"  ICV stats: min={master['icv'].min():.1f} | mean={master['icv'].mean():.1f} | max={master['icv'].max():.1f}")

    # ── Rankings ─────────────────────────────────────────────────────
    print("=== RANKINGS ===")
    corredores = _rank_corredores(master, config)
    corredores_ivc = _rank_corredores_ivc(master, config)
    comunas = _rank_comunas(master, config)
    hotspots = _identify_hotspots(master, config)

    print(f"  Top corredores: {len(corredores)}")
    print(f"  Top corredores IVC: {len(corredores_ivc)}")
    print(f"  Top comunas:    {len(comunas)}")
    print(f"  Hotspots:       {len(hotspots)}")

    # ── Persistencia ─────────────────────────────────────────────────
    master_path = proc_dir / "master.csv"
    master.to_csv(master_path, index=False)
    print(f"  master.csv guardado: {master_path}")

    pax_path = proc_dir / "pasajeros_clean.csv"
    pax.to_csv(pax_path, index=False)

    # ── Detección de anomalías ────────────────────────────────────────
    print("=== DETECCIÓN DE ANOMALÍAS ===")
    anomaly_results: dict = {}
    try:
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector(config=config)
        anomaly_results = detector.run(master)
        stats = anomaly_results.get("summary_stats", {})
        print(f"  Anomalías detectadas: {stats.get('total_anomalias', 0):,}")
        print(f"  Porcentaje anómalo:   {stats.get('pct_anomalos', 0):.1f}%")
        print(f"  CRÍTICO: {stats.get('por_severidad', {}).get('CRÍTICO', 0):,} | "
              f"ALTO: {stats.get('por_severidad', {}).get('ALTO', 0):,} | "
              f"MODERADO: {stats.get('por_severidad', {}).get('MODERADO', 0):,}")
        print(f"  Modelos IF guardados en: {anomaly_results.get('models_path', 'N/D')}")
    except Exception as exc:
        print(f"  [WARN] Detección de anomalías omitida: {exc}")

    return {
        "master": master,
        "corredores": corredores,
        "corredores_ivc": corredores_ivc,
        "comunas": comunas,
        "hotspots": hotspots,
        "pasajeros": pax,
        "ciclorrutas": cic,
        "anomalias": anomaly_results,
    }
