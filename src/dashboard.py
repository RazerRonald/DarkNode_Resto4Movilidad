"""
dashboard.py - Dashboard interactivo de Movilidad Medellin (Streamlit).

7 secciones: KPIs, Top Corredores, Comunas, Hotspots,
Patrones Temporales, Tendencia Metro y Mapa de Calor Vial.

Carga desde data/processed/master.csv si existe;
si no, ejecuta pipeline.run() para generarlo.
"""

from __future__ import annotations

from html import escape
import json
import sys
from pathlib import Path

from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_TTL = 3600
GEOJSON_SIMPLIFY_TOLERANCE = 0.001
CORRIDOR_SIMPLIFY_TOLERANCE = 0.00035
MAX_MAP_POINTS = 50_000
COMUNA_FILTER_ALL = "Todas"
CORRIDOR_FILTER_ALL = "Todos"
SECTION_OPTIONS = [
    "1. Corredores",
    "2. Comunas",
    "3. Hotspots",
    "4. Patrones Temporales",
    "5. Tendencia Metro",
    "6. Mapa de Calor Vial",
    "7. Velocidad y Flujo",
    "8. Anomalías de Congestión",
]
ANOMALIES_USECOLS = [
    "fecha_trafico",
    "hora",
    "dia_num",
    "corredor",
    "nombre_comuna",
    "velocidad_km_h",
    "intensidad",
    "ocupacion",
    "icv",
    "es_hora_pico",
    "es_fin_semana",
    "z_score_anomaly",
    "iqr_anomaly",
    "speed_drop_anomaly",
    "speed_drop_pct",
    "isolation_forest_anomaly",
    "isolation_score",
    "dbscan_anomaly",
    "anomaly_score",
    "severity_level",
]
# Colores canónicos por nivel de severidad
SEVERITY_COLORS: dict[str, str] = {
    "CRÍTICO" : "#d62728",
    "ALTO"    : "#ff7f0e",
    "MODERADO": "#f7c948",
    "NORMAL"  : "#2ca02c",
}
# Mapeo numérico día → etiqueta corta
DIA_LABELS: dict[int, str] = {
    1: "Lun", 2: "Mar", 3: "Mié",
    4: "Jue", 5: "Vie", 6: "Sáb", 7: "Dom",
}
MASTER_USECOLS = [
    "hora",
    "dia_num",
    "velocidad_km_h",
    "corredor",
    "intensidad",
    "ocupacion",
    "codigo_comuna",
    "nombre_comuna",
    "es_fin_semana",
    "franja_horaria",
    "es_hora_pico",
    "poblacion_2019",
    "icv",
]
MASTER_DTYPES = {
    "hora": "Int16",
    "dia_num": "Int8",
    "velocidad_km_h": "float32",
    "corredor": "string",
    "intensidad": "float32",
    "ocupacion": "float32",
    "codigo_comuna": "string",
    "nombre_comuna": "string",
    "es_fin_semana": "string",
    "franja_horaria": "string",
    "es_hora_pico": "string",
    "poblacion_2019": "float32",
    "icv": "float32",
}
PASAJEROS_DTYPES = {
    "ano": "Int16",
    "num_mes": "Int8",
    "semestre": "Int8",
    "total_pax": "float64",
}
COMUNA_MAP_COLUMNS = [
    "codigo_geo",
    "nombre_comuna",
    "icv_medio",
    "registros",
    "velocidad_media",
    "intensidad_media",
    "corredores_activos",
    "comuna_mapa",
]
OFFICIAL_COMUNA_COLUMNS = [
    "codigo_geo",
    "nombre_comuna",
    "lon_label",
    "lat_label",
    "icv_medio",
    "registros",
    "velocidad_media",
    "intensidad_media",
    "corredores_activos",
    "comuna_mapa",
]
CORRIDOR_LABEL_COLUMNS = ["corredor", "lon", "lat"]

st.set_page_config(
    page_title="Movilidad Medellin",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _read_csv_header(path_value: str | Path) -> list[str]:
    path = _resolve_path(path_value)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
    return header.split(",") if header else []


def _restore_boolean_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    truthy = {"true", "1", "yes", "si", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", ""}
    mapping = {value: True for value in truthy} | {value: False for value in falsy}

    for col in columns:
        if col not in df.columns or pd.api.types.is_bool_dtype(df[col]):
            continue

        normalized = df[col].astype(str).str.strip().str.lower()
        parsed = normalized.map(mapping).astype("boolean")

        if parsed.notna().any():
            df = df.assign(**{col: parsed.fillna(False).astype("boolean")})

    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_master_data(path_value: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_value)
    if not path.exists():
        return pd.DataFrame()

    available_cols = _read_csv_header(path)
    usecols = [col for col in MASTER_USECOLS if col in available_cols]
    dtypes = {col: dtype for col, dtype in MASTER_DTYPES.items() if col in usecols}
    master = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        low_memory=False,
    )
    return _restore_boolean_columns(master, ("es_fin_semana", "es_hora_pico"))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_pasajeros_data(path_value: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_value)
    if not path.exists():
        return pd.DataFrame()

    available_cols = _read_csv_header(path)
    dtypes = {col: dtype for col, dtype in PASAJEROS_DTYPES.items() if col in available_cols}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_ciclorrutas_data(path_value: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_value)
    if not path.exists():
        return pd.DataFrame()

    from src.pipeline import _clean_ciclorrutas, _safe_read_csv

    return _clean_ciclorrutas(_safe_read_csv(str(path)))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _run_pipeline_if_needed(config: dict) -> dict:
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"])
    cfg["paths"]["raw"] = str(_resolve_path(config["paths"]["raw"]))
    cfg["paths"]["processed"] = str(_resolve_path(config["paths"]["processed"]))

    from src.pipeline import run as run_pipeline

    return run_pipeline(cfg)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Cargando datasets del dashboard...")
def load_dashboard_datasets() -> dict:
    cfg = _load_config()
    proc_dir = _resolve_path(cfg["paths"]["processed"])
    raw_dir = _resolve_path(cfg["paths"]["raw"])
    master_path = proc_dir / "master.csv"
    pax_path = proc_dir / "pasajeros_clean.csv"
    cic_path = raw_dir / cfg["files"]["ciclorrutas"]

    if master_path.exists():
        return {
            "master": _load_master_data(master_path),
            "pasajeros": _load_pasajeros_data(pax_path),
            "ciclorrutas": _load_ciclorrutas_data(cic_path),
            "config": cfg,
        }

    results = _run_pipeline_if_needed(cfg)
    results["master"] = _restore_boolean_columns(
        results.get("master", pd.DataFrame()),
        ("es_fin_semana", "es_hora_pico"),
    )
    results["config"] = cfg
    return results


def _apply_filters(
    df: pd.DataFrame,
    comunas: list[str],
    corredores: list[str],
    franja: str,
    dia: str,
) -> pd.DataFrame:
    filtered = df

    if comunas and "nombre_comuna" in filtered.columns:
        filtered = filtered[filtered["nombre_comuna"].isin(comunas)]
    if corredores and "corredor" in filtered.columns:
        filtered = filtered[filtered["corredor"].isin(corredores)]
    if franja != "Todas" and "franja_horaria" in filtered.columns:
        filtered = filtered[filtered["franja_horaria"] == franja]
    if dia != "Todos" and "dia_num" in filtered.columns:
        dias_map = {
            "Lunes": 1,
            "Martes": 2,
            "Miercoles": 3,
            "Jueves": 4,
            "Viernes": 5,
            "Sabado": 6,
            "Domingo": 7,
        }
        if dia in dias_map:
            filtered = filtered[filtered["dia_num"] == dias_map[dia]]

    return filtered


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _build_filter_relationships(df: pd.DataFrame) -> dict:
    if df.empty or not {"nombre_comuna", "corredor"}.issubset(df.columns):
        return {"comunas": [], "all_corridors": [], "by_comuna": {}}

    pairs = (
        df[["nombre_comuna", "corredor"]]
        .dropna(subset=["nombre_comuna", "corredor"])
        .astype({"nombre_comuna": "string", "corredor": "string"})
        .assign(
            nombre_comuna=lambda d: d["nombre_comuna"].str.strip(),
            corredor=lambda d: d["corredor"].str.strip(),
        )
    )
    pairs = pairs[(pairs["nombre_comuna"] != "") & (pairs["corredor"] != "")]
    if pairs.empty:
        return {"comunas": [], "all_corridors": [], "by_comuna": {}}

    pairs = pairs.drop_duplicates().sort_values(["nombre_comuna", "corredor"]).reset_index(drop=True)
    by_comuna = (
        pairs.groupby("nombre_comuna")["corredor"]
        .agg(lambda s: sorted(pd.unique(s).tolist()))
        .to_dict()
    )

    return {
        "comunas": sorted(by_comuna.keys()),
        "all_corridors": sorted(pd.unique(pairs["corredor"]).tolist()),
        "by_comuna": by_comuna,
    }


def _get_corridor_options(filter_relationships: dict, selected_comuna: str) -> list[str]:
    if selected_comuna == COMUNA_FILTER_ALL:
        return filter_relationships.get("all_corridors", [])
    return filter_relationships.get("by_comuna", {}).get(selected_comuna, [])


def _on_comuna_change() -> None:
    current = st.session_state.get("filter_comuna", COMUNA_FILTER_ALL)
    previous = st.session_state.get("_prev_filter_comuna", COMUNA_FILTER_ALL)

    # Correccion: al cambiar la comuna, limpiamos el corredor para evitar selecciones invalidas.
    if current != previous:
        st.session_state["filter_corredor"] = CORRIDOR_FILTER_ALL

    st.session_state["_prev_filter_comuna"] = current


def _normalize_cascading_filters(selected_comuna: str, selected_corredor: str) -> tuple[list[str], list[str]]:
    comunas = [] if selected_comuna == COMUNA_FILTER_ALL else [selected_comuna]
    corredores = [] if selected_corredor == CORRIDOR_FILTER_ALL else [selected_corredor]
    return comunas, corredores


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _read_json_file(path_value: str | Path) -> dict:
    path = _resolve_path(path_value)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        return float(np.hypot(point[0] - start[0], point[1] - start[1]))

    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    point_arr = np.array(point, dtype=float)
    seg = end_arr - start_arr
    t = np.clip(np.dot(point_arr - start_arr, seg) / np.dot(seg, seg), 0.0, 1.0)
    proj = start_arr + t * seg
    return float(np.linalg.norm(point_arr - proj))


def _rdp(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    max_index = -1

    for idx in range(1, len(points) - 1):
        distance = _point_line_distance(points[idx], start, end)
        if distance > max_distance:
            max_distance = distance
            max_index = idx

    if max_distance <= tolerance or max_index < 0:
        return [start, end]

    left = _rdp(points[: max_index + 1], tolerance)
    right = _rdp(points[max_index:], tolerance)
    return left[:-1] + right


def _simplify_ring(coords: list[list[float]], tolerance: float) -> list[list[float]]:
    points = [(float(x), float(y)) for x, y in coords]
    if len(points) <= 4:
        return [[x, y] for x, y in points]

    is_closed = points[0] == points[-1]
    work_points = points[:-1] if is_closed else points
    simplified = _rdp(work_points, tolerance)

    if len(simplified) < 3:
        simplified = work_points

    if is_closed:
        simplified = simplified + [simplified[0]]

    return [[x, y] for x, y in simplified]


def _simplify_line(coords: list[list[float]], tolerance: float) -> list[list[float]]:
    points = [(float(x), float(y)) for x, y in coords]
    if len(points) <= 2:
        return [[x, y] for x, y in points]
    simplified = _rdp(points, tolerance)
    return [[x, y] for x, y in simplified]


def _simplify_geometry(geometry: dict, tolerance: float) -> dict:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        return {
            "type": gtype,
            "coordinates": [_simplify_ring(ring, tolerance) for ring in coords],
        }
    if gtype == "MultiPolygon":
        return {
            "type": gtype,
            "coordinates": [
                [_simplify_ring(ring, tolerance) for ring in polygon]
                for polygon in coords
            ],
        }
    if gtype == "LineString":
        return {"type": gtype, "coordinates": _simplify_line(coords, tolerance)}
    if gtype == "MultiLineString":
        return {
            "type": gtype,
            "coordinates": [_simplify_line(part, tolerance) for part in coords],
        }
    return geometry


def _normalize_comuna_geojson(geojson: dict) -> dict:
    normalized = {"type": geojson.get("type"), "features": []}

    for feature in geojson.get("features", []):
        props = dict(feature.get("properties", {}))
        code = str(props.get("codigo", "")).strip()
        if not code.isdigit():
            continue

        norm_code = code.zfill(2)
        props["codigo"] = norm_code
        normalized["features"].append(
            {
                "type": feature.get("type", "Feature"),
                "id": norm_code,
                "properties": props,
                "geometry": feature.get("geometry", {}),
            }
        )

    return normalized


def _simplify_feature_collection(geojson: dict, tolerance: float) -> dict:
    return {
        "type": geojson.get("type"),
        "features": [
            {
                "type": feature.get("type", "Feature"),
                "id": feature.get("id"),
                "properties": dict(feature.get("properties", {})),
                "geometry": _simplify_geometry(feature.get("geometry", {}), tolerance),
            }
            for feature in geojson.get("features", [])
        ],
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_comunas_geojson(path_value: str | Path) -> dict:
    path = _resolve_path(path_value)
    geojson = _read_json_file(path)
    if not geojson:
        return {}

    normalized = _normalize_comuna_geojson(geojson)
    if path.stat().st_size > 1_000_000:
        return _simplify_feature_collection(normalized, GEOJSON_SIMPLIFY_TOLERANCE)
    return normalized


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_corridors_geojson(path_value: str | Path) -> dict:
    path = _resolve_path(path_value)
    geojson = _read_json_file(path)
    if not geojson:
        return {}

    if path.stat().st_size > 1_000_000:
        return _simplify_feature_collection(geojson, CORRIDOR_SIMPLIFY_TOLERANCE)
    return geojson


def _format_comuna_code(value) -> str | None:
    if pd.isna(value):
        return None

    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(2)


def _extract_polygon_points(geometry: dict) -> list[tuple[float, float]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        rings = coords
    elif gtype == "MultiPolygon":
        rings = [ring for polygon in coords for ring in polygon]
    else:
        return []

    best_ring: list[list[float]] = []
    best_size = -1.0
    for ring in rings:
        if len(ring) < 3:
            continue
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        size = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if size > best_size:
            best_ring = ring
            best_size = size

    return [(float(x), float(y)) for x, y in best_ring]


def _extract_polygon_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        polygons = [coords]
    elif gtype == "MultiPolygon":
        polygons = coords
    else:
        return []

    rings: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        if not polygon:
            continue
        exterior = polygon[0]
        if len(exterior) < 3:
            continue
        rings.append([(float(x), float(y)) for x, y in exterior])
    return rings


def _polygon_centroid(geometry: dict) -> tuple[float | None, float | None]:
    points = _extract_polygon_points(geometry)
    if len(points) < 3:
        return None, None

    area = 0.0
    cx = 0.0
    cy = 0.0
    ring = points + [points[0]]
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    area *= 0.5
    if abs(area) < 1e-12:
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        return float(np.mean(xs)), float(np.mean(ys))

    return cx / (6 * area), cy / (6 * area)


def _empty_dataframe(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _build_official_comuna_frame(comuna_geojson: dict, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for feature in comuna_geojson.get("features", []):
        props = feature.get("properties", {})
        code = str(props.get("codigo", "")).strip()
        if not code.isdigit():
            continue

        lon_label, lat_label = _polygon_centroid(feature.get("geometry", {}))
        records.append(
            {
                "codigo_geo": code.zfill(2),
                "nombre_comuna": str(props.get("nombre", "")).strip(),
                "lon_label": lon_label,
                "lat_label": lat_label,
            }
        )

    # Correccion: garantizar esquema estable incluso si el GeoJSON llega vacio o incompleto.
    official = pd.DataFrame.from_records(records, columns=OFFICIAL_COMUNA_COLUMNS[:4])
    if official.empty:
        return _empty_dataframe(OFFICIAL_COMUNA_COLUMNS)

    official = official.sort_values("codigo_geo").reset_index(drop=True)

    stats = _build_comuna_map_data(df)

    # Correccion: si el filtro deja el agregado sin llaves, devolvemos la cartografia base sin romper el merge.
    join_keys = {"codigo_geo", "nombre_comuna"}
    if not join_keys.issubset(stats.columns):
        return official.assign(comuna_mapa=lambda d: d["codigo_geo"] + " - " + d["nombre_comuna"])

    merged = official.merge(
        stats.drop(columns=["comuna_mapa"], errors="ignore"),
        on=["codigo_geo", "nombre_comuna"],
        how="left",
    )
    return (
        merged.assign(comuna_mapa=lambda d: d["codigo_geo"] + " - " + d["nombre_comuna"])
        .reindex(columns=OFFICIAL_COMUNA_COLUMNS)
    )


def _line_parts(geometry: dict) -> list[list[tuple[float, float]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "LineString":
        return [[(float(x), float(y)) for x, y in coords]]
    if gtype == "MultiLineString":
        return [[(float(x), float(y)) for x, y in part] for part in coords]
    return []


def _build_urban_paths(comuna_geojson: dict) -> list[tuple[tuple[float, float, float, float], MplPath]]:
    paths: list[tuple[tuple[float, float, float, float], MplPath]] = []
    for feature in comuna_geojson.get("features", []):
        for ring in _extract_polygon_rings(feature.get("geometry", {})):
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            paths.append((bbox, MplPath(ring)))
    return paths


def _point_in_urban_area(lon: float, lat: float, urban_paths: list[tuple[tuple[float, float, float, float], MplPath]]) -> bool:
    point = (lon, lat)
    for bbox, path in urban_paths:
        min_lon, min_lat, max_lon, max_lat = bbox
        if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
            continue
        if path.contains_point(point, radius=1e-6):
            return True
    return False


def _clip_line_part_to_urban_area(
    part: list[tuple[float, float]],
    urban_paths: list[tuple[tuple[float, float, float, float], MplPath]],
) -> list[list[tuple[float, float]]]:
    clipped_parts: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for lon, lat in part:
        if _point_in_urban_area(lon, lat, urban_paths):
            current.append((lon, lat))
            continue

        if len(current) >= 2:
            clipped_parts.append(current)
        current = []

    if len(current) >= 2:
        clipped_parts.append(current)

    return clipped_parts


def _index_corridor_features(corridor_geojson: dict) -> dict[str, list[dict]]:
    indexed: dict[str, list[dict]] = {}
    for feature in corridor_geojson.get("features", []):
        props = feature.get("properties", {})
        corridor = str(props.get("corredor_dataset", "")).strip()
        if not corridor:
            continue
        indexed.setdefault(corridor, []).append(feature)
    return indexed


def _build_corridor_traces_for_selection(
    corridor_features: dict[str, list[dict]],
    corridor_names: tuple[str, ...],
    urban_paths: list[tuple[tuple[float, float, float, float], MplPath]],
) -> tuple[dict[str, tuple[list[float | None], list[float | None]]], pd.DataFrame]:
    line_data: dict[str, tuple[list[float | None], list[float | None]]] = {}
    label_rows = []

    for corridor in corridor_names:
        features = corridor_features.get(corridor, [])
        if not features:
            continue

        lons: list[float | None] = []
        lats: list[float | None] = []
        mid_lons: list[float] = []
        mid_lats: list[float] = []

        for feature in features:
            for part in _line_parts(feature.get("geometry", {})):
                for clipped_part in _clip_line_part_to_urban_area(part, urban_paths):
                    for lon, lat in clipped_part:
                        lons.append(lon)
                        lats.append(lat)
                    lons.append(None)
                    lats.append(None)

                    mid_idx = len(clipped_part) // 2
                    mid_lons.append(clipped_part[mid_idx][0])
                    mid_lats.append(clipped_part[mid_idx][1])

        if not lons:
            continue

        line_data[corridor] = (lons, lats)
        label_rows.append(
            {
                "corredor": corridor,
                "lon": float(np.mean(mid_lons)) if mid_lons else None,
                "lat": float(np.mean(mid_lats)) if mid_lats else None,
            }
        )

    # Correccion: mantener columnas aunque no haya corredores renderizables.
    return line_data, pd.DataFrame.from_records(label_rows, columns=CORRIDOR_LABEL_COLUMNS)


def _build_comuna_map_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"codigo_comuna", "nombre_comuna", "icv"}
    if df.empty or not required.issubset(df.columns):
        # Correccion: evitar KeyError posteriores devolviendo un esquema fijo.
        return _empty_dataframe(COMUNA_MAP_COLUMNS)

    work = (
        df.dropna(subset=["nombre_comuna", "icv"])
        .copy()
        .assign(codigo_geo=lambda d: d["codigo_comuna"].apply(_format_comuna_code))
    )
    work = work.dropna(subset=["codigo_geo"])
    if work.empty:
        # Correccion: los filtros pueden dejar el agregado sin filas; el mapa debe seguir renderizando.
        return _empty_dataframe(COMUNA_MAP_COLUMNS)

    agg_map = {
        "icv_medio": ("icv", "mean"),
        "registros": ("icv", "size"),
    }
    if "velocidad_km_h" in work.columns:
        agg_map["velocidad_media"] = ("velocidad_km_h", "mean")
    if "intensidad" in work.columns:
        agg_map["intensidad_media"] = ("intensidad", "mean")
    if "corredor" in work.columns:
        agg_map["corredores_activos"] = ("corredor", "nunique")

    summary = (
        work.groupby(["codigo_geo", "nombre_comuna"], as_index=False)
        .agg(**agg_map)
        .sort_values("codigo_geo")
        .reset_index(drop=True)
    )
    summary = summary.assign(comuna_mapa=lambda d: d["codigo_geo"] + " - " + d["nombre_comuna"])
    return summary.reindex(columns=COMUNA_MAP_COLUMNS)


@st.cache_resource(show_spinner=False)
def _build_map_resources(processed_dir_value: str | Path) -> dict:
    proc_dir = _resolve_path(processed_dir_value)
    comuna_geojson = _load_comunas_geojson(proc_dir / "comunas_medellin.geojson")
    corridor_geojson = _load_corridors_geojson(proc_dir / "corredores_oficiales_medellin.geojson")
    corridor_features = _index_corridor_features(corridor_geojson)
    urban_paths = _build_urban_paths(comuna_geojson)
    available_corridors = sorted(corridor_features.keys())

    return {
        "comuna_geojson": comuna_geojson,
        "corridor_features": corridor_features,
        "urban_paths": urban_paths,
        "available_corridors": available_corridors,
    }


@st.cache_resource(show_spinner=False)
def _get_corridor_traces(
    processed_dir_value: str | Path,
    corridor_names: tuple[str, ...],
) -> tuple[dict[str, tuple[list[float | None], list[float | None]]], pd.DataFrame]:
    resources = _build_map_resources(processed_dir_value)
    return _build_corridor_traces_for_selection(
        resources["corridor_features"],
        corridor_names,
        resources["urban_paths"],
    )


def _count_trace_points(lons: list[float | None]) -> int:
    return sum(value is not None for value in lons)


def _downsample_line_trace(
    lons: list[float | None],
    lats: list[float | None],
    step: int,
) -> tuple[list[float | None], list[float | None]]:
    if step <= 1:
        return lons, lats

    sampled_lons: list[float | None] = []
    sampled_lats: list[float | None] = []
    segment_lons: list[float] = []
    segment_lats: list[float] = []

    def flush_segment() -> None:
        if not segment_lons:
            return
        sampled_lons.extend(segment_lons[::step])
        sampled_lats.extend(segment_lats[::step])
        if sampled_lons[-1] != segment_lons[-1] or sampled_lats[-1] != segment_lats[-1]:
            sampled_lons.append(segment_lons[-1])
            sampled_lats.append(segment_lats[-1])
        sampled_lons.append(None)
        sampled_lats.append(None)
        segment_lons.clear()
        segment_lats.clear()

    for lon, lat in zip(lons, lats):
        if lon is None or lat is None:
            flush_segment()
            continue
        segment_lons.append(lon)
        segment_lats.append(lat)

    flush_segment()
    return sampled_lons, sampled_lats


def _limit_corridor_points(
    selected_traces: dict[str, tuple[list[float | None], list[float | None]]],
    max_points: int = MAX_MAP_POINTS,
) -> dict[str, tuple[list[float | None], list[float | None]]]:
    total_points = sum(_count_trace_points(lons) for lons, _ in selected_traces.values())
    if total_points <= max_points or total_points == 0:
        return selected_traces

    step = int(np.ceil(total_points / max_points))
    return {
        corridor: _downsample_line_trace(lons, lats, step)
        for corridor, (lons, lats) in selected_traces.items()
    }


def _render_color_table(title: str, rows: list[dict[str, str]]) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.info(f"No hay elementos para {title.lower()}.")
        return

    html_rows = []
    for row in rows:
        color = escape(row["color"])
        label = escape(row["label"])
        html_rows.append(
            "<tr>"
            f"<td style='padding:6px 10px; width:56px; text-align:center;'>"
            f"<span style='display:inline-block; width:18px; height:18px; border-radius:4px; "
            f"background:{color}; border:1px solid #d0d7de;'></span></td>"
            f"<td style='padding:6px 10px;'>{label}</td>"
            "</tr>"
        )

    st.markdown(
        (
            "<table style='width:100%; border-collapse:collapse; font-size:0.95rem;'>"
            "<thead><tr>"
            "<th style='text-align:center; padding:6px 10px;'>Color</th>"
            "<th style='text-align:left; padding:6px 10px;'>Nombre</th>"
            "</tr></thead>"
            f"<tbody>{''.join(html_rows)}</tbody></table>"
        ),
        unsafe_allow_html=True,
    )


def _kpi_card(col, label: str, value, delta=None):
    with col:
        if delta is not None:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)


def _prepare_filtered_views(
    master: pd.DataFrame,
    cfg: dict,
    sel_comunas: list[str],
    sel_corredores: list[str],
    sel_franja: str,
    sel_dia: str,
) -> dict:
    df_f = _apply_filters(master, sel_comunas, sel_corredores, sel_franja, sel_dia)

    from src.pipeline import _identify_hotspots, _rank_comunas, _rank_corredores, _rank_corredores_ivc

    ranking_cfg = dict(cfg)
    ranking_cfg["rankings"] = dict(cfg.get("rankings", {}))
    if sel_dia != "Todos":
        ranking_cfg["rankings"]["solo_dias_habiles"] = False

    corredores_f = _rank_corredores(df_f, ranking_cfg) if not df_f.empty else pd.DataFrame()
    corredores_ivc_f = _rank_corredores_ivc(df_f, cfg) if not df_f.empty else pd.DataFrame()
    comunas_f = _rank_comunas(df_f, ranking_cfg) if not df_f.empty else pd.DataFrame()
    hotspots_f = (
        _identify_hotspots(df_f, cfg)
        if not df_f.empty and {"velocidad_km_h", "intensidad"}.issubset(df_f.columns)
        else pd.DataFrame()
    )

    return {
        "df_f": df_f,
        "corredores_f": corredores_f,
        "corredores_ivc_f": corredores_ivc_f,
        "comunas_f": comunas_f,
        "hotspots_f": hotspots_f,
    }


def _render_top_corredores(master: pd.DataFrame, cfg: dict, ivc_cfg: dict) -> None:
    st.header("Top 10 Corredores Criticos")
    
    # Filtro independiente: Comuna
    filter_relationships = _build_filter_relationships(master)
    comunas_opts = filter_relationships.get("comunas", [])
    
    col_filter_left, col_filter_right = st.columns([1, 3])
    with col_filter_left:
        selected_comuna_tc = st.selectbox(
            "Filtrar por Comuna",
            [COMUNA_FILTER_ALL, *comunas_opts],
            key="tc_filter_comuna",
            help="Selecciona una comuna para ver solo los corredores de esa zona"
        )
    
    # Aplicar filtro a master
    if selected_comuna_tc != COMUNA_FILTER_ALL:
        df_tc = master[master["nombre_comuna"] == selected_comuna_tc].copy()
    else:
        df_tc = master.copy()
    
    # Calcular rankings con datos filtrados
    from src.pipeline import _rank_corredores, _rank_corredores_ivc
    
    if not df_tc.empty:
        corredores_f = _rank_corredores(df_tc, cfg)
        corredores_ivc_f = _rank_corredores_ivc(df_tc, cfg)
    else:
        corredores_f = pd.DataFrame()
        corredores_ivc_f = pd.DataFrame()

    col_t, col_g = st.columns([1.4, 1])
    with col_t:
        if not corredores_f.empty:
            st.dataframe(
                corredores_f,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "corredor": "Corredor",
                    "icv_medio": st.column_config.ProgressColumn(
                        "ICV Medio", min_value=0, max_value=100, format="%.1f"
                    ),
                    "icv_maximo": st.column_config.NumberColumn("ICV Maximo", format="%.1f"),
                    "n_registros": st.column_config.NumberColumn("Registros", format="%d"),
                },
            )
        else:
            st.info("No hay datos de corredores para el filtro seleccionado.")
    st.header("Top 15 Corredores Criticos")
    with col_g:
        if not corredores_f.empty:
            fig = px.bar(
                corredores_f.sort_values("icv_medio"),
                x="icv_medio",
                y="corredor",
                orientation="h",
                color="icv_maximo",
                color_continuous_scale="Reds",
                labels={"icv_medio": "ICV Medio", "corredor": "", "icv_maximo": "ICV Maximo"},
                title="Top 10 por ICV medio",
            )
            fig.update_layout(
                height=380,
                coloraxis_colorbar={"title": "ICV Max"},
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Ranking complementario por IVC: intensidad promedio / {ivc_cfg.get('capacidad_base', 2500)} veh/h. "
        "Se calcula con limpieza y agrupacion alineadas con la validacion en Colab."
    )

    col_t_ivc, col_g_ivc = st.columns([1.4, 1])
    with col_t_ivc:
        if not corredores_ivc_f.empty:
            st.dataframe(
                corredores_ivc_f,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "corredor": "Corredor",
                    "ivc_medio": st.column_config.NumberColumn("IVC Medio", format="%.3f"),
                    "velocidad_media": st.column_config.NumberColumn("Velocidad", format="%.1f km/h"),
                    "intensidad_media": st.column_config.NumberColumn("Intensidad", format="%.0f veh/h"),
                    "icv_simple_medio": st.column_config.NumberColumn("Perdida Vel.", format="%.3f"),
                    "n_registros": st.column_config.NumberColumn("Registros", format="%d"),
                },
            )
        else:
            st.info("No hay datos para calcular el ranking IVC.")

    with col_g_ivc:
        if not corredores_ivc_f.empty:
            fig_ivc = px.bar(
                corredores_ivc_f.sort_values("ivc_medio"),
                x="ivc_medio",
                y="corredor",
                orientation="h",
                color="ivc_medio",
                color_continuous_scale="Turbo",
                labels={"ivc_medio": "IVC Medio", "corredor": ""},
                title=f"Top {len(corredores_ivc_f)} por IVC promedio",
                text="ivc_medio",
                hover_data={
                    "velocidad_media": ":.1f",
                    "intensidad_media": ":.0f",
                    "icv_simple_medio": ":.3f",
                    "n_registros": True,
                },
            )
            fig_ivc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_ivc.update_layout(
                height=520,
                coloraxis_colorbar={"title": "Nivel<br>Saturacion"},
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Indice Volumen/Capacidad (Promedio)",
                yaxis_title="",
            )
            st.plotly_chart(fig_ivc, use_container_width=True)


def _render_comunas(master: pd.DataFrame, cfg: dict) -> None:
    st.header("Comunas con Mayor Presion Vehicular")
    st.caption("Filtros para análisis por comuna")
    
    # Filtros independientes: Franja horaria y Día de semana
    franjas_opts = ["Todas", "madrugada", "manana", "mediodia", "tarde", "noche"]
    dias_opts = ["Todos", "Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    
    col_filter_1, col_filter_2 = st.columns(2)
    with col_filter_1:
        sel_franja_cm = st.selectbox(
            "Franja horaria",
            franjas_opts,
            key="cm_filter_franja",
            help="Selecciona una franja para filtrar los datos"
        )
    
    with col_filter_2:
        sel_dia_cm = st.selectbox(
            "Día de semana",
            dias_opts,
            key="cm_filter_dia",
            help="Selecciona un día para filtrar los datos"
        )
    
    # Aplicar filtros
    df_cm = master.copy()
    
    if sel_franja_cm != "Todas" and "franja_horaria" in df_cm.columns:
        df_cm = df_cm[df_cm["franja_horaria"] == sel_franja_cm]
    
    if sel_dia_cm != "Todos" and "dia_num" in df_cm.columns:
        dias_map = {
            "Lunes": 1, "Martes": 2, "Miercoles": 3, "Jueves": 4,
            "Viernes": 5, "Sabado": 6, "Domingo": 7,
        }
        if sel_dia_cm in dias_map:
            df_cm = df_cm[df_cm["dia_num"] == dias_map[sel_dia_cm]]
    
    # Calcular ranking de comunas
    from src.pipeline import _rank_comunas
    ranking_cfg = dict(cfg)
    ranking_cfg["rankings"] = dict(cfg.get("rankings", {}))
    if sel_dia_cm != "Todos":
        ranking_cfg["rankings"]["solo_dias_habiles"] = False
    
    comunas_f = _rank_comunas(df_cm, ranking_cfg) if not df_cm.empty else pd.DataFrame()

    if not comunas_f.empty:
        fig_c = px.bar(
            comunas_f.sort_values("icv_medio"),
            x="icv_medio",
            y="nombre_comuna",
            orientation="h",
            color="icv_medio",
            color_continuous_scale="Oranges",
            labels={"icv_medio": "ICV Medio", "nombre_comuna": ""},
            title="Comunas - ICV en hora pico",
            text="icv_medio",
        )
        fig_c.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_c.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Sin datos de comunas con los filtros seleccionados.")


def _render_hotspots(master: pd.DataFrame, cfg: dict) -> None:
    st.header("Hotspots de Congestion")

    vel_pct = cfg.get("rankings", {}).get("hotspot_velocidad_pct", 30)
    int_pct = cfg.get("rankings", {}).get("hotspot_intensidad_pct", 70)
    
    # Filtro independiente: Comuna
    filter_relationships = _build_filter_relationships(master)
    comunas_opts = filter_relationships.get("comunas", [])
    
    selected_comuna_hs = st.selectbox(
        "Filtrar Hotspots por Comuna",
        [COMUNA_FILTER_ALL, *comunas_opts],
        key="hs_filter_comuna",
        help="Selecciona una comuna para ver sus hotspots"
    )
    
    # Aplicar filtro
    df_hs = master.copy()
    if selected_comuna_hs != COMUNA_FILTER_ALL:
        df_hs = df_hs[df_hs["nombre_comuna"] == selected_comuna_hs]
    
    # Identificar hotspots
    from src.pipeline import _identify_hotspots
    hotspots_f = (
        _identify_hotspots(df_hs, cfg)
        if not df_hs.empty and {"velocidad_km_h", "intensidad"}.issubset(df_hs.columns)
        else pd.DataFrame()
    )
    n_hotspots = len(hotspots_f)
    
    st.caption(
        f"Definicion: zonas con velocidad <= P{vel_pct} (velocidad baja) "
        f"e intensidad >= P{int_pct} (flujo alto) - **{n_hotspots:,} zonas criticas detectadas**"
    )

    if not hotspots_f.empty and "velocidad_km_h" in hotspots_f.columns and "intensidad" in hotspots_f.columns:
        sample_hs = hotspots_f.sample(min(5000, len(hotspots_f)), random_state=42)
        hover_data = {}
        if "corredor" in sample_hs.columns:
            hover_data["corredor"] = True
        fig_hs = px.scatter(
            sample_hs,
            x="intensidad",
            y="velocidad_km_h",
            color="icv",
            color_continuous_scale="RdYlGn_r",
            hover_data=hover_data,
            labels={
                "intensidad": "Intensidad (veh/h)",
                "velocidad_km_h": "Velocidad (km/h)",
                "icv": "ICV",
            },
            title="Hotspots: Intensidad vs Velocidad (color = ICV)",
            opacity=0.5,
        )
        fig_hs.update_layout(height=420)
        st.plotly_chart(fig_hs, use_container_width=True)
    else:
        st.info("Sin hotspots con los filtros actuales.")


def _render_patrones_temporales(master: pd.DataFrame, cfg: dict) -> None:
    st.header("Patrones Temporales")
    
    # Filtro independiente: Comuna
    filter_relationships = _build_filter_relationships(master)
    comunas_opts = filter_relationships.get("comunas", [])
    
    selected_comuna_pt = st.selectbox(
        "Filtrar Patrones por Comuna",
        [COMUNA_FILTER_ALL, *comunas_opts],
        key="pt_filter_comuna",
        help="Selecciona una comuna para ver sus patrones temporales"
    )
    
    # Aplicar filtro
    df_pt = master.copy()
    if selected_comuna_pt != COMUNA_FILTER_ALL:
        df_pt = df_pt[df_pt["nombre_comuna"] == selected_comuna_pt]
    
    if not df_pt.empty and "hora" in df_pt.columns:
        hourly = df_pt.groupby("hora")["icv"].mean().reset_index()
        hourly.columns = ["hora", "icv_medio"]

        pico_manana = cfg.get("eta", {}).get("hora_pico_manana", [7, 8])
        pico_tarde = cfg.get("eta", {}).get("hora_pico_tarde", [17, 18])

        fig_h = go.Figure()
        fig_h.add_trace(
            go.Scatter(
                x=hourly["hora"],
                y=hourly["icv_medio"],
                mode="lines+markers",
                name="ICV medio",
                line=dict(color="#e74c3c", width=2.5),
            )
        )
        fig_h.add_vrect(
            x0=min(pico_manana) - 0.5,
            x1=max(pico_manana) + 0.5,
            fillcolor="rgba(231,76,60,0.12)",
            line_width=0,
            annotation_text="Pico AM",
            annotation_position="top left",
        )
        fig_h.add_vrect(
            x0=min(pico_tarde) - 0.5,
            x1=max(pico_tarde) + 0.5,
            fillcolor="rgba(231,76,60,0.12)",
            line_width=0,
            annotation_text="Pico PM",
            annotation_position="top left",
        )
        fig_h.update_layout(
            title="ICV promedio por hora del dia",
            xaxis_title="Hora (0-23)",
            yaxis_title="ICV promedio",
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
            height=380,
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Sin datos suficientes para el analisis horario.")


def _render_pasajeros(pasajeros: pd.DataFrame) -> None:
    st.header("Tendencia Pasajeros Metro (2014-2021)")

    if pasajeros.empty or "total_pax" not in pasajeros.columns:
        st.info("Datos de pasajeros no disponibles.")
        return

    if "fecha_periodo" in pasajeros.columns:
        pax_trend = (
            pasajeros.dropna(subset=["fecha_periodo", "total_pax"])
            .assign(fecha_periodo=lambda d: pd.to_datetime(d["fecha_periodo"], errors="coerce"))
            .dropna(subset=["fecha_periodo"])
            .groupby("fecha_periodo")["total_pax"]
            .sum()
            .reset_index()
            .sort_values("fecha_periodo")
        )
    elif "ano" in pasajeros.columns:
        pax_trend = (
            pasajeros.dropna(subset=["total_pax"])
            .groupby("ano")["total_pax"]
            .sum()
            .reset_index()
            .rename(columns={"ano": "fecha_periodo"})
            .sort_values("fecha_periodo")
        )
    else:
        pax_trend = pd.DataFrame()

    if pax_trend.empty:
        st.info("No se pudo generar la tendencia temporal de pasajeros.")
        return

    fig_pax = px.line(
        pax_trend,
        x="fecha_periodo",
        y="total_pax",
        markers=True,
        labels={"fecha_periodo": "Periodo", "total_pax": "Pasajeros movilizados"},
        title="Evolucion de pasajeros movilizados en el Metro de Medellin",
        color_discrete_sequence=["#2980b9"],
    )
    fig_pax.update_layout(height=380)
    fig_pax.update_traces(line_width=2.5)
    st.plotly_chart(fig_pax, use_container_width=True)


def _render_velocidad_flujo(master: pd.DataFrame, cfg: dict) -> None:
    st.header("Velocidad y Flujo: Análisis de Congestión")
    st.caption(
        "Identificación de puntos con **baja velocidad Y alto flujo simultáneamente**. "
        "Estos son los principales cuellos de botella del sistema."
    )
    
    # Filtros independientes: Comuna y Corredor
    filter_relationships = _build_filter_relationships(master)
    comunas_opts = filter_relationships.get("comunas", [])
    
    col_filter_1, col_filter_2 = st.columns(2)
    
    with col_filter_1:
        selected_comuna_vf = st.selectbox(
            "Filtrar por Comuna",
            [COMUNA_FILTER_ALL, *comunas_opts],
            key="vf_filter_comuna",
            help="Selecciona una comuna para filtrar datos"
        )
    
    # Obtener corredores según la comuna seleccionada
    available_corridors = filter_relationships.get("all_corridors", []) if selected_comuna_vf == COMUNA_FILTER_ALL else filter_relationships.get("by_comuna", {}).get(selected_comuna_vf, [])
    
    with col_filter_2:
        selected_corredor_vf = st.selectbox(
            "Filtrar por Corredor",
            [CORRIDOR_FILTER_ALL, *available_corridors],
            key="vf_filter_corredor",
            help="Selecciona un corredor para filtrar datos"
        )
    
    # Aplicar filtros
    df_vf = master.copy()
    
    if selected_comuna_vf != COMUNA_FILTER_ALL and "nombre_comuna" in df_vf.columns:
        df_vf = df_vf[df_vf["nombre_comuna"] == selected_comuna_vf]
    
    if selected_corredor_vf != CORRIDOR_FILTER_ALL and "corredor" in df_vf.columns:
        df_vf = df_vf[df_vf["corredor"] == selected_corredor_vf]
    
    # Validar que tenemos los datos necesarios
    if df_vf.empty or not {"velocidad_km_h", "intensidad"}.issubset(df_vf.columns):
        st.info("No hay datos disponibles con los filtros seleccionados.")
        return
    
    # Crear categoría para identificar puntos críticos
    # Baja velocidad: < percentil 30
    # Alto flujo: > percentil 70
    vel_p30 = df_vf["velocidad_km_h"].quantile(0.30)
    int_p70 = df_vf["intensidad"].quantile(0.70)
    
    df_vf = df_vf.copy()
    df_vf["criticidad"] = df_vf.apply(
        lambda row: "Crítico (Baja vel + Alto flujo)" 
        if (row["velocidad_km_h"] <= vel_p30 and row["intensidad"] >= int_p70)
        else ("Alto flujo" if row["intensidad"] >= int_p70 else ("Baja velocidad" if row["velocidad_km_h"] <= vel_p30 else "Normal")),
        axis=1
    )
    
    # Estadísticas
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        vel_prom = df_vf["velocidad_km_h"].mean()
        st.metric("Velocidad promedio", f"{vel_prom:.1f} km/h")
    
    with stats_col2:
        int_prom = df_vf["intensidad"].mean()
        st.metric("Intensidad promedio", f"{int_prom:.0f} veh/h")
    
    with stats_col3:
        criticos = len(df_vf[df_vf["criticidad"] == "Crítico (Baja vel + Alto flujo)"])
        pct_criticos = 100 * criticos / len(df_vf) if len(df_vf) > 0 else 0
        st.metric("Puntos críticos", f"{criticos:,} ({pct_criticos:.1f}%)")
    
    with stats_col4:
        icv_prom = df_vf["icv"].mean() if "icv" in df_vf.columns else 0
        st.metric("ICV promedio", f"{icv_prom:.1f}")
    
    st.markdown("---")
    
    # Gráfico principal: Intensidad vs Velocidad
    fig_vf = px.scatter(
        df_vf,
        x="intensidad",
        y="velocidad_km_h",
        color="criticidad",
        size="icv" if "icv" in df_vf.columns else None,
        hover_data=["corredor", "nombre_comuna", "hora"] if all(col in df_vf.columns for col in ["corredor", "nombre_comuna", "hora"]) else [],
        color_discrete_map={
            "Crítico (Baja vel + Alto flujo)": "#d62728",  # Rojo
            "Alto flujo": "#ff7f0e",  # Naranja
            "Baja velocidad": "#f7c948",  # Amarillo
            "Normal": "#2ca02c",  # Verde
        },
        labels={
            "intensidad": "Intensidad (veh/h)",
            "velocidad_km_h": "Velocidad (km/h)",
            "criticidad": "Clasificación",
        },
        title="Análisis de Velocidad vs Flujo Vehicular",
        opacity=0.6,
    )
    
    # Agregar líneas de referencia
    fig_vf.add_hline(
        y=vel_p30,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"P30 Velocidad ({vel_p30:.1f} km/h)",
        annotation_position="right",
        opacity=0.5,
    )
    fig_vf.add_vline(
        x=int_p70,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"P70 Intensidad ({int_p70:.0f} veh/h)",
        annotation_position="top",
        opacity=0.5,
    )
    
    fig_vf.update_layout(
        height=500,
        hovermode="closest",
        font=dict(size=11),
    )
    st.plotly_chart(fig_vf, use_container_width=True)
    
    # Tabla de datos filtrados ordenada por criticidad
    st.markdown("### Datos detallados (primeras 50 filas)")
    
    display_cols = ["corredor", "nombre_comuna", "velocidad_km_h", "intensidad", "icv", "hora", "criticidad"]
    available_display_cols = [col for col in display_cols if col in df_vf.columns]
    
    df_display = df_vf[available_display_cols].sort_values("criticidad").head(50)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "velocidad_km_h": st.column_config.NumberColumn("Velocidad (km/h)", format="%.1f"),
            "intensidad": st.column_config.NumberColumn("Intensidad (veh/h)", format="%.0f"),
            "icv": st.column_config.NumberColumn("ICV", format="%.1f"),
            "hora": st.column_config.NumberColumn("Hora", format="%d:00"),
            "criticidad": st.column_config.TextColumn("Clasificación"),
        }
    )


def _render_mapa_vial(
    master: pd.DataFrame,
    ciclorrutas: pd.DataFrame,
    cfg: dict,
    map_cfg: dict,
) -> None:
    st.header("Mapa de Calor Vial")
    
    # Filtro independiente: Comuna
    filter_relationships = _build_filter_relationships(master)
    comunas_opts = filter_relationships.get("comunas", [])
    
    selected_comuna_mv = st.selectbox(
        "Filtrar Mapa por Comuna",
        [COMUNA_FILTER_ALL, *comunas_opts],
        key="mv_filter_comuna",
        help="Selecciona una comuna para ver el mapa filtrado"
    )
    
    # Aplicar filtro
    df_f = master.copy()
    if selected_comuna_mv != COMUNA_FILTER_ALL:
        df_f = df_f[df_f["nombre_comuna"] == selected_comuna_mv]
    
    # Calcular corredores críticos del dataset filtrado
    from src.pipeline import _rank_corredores
    corredores_f = _rank_corredores(df_f, cfg) if not df_f.empty else pd.DataFrame()
    sel_corredores = corredores_f["corredor"].tolist()[:10] if not corredores_f.empty else []

    proc_dir = _resolve_path(cfg.get("paths", {}).get("processed", "data/processed"))
    placeholder = st.empty()
    placeholder.info("Preparando cartografia oficial y trazas viales optimizadas...")

    with st.spinner("Cargando mapa oficial de Medellin..."):
        resources = _build_map_resources(proc_dir)
        comuna_geojson = resources["comuna_geojson"]
        available_corridors = resources["available_corridors"]

        comuna_map = _build_official_comuna_frame(comuna_geojson, df_f)

        if sel_corredores:
            corridor_names = [c for c in sel_corredores if c in available_corridors]
        elif not corredores_f.empty:
            corridor_names = [c for c in corredores_f["corredor"].tolist() if c in available_corridors][:10]
        else:
            corridor_names = available_corridors[:10]

        selected_traces, corridor_labels = _get_corridor_traces(proc_dir, tuple(corridor_names))
        selected_traces = _limit_corridor_points(selected_traces, MAX_MAP_POINTS)

    placeholder.empty()

    if not comuna_geojson or comuna_map.empty:
        st.info(
            "No se pudo construir el mapa por comunas. Verifica que existan `master.csv` "
            "y los archivos oficiales en `data/processed/`."
        )
        return

    if df_f.empty:
        # Correccion: cuando los filtros no devuelven filas, mostramos la cartografia base sin lanzar excepciones.
        st.info("No hay registros para los filtros actuales. Se muestra la cartografia oficial de referencia.")

    lat_c = map_cfg.get("lat_centro", 6.2442)
    lon_c = map_cfg.get("lon_centro", -75.5812)
    zoom_c = map_cfg.get("zoom", 11)
    comuna_palette = [
        "#e76f51", "#2a9d8f", "#e9c46a", "#f4a261",
        "#457b9d", "#8d99ae", "#ef476f", "#06d6a0",
        "#118ab2", "#ffd166", "#8338ec", "#3a86ff",
        "#fb8500", "#90be6d", "#577590", "#f94144",
    ]
    comuna_label_texts = comuna_map["comuna_mapa"].tolist()
    comuna_colors = {
        label: comuna_palette[i % len(comuna_palette)]
        for i, label in enumerate(comuna_label_texts)
    }
    corridor_palette = px.colors.qualitative.Dark24
    corridor_colors = {
        corridor: corridor_palette[i % len(corridor_palette)]
        for i, corridor in enumerate(corridor_names)
    }
    rendered_corridor_names: list[str] = []

    fig_map = px.choropleth_mapbox(
        comuna_map,
        geojson=comuna_geojson,
        locations="codigo_geo",
        featureidkey="properties.codigo",
        color="comuna_mapa",
        color_discrete_map=comuna_colors,
        custom_data=["nombre_comuna"],
        opacity=0.38,
        zoom=zoom_c,
        center={"lat": lat_c, "lon": lon_c},
        mapbox_style="carto-positron",
        title="Comunas oficiales de Medellin y corredores observados",
    )
    fig_map.update_traces(
        marker_line_width=1.2,
        marker_line_color="#243447",
        hovertemplate="%{location} - %{customdata[0]}<extra></extra>",
    )

    comuna_labels_df = comuna_map.dropna(subset=["lat_label", "lon_label"]).copy()
    if not comuna_labels_df.empty:
        fig_map.add_trace(
            go.Scattermapbox(
                lat=comuna_labels_df["lat_label"],
                lon=comuna_labels_df["lon_label"],
                mode="text",
                text=comuna_labels_df["comuna_mapa"],
                textfont={"size": 11, "color": "#14213d"},
                name="16 comunas oficiales",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for corridor in corridor_names:
        if corridor not in selected_traces:
            continue

        rendered_corridor_names.append(corridor)
        lons, lats = selected_traces[corridor]
        fig_map.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                name=f"Corredor: {corridor}",
                legendgroup=f"corredor-{corridor}",
                line={"width": 3.2, "color": corridor_colors[corridor]},
                text=[corridor] * len(lats),
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    if not corridor_labels.empty:
        corridor_labels = (
            corridor_labels[corridor_labels["corredor"].isin(rendered_corridor_names)]
            .dropna(subset=["lat", "lon"])
            .sort_values("corredor")
        )
        if not corridor_labels.empty:
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=corridor_labels["lat"],
                    lon=corridor_labels["lon"],
                    mode="text",
                    text=corridor_labels["corredor"],
                    textfont={"size": 11, "color": "#1f2933"},
                    name="Etiquetas de corredor",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if sel_corredores and not rendered_corridor_names:
        st.info("Los corredores seleccionados no tienen geometria oficial cargada en la capa vial local.")

    fig_map.update_layout(
        height=620,
        legend_title_text="Comunas y corredores",
        margin=dict(l=0, r=0, t=42, b=0),
        uirevision="mapa-vial",
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "Las 16 comunas se pintan con la geometria oficial de Medellin y quedan siempre marcadas. "
        "Los corredores se dibujan con segmentos oficiales de la malla vial municipal."
    )

    legend_left, legend_right = st.columns(2)
    with legend_left:
        _render_color_table(
            "Comunas",
            [{"label": label, "color": comuna_colors[label]} for label in comuna_label_texts],
        )
    with legend_right:
        _render_color_table(
            "Corredores",
            [
                {"label": corridor, "color": corridor_colors[corridor]}
                for corridor in rendered_corridor_names
                if corridor in corridor_colors
            ],
        )

    if not ciclorrutas.empty:
        with st.expander("Red de Ciclorrutas"):
            col_stats, col_tabla = st.columns([1, 2])
            with col_stats:
                estado_counts = ciclorrutas["estado"].value_counts() if "estado" in ciclorrutas.columns else pd.Series(dtype="int64")
                if not estado_counts.empty:
                    fig_cic = px.pie(
                        values=estado_counts.values,
                        names=estado_counts.index,
                        color_discrete_sequence=["#27ae60", "#f39c12", "#8e44ad"],
                        title="Tramos por estado",
                    )
                    fig_cic.update_layout(height=280)
                    st.plotly_chart(fig_cic, use_container_width=True)
            with col_tabla:
                st.dataframe(ciclorrutas, use_container_width=True, hide_index=True)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _load_anomalies_data(path_value: str | Path) -> pd.DataFrame:
    """Carga anomalies.csv con las columnas disponibles y restaura booleanos."""
    path = _resolve_path(path_value)
    if not path.exists():
        return pd.DataFrame()

    available = _read_csv_header(path)
    usecols   = [c for c in ANOMALIES_USECOLS if c in available]

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = _restore_boolean_columns(df, ("es_hora_pico", "es_fin_semana",
                                       "z_score_anomaly", "iqr_anomaly",
                                       "speed_drop_anomaly", "isolation_forest_anomaly",
                                       "dbscan_anomaly"))

    # Construir timestamp completo si fecha_trafico está disponible
    if "fecha_trafico" in df.columns and "hora" in df.columns:
        df["fecha_trafico"] = pd.to_datetime(df["fecha_trafico"], errors="coerce")
        df["timestamp"] = df["fecha_trafico"] + pd.to_timedelta(
            df["hora"].fillna(0).astype(int), unit="h"
        )

    return df


def _run_anomaly_detection_in_dashboard(
    master: pd.DataFrame,
    config: dict,
    max_rows: int = 30_000,
) -> bool:
    """
    Ejecuta AnomalyDetector sobre el master ya cargado en el dashboard y
    persiste anomalies.csv. Retorna True si tuvo éxito.

    Se llama directamente desde la UI (botón), por lo que no está cacheada.
    """
    try:
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector(config=config)
        detector.run(master, max_rows=max_rows)
        return True
    except Exception as exc:
        st.error(f"Error durante la detección de anomalías: {exc}")
        return False


def _render_anomalias(
    df_anom: pd.DataFrame,
    master: pd.DataFrame,
    config: dict,
) -> None:
    """Renderiza la sección 'Anomalías de Congestión' del dashboard."""

    st.header("Anomalías de Congestión Vial")
    
    # Filtro independiente: Comuna
    filter_relationships = _build_filter_relationships(master)
    comunas_opts = filter_relationships.get("comunas", [])
    
    selected_comuna_anom = st.selectbox(
        "Filtrar Anomalías por Comuna",
        [COMUNA_FILTER_ALL, *comunas_opts],
        key="anom_filter_comuna",
        help="Selecciona una comuna para ver sus anomalías"
    )
    
    st.caption(
        "Detección combinada: Z-Score rodante · IQR por franja · "
        "Degradación de velocidad · Isolation Forest · DBSCAN espaciotemporal"
    )

    if df_anom.empty:
        st.warning(
            "El archivo `anomalies.csv` aún no existe. "
            "Puedes generarlo ahora directamente desde el dashboard "
            "sin necesidad de re-ejecutar el pipeline completo."
        )

        n_total = len(master)

        st.markdown("#### ⚙️ Configuración de muestra")
        st.caption(
            f"El master tiene **{n_total:,} registros**. "
            "Usar el total puede ser lento o colapsar equipos con poca RAM. "
            "Selecciona cuántos analizar — el muestreo es **estratificado por comuna** "
            "para conservar representatividad."
        )

        # Slider de tamaño de muestra
        opciones = {
            f"Rápido  — 10 000 registros (~15 s)": 10_000,
            f"Normal  — 30 000 registros (~45 s)": 30_000,
            f"Completo — 60 000 registros (~90 s)": 60_000,
            f"Máximo  — todos ({n_total:,} registros, puede ser muy lento)": 0,
        }
        seleccion = st.radio(
            "Tamaño de muestra",
            list(opciones.keys()),
            index=1,          # "Normal" por defecto
            help="A mayor muestra, más precisión pero más tiempo y RAM.",
        )
        max_rows = opciones[seleccion]

        label_rows = f"{max_rows:,}" if max_rows > 0 else f"todos ({n_total:,})"
        st.info(f"Se analizarán **{label_rows} registros**.")

        if st.button("🔍 Generar anomalías ahora", type="primary", use_container_width=False):
            with st.spinner(
                "Ejecutando detección de anomalías (Z-Score · IQR · Speed-drop · "
                "Isolation Forest · DBSCAN)… Esto puede tardar hasta 2 min."
            ):
                ok = _run_anomaly_detection_in_dashboard(master, config, max_rows=max_rows)

            if ok:
                st.success("✅ anomalies.csv generado correctamente. Recargando sección…")
                st.cache_data.clear()
                st.rerun()
        return

    # ── Aplicar filtro de comuna ────────────────────────
    df_view = df_anom.copy()
    if selected_comuna_anom != COMUNA_FILTER_ALL and "nombre_comuna" in df_view.columns:
        df_view = df_view[df_view["nombre_comuna"] == selected_comuna_anom]

    if df_view.empty:
        st.info("Sin anomalías para la comuna seleccionada.")
        return

    total_anom    = len(df_view)
    total_orig    = len(df_anom)
    pct_anom      = round(100 * total_anom / max(total_orig, 1), 2)
    n_criticos    = int((df_view["severity_level"] == "CRÍTICO").sum()) if "severity_level" in df_view.columns else 0

    # ── KPI Cards ────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4 = st.columns(4)

    _kpi_card(kc1, "Anomalías en vista", f"{total_anom:,}")
    _kpi_card(kc2, "Nivel CRÍTICO", f"{n_criticos:,}", f"{round(100*n_criticos/max(total_anom,1),1)} % del total")

    if "corredor" in df_view.columns and "severity_level" in df_view.columns:
        criticos_df = df_view[df_view["severity_level"] == "CRÍTICO"]
        if not criticos_df.empty:
            top_corr = criticos_df["corredor"].value_counts().idxmax()
        else:
            top_corr = df_view.groupby("corredor")["anomaly_score"].mean().idxmax() if "anomaly_score" in df_view.columns else "N/D"
        _kpi_card(kc3, "Corredor más crítico", top_corr)
    else:
        _kpi_card(kc3, "Corredor más crítico", "N/D")

    if "hora" in df_view.columns and "anomaly_score" in df_view.columns:
        hora_pico_anom = int(df_view.groupby("hora")["anomaly_score"].mean().idxmax())
        _kpi_card(kc4, "Hora con más picos", f"{hora_pico_anom:00d}:00 h")
    else:
        _kpi_card(kc4, "Hora con más picos", "N/D")

    st.markdown("---")

    # ── 1. Heatmap horario: hora × día de semana ─────────────────────
    st.subheader("Concentración horaria de anomalías (CRÍTICO + ALTO)")

    if "hora" in df_view.columns and "dia_num" in df_view.columns and "severity_level" in df_view.columns:
        df_heat = df_view[df_view["severity_level"].isin(["CRÍTICO", "ALTO"])].copy()
        if not df_heat.empty:
            df_heat["dia_label"] = (
                df_heat["dia_num"]
                .map(DIA_LABELS)
                .fillna("?")
            )
            heat_pivot = (
                df_heat.groupby(["hora", "dia_label"])
                .size()
                .reset_index(name="count")
                .pivot(index="hora", columns="dia_label", values="count")
                .reindex(index=range(24))
                .reindex(columns=[DIA_LABELS[d] for d in sorted(DIA_LABELS)], fill_value=0)
                .fillna(0)
                .astype(int)
            )

            fig_heat = px.imshow(
                heat_pivot,
                color_continuous_scale="RdYlGn_r",
                labels=dict(x="Día de semana", y="Hora del día", color="Anomalías"),
                title="Frecuencia de anomalías CRÍTICAS y ALTAS por hora y día",
                aspect="auto",
                text_auto=True,
            )
            fig_heat.update_layout(
                height=520,
                xaxis=dict(side="top"),
                margin=dict(l=10, r=10, t=60, b=10),
                coloraxis_colorbar=dict(title="Cantidad"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Sin anomalías CRÍTICAS o ALTAS para el heatmap con los filtros actuales.")
    else:
        st.info("Sin columnas `hora`, `dia_num` o `severity_level` para el heatmap.")

    # ── 3. Tabla de alertas recientes ────────────────────────────────
    st.subheader("Alertas recientes — Top 50 por Anomaly Score")

    display_cols = [
        c for c in (
            "fecha_trafico", "hora", "corredor", "nombre_comuna",
            "velocidad_km_h", "intensidad", "icv",
            "anomaly_score", "severity_level",
        )
        if c in df_view.columns
    ]

    if display_cols and "anomaly_score" in df_view.columns:
        df_top50 = (
            df_view[display_cols]
            .sort_values("anomaly_score", ascending=False)
            .head(50)
            .reset_index(drop=True)
        )

        def _row_style(row):
            sev = row.get("severity_level", "NORMAL")
            if sev == "CRÍTICO":
                return ["background-color: #ffd6d6; color: #7b0000"] * len(row)
            if sev == "ALTO":
                return ["background-color: #ffe8cc; color: #7a3900"] * len(row)
            if sev == "MODERADO":
                return ["background-color: #fffac7; color: #5a4a00"] * len(row)
            return [""] * len(row)

        col_cfg: dict = {}
        if "anomaly_score" in df_top50.columns:
            col_cfg["anomaly_score"] = st.column_config.ProgressColumn(
                "Anomaly Score", min_value=0, max_value=100, format="%.1f"
            )
        if "velocidad_km_h" in df_top50.columns:
            col_cfg["velocidad_km_h"] = st.column_config.NumberColumn(
                "Velocidad (km/h)", format="%.1f"
            )
        if "intensidad" in df_top50.columns:
            col_cfg["intensidad"] = st.column_config.NumberColumn(
                "Intensidad (veh/h)", format="%.0f"
            )
        if "icv" in df_top50.columns:
            col_cfg["icv"] = st.column_config.NumberColumn("ICV", format="%.1f")
        if "hora" in df_top50.columns:
            col_cfg["hora"] = st.column_config.NumberColumn("Hora", format="%d:00")

        styled = df_top50.style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True, column_config=col_cfg)
    else:
        st.info("Sin datos de anomalías con las columnas requeridas.")

    # ── 4. Desglose por detector ──────────────────────────────────────
    with st.expander("Desglose por detector", expanded=False):
        detector_cols = {
            "z_score_anomaly"          : "Z-Score rodante",
            "iqr_anomaly"              : "IQR franja horaria",
            "speed_drop_anomaly"       : "Caída de velocidad",
            "isolation_forest_anomaly" : "Isolation Forest",
            "dbscan_anomaly"           : "DBSCAN espaciotemporal",
        }
        det_rows = []
        for col, label in detector_cols.items():
            if col in df_view.columns:
                n = int(df_view[col].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
                det_rows.append({"Detector": label, "Anomalías detectadas": n,
                                  "% sobre vista": f"{100*n/max(len(df_view),1):.1f}%"})
        if det_rows:
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

    # ── 5. Recomendaciones ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Recomendaciones")
    st.caption(
        "Derivadas del análisis de ICV, hotspots y anomalías detectadas sobre 424 772 registros "
        "de julio 2020 · 49 corredores · 10 comunas."
    )

    _CARD_CSS = (
        "border-radius:10px; padding:16px 20px; margin-bottom:12px; "
        "font-family:'Segoe UI', sans-serif; font-size:0.92rem; line-height:1.7;"
    )
    _ROW_CSS  = "display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px;"
    _NAME_CSS = "font-weight:600; color:{fg}; min-width:230px;"
    _ACT_CSS  = "color:{fg}; flex:1; padding-left:12px;"

    def _rec_row(name: str, action: str, fg: str) -> str:
        return (
            f"<div style='{_ROW_CSS}'>"
            f"<span style='{_NAME_CSS.format(fg=fg)}'>{escape(name)}</span>"
            f"<span style='{_ACT_CSS.format(fg=fg)}'>→ {escape(action)}</span>"
            "</div>"
        )

    # ── Nivel Crítico ────────────────────────────────────────────────
    critico_rows = [
        ("Calle 30A",           "Gestión semafórica urgente (vel. 11 km/h · ICV 52)"),
        ("Colombia",            "Rutas alternas + carril bus exclusivo franja tarde"),
        ("Avenida 80 / Belén",  "Refuerzo de flota Metroplús + señalización dinámica"),
        ("Avenida Oriental",    "Coordinación semafórica + rutas alternas al centro"),
    ]
    critico_html = "".join(_rec_row(n, a, "#7b0000") for n, a in critico_rows)
    st.markdown(
        f"<div style='{_CARD_CSS} background:#ffd6d6; border-left:5px solid #d62728;'>"
        f"<div style='font-size:1rem; font-weight:700; color:#7b0000; margin-bottom:10px;'>"
        f"🔴 NIVEL CRÍTICO — Actuar de inmediato</div>"
        f"{critico_html}</div>",
        unsafe_allow_html=True,
    )

    # ── Nivel Alto ───────────────────────────────────────────────────
    alto_rows = [
        ("Carrera 65",                  "Ciclovía + semáforos coordinados con Colombia / San Juan"),
        ("Calle 65 / Calle 71 (Robledo)", "Infraestructura ciclista hacia el centro"),
        ("Autopista Norte",             "Redistribución anticipada hacia Regional Paralela"),
        ("Avenida El Poblado",          "Monitoreo continuo de anomalías (mayor recuento: 901)"),
    ]
    alto_html = "".join(_rec_row(n, a, "#7a3900") for n, a in alto_rows)
    st.markdown(
        f"<div style='{_CARD_CSS} background:#ffe8cc; border-left:5px solid #ff7f0e;'>"
        f"<div style='font-size:1rem; font-weight:700; color:#7a3900; margin-bottom:10px;'>"
        f"🟠 NIVEL ALTO — Planificar a 6 meses</div>"
        f"{alto_html}</div>",
        unsafe_allow_html=True,
    )

    # ── Comunas a priorizar ──────────────────────────────────────────
    comuna_rows = [
        ("Laureles Estadio", "Mayor concentración de hotspots (4 corredores en top-10)"),
        ("Belén",            "ICV pico más alto · mayor población de Medellín (197 593 hab.)"),
        ("Robledo",          "Ratio veh/hab. alto · sin alternativas modales actuales"),
        ("La Candelaria",    "Centro histórico · 11 803 registros en hora pico"),
    ]
    comuna_html = "".join(_rec_row(n, a, "#1a3a5c") for n, a in comuna_rows)
    st.markdown(
        f"<div style='{_CARD_CSS} background:#dbeafe; border-left:5px solid #2563eb;'>"
        f"<div style='font-size:1rem; font-weight:700; color:#1a3a5c; margin-bottom:10px;'>"
        f"🔵 COMUNAS A PRIORIZAR</div>"
        f"{comuna_html}</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "⏱ **Ventana clave:** martes–jueves · 14 h – 20 h concentra el **37,8 % de todos los "
        "eventos de congestión simultánea** del sistema. Cualquier medida de gestión de demanda "
        "(teletrabajo escalonado, precios dinámicos de parqueo, refuerzo de transporte público) "
        "en esa franja impacta directamente el indicador más crítico."
    )


def run_dashboard() -> None:
    loading_placeholder = st.empty()
    loading_placeholder.info("Leyendo archivos procesados y preparando caché del dashboard...")

    try:
        with st.spinner("Cargando datos principales..."):
            results = load_dashboard_datasets()
    except Exception as exc:
        st.error(f"Error al cargar los datos: {exc}")
        st.info("Ejecuta `python main.py` para generar los datos procesados.")
        st.stop()
    finally:
        loading_placeholder.empty()

    master: pd.DataFrame = results.get("master", pd.DataFrame())
    pasajeros: pd.DataFrame = results.get("pasajeros", pd.DataFrame())
    ciclorrutas: pd.DataFrame = results.get("ciclorrutas", pd.DataFrame())
    cfg = results.get("config", {})

    # Carga lazy de anomalies.csv (solo si existe; no bloquea si el pipeline aún no las generó)
    proc_dir_anom = _resolve_path(cfg.get("paths", {}).get("processed", "data/processed"))
    anomalies_path = proc_dir_anom / "anomalies.csv"
    anomalies: pd.DataFrame = _load_anomalies_data(anomalies_path)
    map_cfg = cfg.get("map", {"lat_centro": 6.2442, "lon_centro": -75.5812, "zoom": 11})
    ivc_cfg = cfg.get("ivc", {"capacidad_base": 2500, "top_corredores": 15})

    # ── CSS: identidad visual azul/blanco ────────────────────────────
    st.markdown("""
<style>
/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #eef2fc 0%, #e4ecf8 100%);
    border-right: 2px solid #c3d4f0;
}
[data-testid="stSidebar"] hr {
    border-color: #c3d4f0 !important;
    margin: 10px 0 !important;
}
[data-testid="stSidebar"] .stCaption p {
    color: #6b84b0 !important;
    font-size: 0.72rem !important;
}
/* Radio de navegación: items como tarjetas */
[data-testid="stSidebar"] .stRadio > div {
    gap: 3px;
}
[data-testid="stSidebar"] .stRadio label {
    background: #ffffff !important;
    border: 1px solid #c8d8f0 !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    color: #1e3a6e !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    margin: 1px 0 !important;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #dbeafe !important;
    border-color: #2563eb !important;
}
/* ── Métricas con acento azul superior ────────────── */
[data-testid="metric-container"] {
    background: #f5f8ff !important;
    border: 1px solid #dbeafe !important;
    border-top: 3px solid #1565c0 !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] p {
    color: #5372a0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: #0d2b5e !important;
    font-weight: 700 !important;
}
/* ── Títulos institucionales ──────────────────────── */
h1 {
    color: #0d2b5e !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
h2 {
    color: #1565c0 !important;
    border-bottom: 2px solid #dbeafe;
    padding-bottom: 6px;
    margin-top: 1.2rem;
}
h3 { color: #1e3a6e !important; }
/* ── Separadores más suaves ───────────────────────── */
hr {
    border-color: #dbeafe !important;
    opacity: 0.9 !important;
}
/* ── Botón primario en azul institucional ─────────── */
[data-testid="stBaseButton-primary"] > div > p {
    color: #ffffff !important;
}
button[kind="primary"] {
    background-color: #1565c0 !important;
    border-color: #1565c0 !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Cabecera de marca en el sidebar ──────────────────────────────
    st.sidebar.markdown(
        "<div style='"
        "background:linear-gradient(135deg,#1565c0 0%,#0d47a1 100%);"
        "border-radius:12px; padding:16px 18px; margin-bottom:18px;'>"
        "<div style='color:#ffffff; font-size:1.05rem; font-weight:700;"
        " letter-spacing:0.01em;'>🚦 Movilidad Medellín</div>"
        "<div style='color:#bbdefb; font-size:0.74rem; margin-top:4px;'>"
        "Panel de análisis vial · Jul. 2020</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Navegación ────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<p style='font-size:0.68rem; font-weight:700; color:#5372a0;"
        " letter-spacing:0.1em; text-transform:uppercase;"
        " margin:0 0 4px 0;'>Navegación</p>",
        unsafe_allow_html=True,
    )
    active_section = st.sidebar.radio(
        "Secciones",
        SECTION_OPTIONS,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("📊 Alcaldía de Medellín · Velocidad 2020 · Aforos 2018 · SIMM 2021")

    st.title("Dashboard de Movilidad — Medellín")
    st.markdown(
        "Análisis integrado del **Índice de Criticidad Vial (ICV)** "
        "sobre velocidad, intensidad y ocupación vehicular."
    )

    st.markdown("---")
    
    # KPIs generales sin filtros
    k1, k2, k3, k4 = st.columns(4)

    icv_prom = round(master["icv"].mean(), 1) if not master.empty and "icv" in master.columns else 0.0
    _kpi_card(k1, "ICV promedio de ciudad", f"{icv_prom:.1f} / 100")

    from src.pipeline import _rank_corredores
    top_corredores = _rank_corredores(master, cfg) if not master.empty else pd.DataFrame()
    if not top_corredores.empty:
        top1 = top_corredores.iloc[0]
        _kpi_card(k2, "Corredor más crítico", top1["corredor"], f"ICV {top1['icv_medio']:.1f}")
    else:
        _kpi_card(k2, "Corredor más crítico", "N/D")

    if not master.empty and "hora" in master.columns:
        hora_critica = int(master.groupby("hora")["icv"].mean().idxmax())
        _kpi_card(k3, "Hora más crítica", f"{hora_critica}:00 h")
    else:
        _kpi_card(k3, "Hora más crítica", "N/D")

    from src.pipeline import _identify_hotspots
    hotspots_total = (
        _identify_hotspots(master, cfg)
        if not master.empty and {"velocidad_km_h", "intensidad"}.issubset(master.columns)
        else pd.DataFrame()
    )
    n_hotspots = len(hotspots_total)
    _kpi_card(k4, "Hotspots detectados", f"{n_hotspots:,}")

    st.markdown("---")

    if active_section == "1. Corredores":
        _render_top_corredores(master, cfg, ivc_cfg)
    elif active_section == "2. Comunas":
        _render_comunas(master, cfg)
    elif active_section == "3. Hotspots":
        _render_hotspots(master, cfg)
    elif active_section == "4. Patrones Temporales":
        _render_patrones_temporales(master, cfg)
    elif active_section == "5. Tendencia Metro":
        _render_pasajeros(pasajeros)
    elif active_section == "6. Mapa de Calor Vial":
        _render_mapa_vial(master, ciclorrutas, cfg, map_cfg)
    elif active_section == "7. Velocidad y Flujo":
        _render_velocidad_flujo(master, cfg)
    elif active_section == "8. Anomalías de Congestión":
        _render_anomalias(anomalies, master, cfg)

    st.markdown("---")
    st.caption("Desarrollado para el Reto de Movilidad Urbana de Medellín · Datos: Alcaldía de Medellín")


if __name__ == "__main__":
    run_dashboard()
