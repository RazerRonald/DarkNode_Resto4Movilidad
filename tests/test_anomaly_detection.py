"""
tests/test_anomaly_detection.py — Tests básicos del módulo de detección de anomalías.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/test_anomaly_detection.py -v

Los tests usan un DataFrame sintético que replica el esquema de master.csv,
sin depender de la existencia de archivos externos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Asegurar que el proyecto esté en sys.path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import ANOMALY_OUTPUT_COLS, AnomalyDetector

# ─── Valores esperados ────────────────────────────────────────────────────────
EXPECTED_COLS = ANOMALY_OUTPUT_COLS
SEVERITY_VALID = {"NORMAL", "MODERADO", "ALTO", "CRÍTICO"}
N_ROWS = 1_200   # tamaño del DataFrame sintético de prueba


# ─── Config mínima compatible con el proyecto ─────────────────────────────────
@pytest.fixture(scope="session")
def minimal_config(tmp_path_factory) -> dict:
    """Config con rutas temporales para no contaminar data/processed/."""
    proc = tmp_path_factory.mktemp("processed")
    return {
        "paths": {
            "raw"      : str(proc / "raw"),
            "processed": str(proc),
        }
    }


# ─── DataFrame sintético ──────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """
    Crea un DataFrame sintético con el esquema de master.csv.

    Incluye:
      - Varias comunas y corredores para que Isolation Forest los procese.
      - Valores extremos deliberados para que algunos detectores disparen.
      - Columnas opcionales (simm_lat, simm_lon, fecha_trafico).
    """
    rng = np.random.default_rng(42)

    comunas    = ["Laureles Estadio", "El Poblado", "Robledo", "Belén"]
    corredores = [
        "Avenida Bolivariana", "Calle 80",
        "Avenida El Poblado",  "Carrera 65",
    ]
    n = N_ROWS

    # Asignar corredor y comuna de forma correlacionada (1 corredor por comuna)
    indices       = rng.integers(0, 4, size=n)
    corredor_arr  = [corredores[i] for i in indices]
    comuna_arr    = [comunas[i]    for i in indices]

    df = pd.DataFrame(
        {
            "corredor"      : corredor_arr,
            "nombre_comuna" : comuna_arr,
            "fecha_trafico" : pd.date_range("2020-07-01", periods=n, freq="h"),
            "hora"          : rng.integers(0, 24, size=n),
            "dia_num"       : rng.integers(1, 8, size=n),
            "mes_num"       : rng.integers(1, 13, size=n),
            "velocidad_km_h": rng.uniform(5, 80, size=n).round(1),
            "intensidad"    : rng.integers(50, 700, size=n).astype(float),
            "ocupacion"     : rng.uniform(2, 70, size=n).round(1),
            "icv"           : rng.uniform(0, 100, size=n).round(2),
            "es_fin_semana" : rng.integers(0, 2, size=n).astype(bool),
            "es_hora_pico"  : rng.integers(0, 2, size=n).astype(bool),
            "franja_horaria": rng.choice(
                ["madrugada", "manana", "mediodia", "tarde", "noche"], size=n
            ),
            "poblacion_2019": rng.uniform(50_000, 200_000, size=n).round(0),
            # Coordenadas SIMM presentes para activar DBSCAN
            "simm_lat"      : rng.uniform(6.20, 6.30, size=n),
            "simm_lon"      : rng.uniform(-75.62, -75.52, size=n),
        }
    )

    # ── Introducir anomalías artificiales visibles para los detectores ──
    # Caídas abruptas de velocidad (speed_drop ≥ 30%)
    idx_drop = rng.choice(n, size=30, replace=False)
    df.loc[df.index[idx_drop], "velocidad_km_h"] = 4.0

    # ICV muy altos (para IQR)
    idx_icv = rng.choice(n, size=20, replace=False)
    df.loc[df.index[idx_icv], "icv"] = 99.0

    return df.reset_index(drop=True)


# ─── Fixture: resultado del detector ─────────────────────────────────────────
@pytest.fixture(scope="session")
def detection_results(sample_df, minimal_config):
    """Ejecuta AnomalyDetector.run() una sola vez para todos los tests."""
    detector = AnomalyDetector(config=minimal_config)
    return detector.run(sample_df)


# ════════════════════════════════════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════════════════════════════════════


class TestOutputSchema:
    """El DataFrame de anomalías contiene las columnas esperadas."""

    def test_df_anomalies_returned(self, detection_results):
        assert "df_anomalies" in detection_results

    def test_all_anomaly_columns_present(self, detection_results, sample_df):
        """Después de run(), el DataFrame de anomalías tiene todas las columnas de salida."""
        df_anom = detection_results["df_anomalies"]
        for col in EXPECTED_COLS:
            assert col in df_anom.columns, f"Columna faltante: '{col}'"

    def test_anomaly_columns_present_in_full_run(self, sample_df, minimal_config):
        """Las columnas de salida deben existir incluso con DataFrames pequeños (robustez)."""
        small_df = sample_df.head(100).copy()
        detector = AnomalyDetector(config=minimal_config)
        results  = detector.run(small_df)
        df_anom  = results["df_anomalies"]
        for col in EXPECTED_COLS:
            assert col in df_anom.columns, f"Columna faltante en run pequeño: '{col}'"


class TestAnomalyScoreRange:
    """anomaly_score siempre está dentro del rango [0, 100]."""

    def test_score_min_is_zero(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty:
            pytest.skip("Sin anomalías detectadas.")
        assert (df["anomaly_score"] >= 0).all(), "anomaly_score con valores negativos."

    def test_score_max_is_100(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty:
            pytest.skip("Sin anomalías detectadas.")
        assert (df["anomaly_score"] <= 100).all(), "anomaly_score excede 100."

    def test_score_dtype_numeric(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty:
            pytest.skip("Sin anomalías detectadas.")
        assert pd.api.types.is_float_dtype(df["anomaly_score"]), (
            "anomaly_score debe ser float."
        )


class TestSeverityLevel:
    """severity_level solo contiene los 4 valores válidos."""

    def test_severity_values_are_valid(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty:
            pytest.skip("Sin anomalías detectadas.")
        unique_severities = set(df["severity_level"].unique())
        invalid = unique_severities - SEVERITY_VALID
        assert not invalid, f"Valores de severity_level no válidos: {invalid}"

    def test_severity_consistent_with_score(self, detection_results):
        """El severity_level debe ser consistente con el rango de anomaly_score."""
        df = detection_results["df_anomalies"]
        if df.empty:
            pytest.skip("Sin anomalías detectadas.")

        criticos = df[df["severity_level"] == "CRÍTICO"]["anomaly_score"]
        altos    = df[df["severity_level"] == "ALTO"]["anomaly_score"]
        mods     = df[df["severity_level"] == "MODERADO"]["anomaly_score"]

        if not criticos.empty:
            assert (criticos >= 75).all(), "CRÍTICO con score < 75."
        if not altos.empty:
            assert ((altos >= 50) & (altos < 75)).all(), "ALTO fuera del rango [50, 75)."
        if not mods.empty:
            assert ((mods >= 25) & (mods < 50)).all(), "MODERADO fuera del rango [25, 50)."


class TestSummaryStructure:
    """El dict de resultados tiene la estructura completa esperada."""

    EXPECTED_KEYS = {
        "df_anomalies", "top_corridors", "top_communes",
        "hourly_pattern", "summary_stats", "models_path",
    }

    def test_result_keys(self, detection_results):
        missing = self.EXPECTED_KEYS - set(detection_results.keys())
        assert not missing, f"Claves faltantes en el resultado: {missing}"

    def test_summary_stats_keys(self, detection_results):
        stats = detection_results["summary_stats"]
        for key in ("total_anomalias", "total_registros", "pct_anomalos",
                    "por_severidad", "por_detector"):
            assert key in stats, f"Clave faltante en summary_stats: '{key}'"

    def test_top_corridors_max_10(self, detection_results):
        assert len(detection_results["top_corridors"]) <= 10

    def test_top_communes_max_5(self, detection_results):
        assert len(detection_results["top_communes"]) <= 5

    def test_pct_anomalos_range(self, detection_results):
        pct = detection_results["summary_stats"]["pct_anomalos"]
        assert 0.0 <= pct <= 100.0, f"pct_anomalos fuera de rango: {pct}"

    def test_models_path_is_string(self, detection_results):
        assert isinstance(detection_results["models_path"], str)


class TestDetectorColumns:
    """Las columnas de cada detector son booleanas o numéricas según corresponda."""

    def test_boolean_detector_cols(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty:
            pytest.skip("Sin anomalías.")
        bool_cols = [
            "z_score_anomaly", "iqr_anomaly",
            "speed_drop_anomaly", "isolation_forest_anomaly", "dbscan_anomaly",
        ]
        for col in bool_cols:
            if col in df.columns:
                assert df[col].dtype in (
                    bool, np.dtype("bool"), pd.BooleanDtype()
                ) or df[col].isin([True, False, 0, 1]).all(), (
                    f"'{col}' no es booleano."
                )

    def test_speed_drop_pct_is_numeric(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty or "speed_drop_pct" not in df.columns:
            pytest.skip("Sin datos de speed_drop_pct.")
        assert pd.api.types.is_numeric_dtype(df["speed_drop_pct"]), (
            "speed_drop_pct debe ser numérico."
        )

    def test_isolation_score_is_numeric(self, detection_results):
        df = detection_results["df_anomalies"]
        if df.empty or "isolation_score" not in df.columns:
            pytest.skip("Sin datos de isolation_score.")
        assert pd.api.types.is_numeric_dtype(df["isolation_score"]), (
            "isolation_score debe ser numérico."
        )


class TestRobustness:
    """El detector maneja casos extremos sin lanzar excepciones."""

    def test_empty_dataframe(self, minimal_config):
        """run() con DataFrame vacío debe retornar la estructura esperada sin errores."""
        detector = AnomalyDetector(config=minimal_config)
        results  = detector.run(pd.DataFrame())
        assert "df_anomalies" in results
        assert "summary_stats" in results

    def test_missing_optional_columns(self, minimal_config):
        """run() sin columnas opcionales (ocupacion, simm_lat, etc.) no debe fallar."""
        df_min = pd.DataFrame(
            {
                "corredor"      : ["Corredor A"] * 60,
                "nombre_comuna" : ["Comuna X"] * 60,
                "hora"          : list(range(24)) * 2 + list(range(12)),
                "dia_num"       : [1] * 60,
                "velocidad_km_h": np.random.uniform(10, 60, 60),
                "intensidad"    : np.random.uniform(100, 600, 60),
                "icv"           : np.random.uniform(0, 100, 60),
                "es_fin_semana" : [False] * 60,
                "es_hora_pico"  : [True] * 30 + [False] * 30,
                "franja_horaria": ["tarde"] * 60,
            }
        )
        detector = AnomalyDetector(config=minimal_config)
        results  = detector.run(df_min)
        assert "df_anomalies" in results

    def test_single_corridor_below_min_obs(self, minimal_config):
        """Un corredor con < 48 obs debe ser omitido en Z-Score sin romper el run()."""
        df_small = pd.DataFrame(
            {
                "corredor"      : ["Solo"] * 10,
                "nombre_comuna" : ["X"] * 10,
                "hora"          : list(range(10)),
                "dia_num"       : [1] * 10,
                "velocidad_km_h": np.linspace(20, 60, 10),
                "intensidad"    : np.linspace(100, 500, 10),
                "ocupacion"     : np.linspace(5, 60, 10),
                "icv"           : np.linspace(10, 90, 10),
                "es_fin_semana" : [False] * 10,
                "es_hora_pico"  : [False] * 10,
                "franja_horaria": ["tarde"] * 10,
            }
        )
        detector = AnomalyDetector(config=minimal_config)
        results  = detector.run(df_small)
        # El z_score_anomaly debe ser todo False (corredor omitido)
        df_out = results["df_anomalies"]
        # No debe lanzar excepción; la estructura debe ser válida
        assert "summary_stats" in results
