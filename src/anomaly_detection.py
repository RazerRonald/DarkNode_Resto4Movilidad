"""
anomaly_detection.py — Detección de anomalías de congestión vial para Movilidad Medellín.

Módulo complementario al pipeline ETL existente. Puede usarse de forma autónoma:

    from src.anomaly_detection import AnomalyDetector
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    detector = AnomalyDetector(config)
    results  = detector.run(master_df)

Detectores implementados:
    1. Z-Score rodante (24 h) sobre velocidad, intensidad y ocupación.
    2. IQR por (corredor, franja_horaria, es_fin_semana) sobre ICV.
    3. Degradación abrupta de velocidad (≥ 30% caída en una hora).
    4. Isolation Forest multivariado, un modelo por comuna.
    5. DBSCAN espaciotemporal (condicional: requiere coords SIMM).

Salida: anomaly_score (0-100) y severity_level (NORMAL/MODERADO/ALTO/CRÍTICO).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─── Columnas mínimas requeridas ──────────────────────────────────────────────
REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "corredor",
        "velocidad_km_h",
        "intensidad",
        "icv",
        "hora",
        "es_fin_semana",
        "es_hora_pico",
        "franja_horaria",
    }
)

# ─── Columnas que este módulo añade al DataFrame ──────────────────────────────
ANOMALY_OUTPUT_COLS: list[str] = [
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

# ─── Mapeo de días para legibilidad en logs ───────────────────────────────────
_DIA_NOMBRE: dict[int, str] = {
    1: "Lun", 2: "Mar", 3: "Mié",
    4: "Jue", 5: "Vie", 6: "Sáb", 7: "Dom",
}


class AnomalyDetector:
    """
    Detector de anomalías de congestión vial para Medellín.

    Combina métodos estadísticos con Machine Learning para identificar picos
    atípicos de congestión en el espacio de corredores, franjas horarias y
    tipos de día.

    Parameters
    ----------
    config : dict
        Configuración del proyecto (cargada desde config.yaml).
        Debe incluir `paths.processed` para la persistencia de modelos y CSV.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        proc_dir = Path(config["paths"]["processed"])
        self.models_path = proc_dir / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.anomalies_path = proc_dir / "anomalies.csv"

        # ── Parámetros de cada detector ──────────────────────────────
        # Z-Score rodante
        self._z_threshold: float = 2.5
        self._z_window: int = 24
        self._z_min_obs: int = 24          # reducido: 48 → 24 para muestras pequeñas

        # IQR
        self._iqr_factor: float = 1.5

        # Degradación de velocidad
        self._speed_drop_threshold: float = 30.0   # % de caída

        # Isolation Forest — parámetros conservadores para hardware limitado
        self._if_contamination: float = 0.05
        self._if_n_estimators: int = 50    # árboles (100 original → 50 ligero)
        self._if_n_jobs: int = 1           # sin paralelismo para no saturar CPU
        self._if_max_per_group: int = 3_000  # máx filas por grupo para entrenar IF
        self._if_features: list[str] = [
            "velocidad_km_h", "intensidad", "ocupacion",
            "hora", "dia_num", "icv",
        ]

    # ═════════════════════════════════════════════════════════════════
    # UTILIDADES INTERNAS
    # ═════════════════════════════════════════════════════════════════

    def _sample_stratified(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        """
        Muestreo estratificado por `nombre_comuna` (o `corredor` si no existe).

        Cada grupo recibe una cuota proporcional a su tamaño, garantizando al
        menos 10 filas por grupo cuando haya suficientes datos.

        Parameters
        ----------
        df      : DataFrame completo a muestrear.
        max_rows: Número máximo de filas en la muestra.

        Returns
        -------
        pd.DataFrame muestreado y con índice reiniciado.
        """
        if max_rows <= 0 or len(df) <= max_rows:
            return df.copy()

        group_col = "nombre_comuna" if "nombre_comuna" in df.columns else "corredor"

        if group_col not in df.columns:
            sampled = df.sample(n=max_rows, random_state=42)
            logger.info("Muestra aleatoria: %d → %d registros.", len(df), max_rows)
            return sampled.reset_index(drop=True)

        total = len(df)
        partes = []
        for _, grp in df.groupby(group_col, sort=False):
            cuota = max(10, int(max_rows * len(grp) / total))
            n     = min(len(grp), cuota)
            partes.append(grp.sample(n=n, random_state=42))

        sampled = pd.concat(partes, ignore_index=True)
        logger.info(
            "Muestra estratificada por %s: %d → %d registros (factor de reducción %.1fx).",
            group_col, total, len(sampled), total / max(len(sampled), 1),
        )
        return sampled

    @staticmethod
    def _to_bool(s: pd.Series) -> pd.Series:
        """
        Convierte una Serie a bool de forma robusta.

        Maneja: dtype bool, dtype boolean (nullable), strings "true"/"false",
        enteros 0/1, y cualquier variante con o sin tilde.
        """
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        if hasattr(s, "dtype") and str(s.dtype) == "boolean":
            return s.fillna(False).astype(bool)
        truthy = {"true", "1", "yes", "si", "t", "y"}
        return s.astype(str).str.strip().str.lower().isin(truthy)

    @staticmethod
    def _safe_name(value: object) -> str:
        """Genera un nombre de archivo seguro a partir de un valor arbitrario."""
        text = str(value)
        replacements = {
            " ": "_", "/": "_", "\\": "_",
            "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "ñ": "n", "Ñ": "N",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return "".join(ch for ch in text if ch.isalnum() or ch == "_")

    def _sort_time(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Ordena un grupo por fecha_trafico → hora (si están disponibles)."""
        sort_cols = [c for c in ("fecha_trafico", "hora") if c in grp.columns]
        return grp.sort_values(sort_cols) if sort_cols else grp

    # ═════════════════════════════════════════════════════════════════
    # SECCIÓN 1 — DETECTORES ESTADÍSTICOS
    # ═════════════════════════════════════════════════════════════════

    def _detect_rolling_zscore(self, df: pd.DataFrame) -> pd.Series:
        """
        Rolling Z-Score por corredor (ventana 24 h).

        Marca como anomalía si |z| > 2.5 en **al menos 2 de 3** variables
        (velocidad_km_h, intensidad, ocupacion) de forma simultánea.

        Corredores con < 48 observaciones se omiten con un warning.

        Returns
        -------
        pd.Series[bool] con el mismo índice que df.
        """
        result = pd.Series(False, index=df.index, dtype=bool)
        vars_target = [v for v in ("velocidad_km_h", "intensidad", "ocupacion") if v in df.columns]

        if len(vars_target) < 2:
            logger.warning("Z-Score: variables disponibles insuficientes (%s). Omitiendo.", vars_target)
            return result

        min_periods = max(3, self._z_window // 4)

        for corredor, grp in df.groupby("corredor", sort=False):
            if len(grp) < self._z_min_obs:
                logger.warning(
                    "Z-Score: corredor '%s' con %d obs (mín=%d). Omitiendo.",
                    corredor, len(grp), self._z_min_obs,
                )
                continue

            grp_s = self._sort_time(grp)
            flag_counts = pd.Series(0, index=grp_s.index, dtype=int)

            for var in vars_target:
                s = grp_s[var].astype(float)
                roll_mean = s.rolling(self._z_window, min_periods=min_periods).mean()
                roll_std  = s.rolling(self._z_window, min_periods=min_periods).std()
                z_abs = ((s - roll_mean) / roll_std.replace(0, np.nan)).abs()
                flag_counts += (z_abs > self._z_threshold).fillna(False).astype(int)

            # Anomalía si al menos 2 variables superan el umbral
            result.loc[grp_s.index[flag_counts >= 2]] = True

        n_det = int(result.sum())
        logger.info("Z-Score: %d anomalías detectadas (%.2f%%)", n_det, 100 * n_det / max(len(df), 1))
        return result

    def _detect_iqr(self, df: pd.DataFrame) -> pd.Series:
        """
        IQR sobre `icv` agrupando por (corredor, franja_horaria, es_fin_semana).

        Límite superior = Q3 + 1.5 × IQR.
        Marca anomalía si icv > límite superior del grupo correspondiente.

        Returns
        -------
        pd.Series[bool] con el mismo índice que df.
        """
        result = pd.Series(False, index=df.index, dtype=bool)

        if "icv" not in df.columns:
            logger.warning("IQR: columna 'icv' no disponible. Omitiendo.")
            return result

        group_cols = [c for c in ("corredor", "franja_horaria", "es_fin_semana") if c in df.columns]
        if not group_cols:
            logger.warning("IQR: sin columnas de agrupación disponibles.")
            return result

        # Normalizar es_fin_semana a string para groupby estable (evita problemas con BooleanDtype)
        work = df[group_cols + ["icv"]].copy()
        if "es_fin_semana" in work.columns:
            work["es_fin_semana"] = self._to_bool(work["es_fin_semana"]).astype(str)

        for _, grp in work.groupby(group_cols, sort=False):
            icv_clean = grp["icv"].dropna()
            if len(icv_clean) < 4:
                continue
            q1    = icv_clean.quantile(0.25)
            q3    = icv_clean.quantile(0.75)
            upper = q3 + self._iqr_factor * (q3 - q1)
            outlier_idx = grp.index[grp["icv"] > upper]
            result.loc[outlier_idx] = True

        n_det = int(result.sum())
        logger.info("IQR: %d anomalías detectadas (%.2f%%)", n_det, 100 * n_det / max(len(df), 1))
        return result

    def _detect_speed_drop(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Detecta degradaciones abruptas de velocidad (≥ 30% de caída en una hora).

        Calcula `(v_t - v_{t-1}) / v_{t-1} * 100` por corredor, ordenando por
        fecha_trafico + hora para garantizar la secuencia temporal correcta.

        Returns
        -------
        anomaly  : pd.Series[bool]   — True donde la caída ≥ umbral
        drop_pct : pd.Series[float]  — Porcentaje de cambio (negativo = caída)
        """
        anomaly  = pd.Series(False, index=df.index, dtype=bool)
        drop_pct = pd.Series(np.nan, index=df.index, dtype=float)

        if "velocidad_km_h" not in df.columns:
            logger.warning("Speed-drop: 'velocidad_km_h' no disponible. Omitiendo.")
            return anomaly, drop_pct

        for corredor, grp in df.groupby("corredor", sort=False):
            if len(grp) < 2:
                continue

            grp_s = self._sort_time(grp)
            v     = grp_s["velocidad_km_h"].astype(float)
            pct   = v.pct_change() * 100                    # + sube, - cae

            drop_mask = pct <= -self._speed_drop_threshold
            anomaly.loc[grp_s.index[drop_mask]] = True
            drop_pct.loc[grp_s.index] = pct.values

        n_det = int(anomaly.sum())
        logger.info("Speed-drop: %d anomalías detectadas (%.2f%%)", n_det, 100 * n_det / max(len(df), 1))
        return anomaly, drop_pct

    # ═════════════════════════════════════════════════════════════════
    # SECCIÓN 2 — DETECTORES ML
    # ═════════════════════════════════════════════════════════════════

    def _detect_isolation_forest(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Isolation Forest multivariado, **un modelo por `nombre_comuna`**.

        Si el modelo serializado existe en `data/processed/models/`, lo carga
        (evita re-entrenamiento en el dashboard). Si no existe, lo entrena y
        persiste con joblib.

        Features: velocidad_km_h, intensidad, ocupacion, hora, dia_num, icv.

        Returns
        -------
        anomaly : pd.Series[bool]   — True para outliers (prediction == -1)
        score   : pd.Series[float]  — score_samples (más negativo = más anómalo)
        """
        anomaly = pd.Series(False, index=df.index, dtype=bool)
        score   = pd.Series(0.0,   index=df.index, dtype=float)

        group_col = "nombre_comuna" if "nombre_comuna" in df.columns else "corredor"
        features  = [f for f in self._if_features if f in df.columns]

        if len(features) < 3:
            logger.warning("IF: features disponibles insuficientes (%s). Omitiendo.", features)
            return anomaly, score

        for group_name, grp in df.groupby(group_col, sort=False):
            X       = grp[features].copy()
            n_valid = int(X.dropna().shape[0])

            if n_valid < 10:
                logger.warning(
                    "IF: grupo '%s' con solo %d obs válidas. Omitiendo.", group_name, n_valid
                )
                continue

            # Imputar NaN con la mediana de cada feature en el grupo
            X_filled = X.fillna(X.median())

            safe     = self._safe_name(group_name)
            clf_path = self.models_path / f"if_{safe}.joblib"

            clf: Optional[IsolationForest] = None

            if clf_path.exists():
                try:
                    clf = joblib.load(clf_path)
                    logger.debug("IF: modelo cargado desde %s", clf_path)
                except Exception as exc:
                    logger.warning("IF: error cargando %s: %s. Reentrenando.", clf_path, exc)
                    clf = None

            if clf is None:
                try:
                    # Submuestra para entrenamiento: máx _if_max_per_group filas
                    X_train = (
                        X_filled.sample(n=self._if_max_per_group, random_state=42)
                        if len(X_filled) > self._if_max_per_group
                        else X_filled
                    )
                    clf = IsolationForest(
                        contamination=self._if_contamination,
                        random_state=42,
                        n_estimators=self._if_n_estimators,
                        n_jobs=self._if_n_jobs,
                    )
                    clf.fit(X_train)
                    joblib.dump(clf, clf_path)
                    logger.info("IF: modelo entrenado (%d filas) → %s", len(X_train), clf_path)
                except Exception as exc:
                    logger.error("IF: entrenamiento fallido para '%s': %s", group_name, exc)
                    # No lanzar; dejar los valores por defecto (False, 0.0)
                    continue

            try:
                preds  = clf.predict(X_filled)          # -1 anomalía / +1 normal
                scores = clf.score_samples(X_filled)    # más negativo = más anómalo
                anomaly.loc[grp.index] = preds == -1
                score.loc[grp.index]   = scores
            except Exception as exc:
                logger.error("IF: predicción fallida para '%s': %s", group_name, exc)

        n_det = int(anomaly.sum())
        logger.info("IF: %d anomalías detectadas (%.2f%%)", n_det, 100 * n_det / max(len(df), 1))
        return anomaly, score

    def _detect_dbscan(self, df: pd.DataFrame) -> pd.Series:
        """
        DBSCAN espaciotemporal sobre [simm_lat, simm_lon, hora] normalizados.

        Solo se activa si las columnas de coordenadas SIMM (WGS84) están
        presentes **y** representan al menos el 10% de los registros.
        Puntos con label == -1 son outliers espaciotemporales.

        Returns
        -------
        pd.Series[bool] con el mismo índice que df.
        """
        result = pd.Series(False, index=df.index, dtype=bool)

        coord_cols = ("simm_lat", "simm_lon")
        if not all(c in df.columns for c in coord_cols):
            logger.info("DBSCAN: columnas %s no disponibles. Omitiendo.", coord_cols)
            return result

        valid_mask  = df[list(coord_cols)].notna().all(axis=1)
        n_valid     = int(valid_mask.sum())
        min_needed  = max(10, int(0.10 * len(df)))

        if n_valid < min_needed:
            logger.info(
                "DBSCAN: %d/%d registros con coordenadas (mín=%d). Omitiendo.",
                n_valid, len(df), min_needed,
            )
            return result

        df_valid = df[valid_mask].copy()
        X_raw    = df_valid[["simm_lat", "simm_lon", "hora"]].fillna(0).astype(float)

        try:
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            labels   = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1).fit_predict(X_scaled)
            outlier_mask = labels == -1
            result.loc[df_valid.index[outlier_mask]] = True
            logger.info(
                "DBSCAN: %d outliers / %d puntos válidos (%.1f%%)",
                int(outlier_mask.sum()), n_valid, 100 * outlier_mask.mean(),
            )
        except Exception as exc:
            logger.error("DBSCAN: error durante clustering: %s", exc)

        return result

    # ═════════════════════════════════════════════════════════════════
    # SECCIÓN 3 — SCORING AGREGADO
    # ═════════════════════════════════════════════════════════════════

    def _compute_anomaly_score(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Puntuación unificada `anomaly_score` (0-100) y `severity_level`.

        Fórmula de pesos:
            z_score_anomaly          × 25
            iqr_anomaly              × 20
            speed_drop_anomaly       × 30
            isolation_forest_anomaly × 25
        (dbscan_anomaly no entra en el score; es un indicador complementario)

        Multiplicadores de severidad:
            × 1.2 si es_hora_pico
            × 1.1 si no es fin de semana (días hábiles = mayor impacto operativo)

        Niveles de severidad:
            NORMAL   → score < 25
            MODERADO → 25 ≤ score < 50
            ALTO     → 50 ≤ score < 75
            CRÍTICO  → score ≥ 75

        Returns
        -------
        anomaly_score : pd.Series[float]  — 0–100
        severity_level: pd.Series[str]    — uno de los 4 niveles
        """
        base = (
            df["z_score_anomaly"].astype(int)          * 25 +
            df["iqr_anomaly"].astype(int)              * 20 +
            df["speed_drop_anomaly"].astype(int)       * 30 +
            df["isolation_forest_anomaly"].astype(int) * 25
        ).astype(float)

        mult = pd.Series(1.0, index=df.index, dtype=float)

        if "es_hora_pico" in df.columns:
            is_pico = self._to_bool(df["es_hora_pico"])
            mult   *= np.where(is_pico, 1.2, 1.0)

        if "es_fin_semana" in df.columns:
            is_fds = self._to_bool(df["es_fin_semana"])
            mult  *= np.where(is_fds, 1.0, 1.1)

        raw = (base * mult).clip(0, 100).round(2)

        severity = pd.cut(
            raw,
            bins=[-np.inf, 25.0, 50.0, 75.0, np.inf],
            labels=["NORMAL", "MODERADO", "ALTO", "CRÍTICO"],
            right=False,
        ).astype(str)

        return raw, severity

    # ═════════════════════════════════════════════════════════════════
    # SECCIÓN 4 — RESÚMENES Y EXPORTS
    # ═════════════════════════════════════════════════════════════════

    def _build_summaries(self, df_full: pd.DataFrame) -> dict:
        """
        Genera los DataFrames de resumen y el dict de estadísticas globales.

        Parameters
        ----------
        df_full : DataFrame con todas las columnas del módulo ya calculadas.

        Returns
        -------
        dict con claves: df_anomalies, top_corridors, top_communes,
                         hourly_pattern, summary_stats.
        """
        df_anom = df_full[df_full["severity_level"] != "NORMAL"].copy()

        # ── Top 10 corredores con más anomalías CRÍTICAS ──────────────
        if not df_anom.empty and "corredor" in df_anom.columns:
            top_corridors = (
                df_anom[df_anom["severity_level"] == "CRÍTICO"]
                .groupby("corredor")
                .agg(
                    n_criticos  =("anomaly_score", "count"),
                    score_medio =("anomaly_score", "mean"),
                )
                .reset_index()
                .nlargest(10, "n_criticos")
                .reset_index(drop=True)
            )
        else:
            top_corridors = pd.DataFrame(
                columns=["corredor", "n_criticos", "score_medio"]
            )

        # ── Top 5 comunas por anomaly_score promedio ──────────────────
        commune_col = "nombre_comuna" if "nombre_comuna" in df_anom.columns else None
        if commune_col and not df_anom.empty:
            top_communes = (
                df_anom.groupby(commune_col)
                .agg(
                    score_medio =("anomaly_score", "mean"),
                    n_anomalias =("anomaly_score", "count"),
                )
                .reset_index()
                .nlargest(5, "score_medio")
                .reset_index(drop=True)
            )
        else:
            top_communes = pd.DataFrame(
                columns=["nombre_comuna", "score_medio", "n_anomalias"]
            )

        # ── Distribución horaria (pivot: hora × severity_level) ───────
        if not df_anom.empty and "hora" in df_anom.columns:
            hc = (
                df_anom.groupby(["hora", "severity_level"])
                .size()
                .reset_index(name="count")
            )
            hourly_pattern = (
                hc.pivot(index="hora", columns="severity_level", values="count")
                .fillna(0)
                .astype(int)
            )
        else:
            hourly_pattern = pd.DataFrame()

        # ── Estadísticas globales ─────────────────────────────────────
        summary_stats: dict = {
            "total_anomalias": len(df_anom),
            "total_registros": len(df_full),
            "pct_anomalos": round(100 * len(df_anom) / max(len(df_full), 1), 2),
            "por_severidad": df_full["severity_level"].value_counts().to_dict(),
            "por_detector": {
                "z_score"          : int(df_full["z_score_anomaly"].sum()),
                "iqr"              : int(df_full["iqr_anomaly"].sum()),
                "speed_drop"       : int(df_full["speed_drop_anomaly"].sum()),
                "isolation_forest" : int(df_full["isolation_forest_anomaly"].sum()),
                "dbscan"           : int(df_full["dbscan_anomaly"].sum()),
            },
        }

        return {
            "df_anomalies"  : df_anom,
            "top_corridors" : top_corridors,
            "top_communes"  : top_communes,
            "hourly_pattern": hourly_pattern,
            "summary_stats" : summary_stats,
        }

    # ═════════════════════════════════════════════════════════════════
    # MÉTODO PÚBLICO PRINCIPAL
    # ═════════════════════════════════════════════════════════════════

    def run(self, df: pd.DataFrame, max_rows: int = 30_000) -> dict:
        """
        Ejecuta la detección de anomalías completa sobre el DataFrame maestro.

        El DataFrame de entrada **no es modificado** (se trabaja sobre una copia).

        Parameters
        ----------
        df       : pd.DataFrame
            DataFrame maestro post-ICV (master.csv) con al menos las columnas
            definidas en REQUIRED_COLS.
        max_rows : int, default 30_000
            Máximo de registros a analizar. Si el DataFrame tiene más filas,
            se aplica un muestreo estratificado por comuna antes de cualquier
            detector. Usa 0 para procesar todos (no recomendado en hardware
            limitado con > 100 000 registros).

        Returns
        -------
        dict con claves:
            df_anomalies   — pd.DataFrame con filas de severity != NORMAL
            top_corridors  — Top 10 corredores por frecuencia de CRÍTICOS
            top_communes   — Top 5 comunas por anomaly_score promedio
            hourly_pattern — pd.DataFrame pivot hora × severity
            summary_stats  — dict con conteos globales y por detector
            models_path    — str ruta donde se guardaron los modelos IF
        """
        logger.info("AnomalyDetector.run() — %d registros recibidos.", len(df))

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning(
                "Columnas faltantes (se usarán valores por defecto): %s", missing
            )

        # ── Muestreo estratificado ───────────────────────────────────
        work = self._sample_stratified(df, max_rows)

        # ── 1. Z-Score rodante ───────────────────────────────────────
        logger.info("  [1/5] Ejecutando Z-Score rodante...")
        work["z_score_anomaly"] = self._detect_rolling_zscore(work)

        # ── 2. IQR por franja horaria ────────────────────────────────
        logger.info("  [2/5] Ejecutando IQR por franja horaria...")
        work["iqr_anomaly"] = self._detect_iqr(work)

        # ── 3. Caída abrupta de velocidad ────────────────────────────
        logger.info("  [3/5] Ejecutando detección de caída de velocidad...")
        work["speed_drop_anomaly"], work["speed_drop_pct"] = self._detect_speed_drop(work)

        # ── 4. Isolation Forest ──────────────────────────────────────
        logger.info("  [4/5] Ejecutando Isolation Forest por comuna...")
        work["isolation_forest_anomaly"], work["isolation_score"] = (
            self._detect_isolation_forest(work)
        )

        # ── 5. DBSCAN espaciotemporal ────────────────────────────────
        logger.info("  [5/5] Ejecutando DBSCAN espaciotemporal...")
        work["dbscan_anomaly"] = self._detect_dbscan(work)

        # ── Scoring agregado ─────────────────────────────────────────
        logger.info("  Calculando anomaly_score y severity_level...")
        work["anomaly_score"], work["severity_level"] = self._compute_anomaly_score(work)

        # ── Exportar anomalías a CSV ─────────────────────────────────
        df_anom = work[work["severity_level"] != "NORMAL"].copy()
        try:
            df_anom.to_csv(self.anomalies_path, index=False)
            logger.info(
                "  anomalies.csv exportado: %d filas → %s",
                len(df_anom), self.anomalies_path,
            )
        except Exception as exc:
            logger.error("  Error exportando anomalies.csv: %s", exc)

        # ── Generar resúmenes ────────────────────────────────────────
        results = self._build_summaries(work)
        results["models_path"] = str(self.models_path)

        stats = results["summary_stats"]
        logger.info(
            "  Resumen final: %d anomalías (%.1f%%) | CRÍTICO=%d | ALTO=%d | MODERADO=%d",
            stats["total_anomalias"],
            stats["pct_anomalos"],
            stats["por_severidad"].get("CRÍTICO", 0),
            stats["por_severidad"].get("ALTO", 0),
            stats["por_severidad"].get("MODERADO", 0),
        )

        return results
