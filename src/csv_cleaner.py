"""
csv_cleaner.py — Limpieza y normalización de texto en archivos CSV del proyecto
               Movilidad Medellín.

Reglas de transformación aplicadas (en orden):
    1. Conservar caracteres españoles válidos (á é í ó ú ñ ü ¿ ¡) — UTF-8 válido.
    2. Eliminar caracteres de control (\x00–\x1F, \x7F) excepto \\t \\n \\r.
    3. Eliminar símbolo de reemplazo Unicode (U+FFFD) y variantes.
    4. Eliminar caracteres de otros alfabetos (cirílico, árabe, CJK, etc.)
       que no corresponden al contexto de datos colombianos en español.
    5. Colapsar espacios múltiples consecutivos a uno solo.
    6. Eliminar espacios al inicio y final de cada celda (strip).
    7. Eliminar comillas redundantes o caracteres de escape rotos al inicio/fin.

Archivos procesados (data/raw):
    - Aforos_Vehiculares.csv            (19 MB, UTF-8, coma)
    - Ciclorrutas.csv                   (5.7 KB, UTF-8, coma)
    - Pasajeros_movilizados.csv         (9.2 KB, UTF-8-BOM, coma)
    - proyecciones_de_poblacion_*.csv   (22 KB, UTF-8, coma)
    - simmtrafficdata.csv               (92 KB, ASCII/UTF-8, coma)
    - velocidad_e_intensidad_*.csv      (76 MB, ASCII/UTF-8, coma) → chunked

Uso:
    python src/csv_cleaner.py                        # limpia todos los CSV
    python src/csv_cleaner.py --file aforos          # limpia uno específico
    python src/csv_cleaner.py --file velocidad --dry-run  # solo muestra stats
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterator

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Rutas del proyecto
# ──────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw"
CLEAN_DIR  = ROOT / "data" / "processed" / "clean"
LOG_PATH   = ROOT / "data" / "outputs" / "cleaning_log.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Definición de archivos a procesar
# ──────────────────────────────────────────────────────────────────────────────
# "text_cols": columnas cuyos valores se limpian.
#   None  → detección automática (dtype object, no numérico).
#   lista → solo esas columnas.
# "skip_cols": columnas object que NO deben tocarse (IDs, coordenadas, fechas).

FILES: dict[str, dict] = {
    "aforos": {
        "filename" : "Aforos_Vehiculares.csv",
        "encoding" : "utf-8",
        "separator": ",",
        "chunksize": 20_000,
        "text_cols": [
            "VÍA_PRINCIPAL", "VÍA_SECUNDARIA", "MOVIMIENTO", "ACCESO",
            "MES", "INTERSECCIÓN", "TIPO_DE_DIA", "COMUNA",
            "NOMBRE_COMUNA", "CONSULTOR", "DÍA_AFORO",
        ],
        "skip_cols": ["FECHA", "FECHA_HORA", "HORA_A", "DIA/MES/AÑO"],
    },
    "ciclorrutas": {
        "filename" : "Ciclorrutas.csv",
        "encoding" : "utf-8",
        "separator": ",",
        "chunksize": None,
        "text_cols": ["nombre", "estado"],
        "skip_cols": [],
    },
    "pasajeros": {
        "filename" : "Pasajeros_movilizados.csv",
        "encoding" : "utf-8-sig",   # BOM detectado
        "separator": ",",
        "chunksize": None,
        "text_cols": ["MES", "SEM_AÑO", "LÍNEA_O", "AÑO-MES"],
        "skip_cols": ["AÑO", "NUM_MES"],
    },
    "poblacion": {
        "filename" : "proyecciones_de_poblacion_medellin_2019.csv",
        "encoding" : "utf-8",
        "separator": ",",
        "chunksize": None,
        "text_cols": ["tipo_division_geografica", "codigo", "grupo_edad"],
        "skip_cols": [],
    },
    "simm": {
        "filename" : "simmtrafficdata.csv",
        "encoding" : "utf-8",
        "separator": ",",
        "chunksize": None,
        "text_cols": [
            "CORREDOR", "DISPOSITIVO", "MES", "SENTIDO",
            "INDRA.MENSAJE", "INDRA.PARAMETRO", "INDRA.TIPO", "INDRA.VERSION",
        ],
        "skip_cols": ["FECHAHORA", "LOCATION", "INDRA.TIMESTAMP", "INDRA.CONSECUTIVO"],
    },
    "velocidad": {
        "filename" : "velocidad_e_intensidad_vehicular_en_medellin.csv",
        "encoding" : "utf-8",
        "separator": ",",
        "chunksize": 50_000,    # chunked: 76 MB
        "text_cols": [
            "carril", "dia", "mes", "corredor", "sentido",
            "operacion", "tipo_subsistema", "nombre_comuna",
        ],
        "skip_cols": [
            "fecha_trafico", "fecha", "hora", "longitud", "latitud",
            "identificador_f_v", "codigo_comuna", "comuna",
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Correcciones de caracteres U+FFFD provenientes de la fuente velocidad CSV
# Se aplican ANTES de la Regla 3 (strip de FFFD) para preservar el contexto.
# ──────────────────────────────────────────────────────────────────────────────
_FFFD = "\uFFFD"

_UNICODE_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bV" + _FFFD + r"+a"),     "Vía"),       # Vía al Mar / Túnel
    (re.compile(r"T"   + _FFFD + r"+nel"),   "Túnel"),     # Túnel de Occidente
    (re.compile(r"Crist" + _FFFD + r"+bal"), "Cristóbal"), # Corregimiento Cristóbal
    (re.compile(r"r"   + _FFFD + r"+o"),     "río"),       # Sistema vial del río
    (re.compile(_FFFD  + r"+XITO"),          "ÉXITO"),     # ÉXITO (supermercado)
    (re.compile(r"DO"  + _FFFD + r"+A"),     "DOÑA"),      # DOÑA MARIA
]


def _fix_unicode_corruption(text: str) -> str:
    """Regla 0: restaura acentos perdidos (U+FFFD) antes de limpiar el resto."""
    if _FFFD not in text:
        return text
    for pattern, replacement in _UNICODE_FIXES:
        text = pattern.sub(replacement, text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Expresión regular para caracteres fuera del espacio Latino + símbolos comunes
# ──────────────────────────────────────────────────────────────────────────────
# Se permiten:
#   \x09 \x0A \x0D       — tab, LF, CR
#   \x20-\x7E            — ASCII imprimible
#   \xA0-\u024F          — Latin-1 Supplement + Latin Extended-A/B
#                          (incluye á é í ó ú ñ ü ¿ ¡ y más)
#   \u2000-\u206F        — Puntuación general Unicode (guiones largos, etc.)
#   \u20A0-\u20CF        — Símbolos de moneda
# Se eliminan: cirílico (U+0400+), árabe (U+0600+), CJK (U+4E00+), etc.
_ALLOWED = re.compile(
    r"[^\x09\x0A\x0D\x20-\x7E\xA0-\u024F\u2000-\u206F\u20A0-\u20CF]"
)

# Caracteres de control: \x00–\x08, \x0B, \x0C, \x0E–\x1F, \x7F
_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Símbolo de reemplazo Unicode y variantes similares
_REPLACEMENT = re.compile(r"[\uFFFD\uFFFE\uFFFF]")

# Comillas redundantes al inicio y final (artefactos de Excel / pandas)
_EDGE_QUOTES = re.compile(r'^["\'\s]+|["\'\s]+$')

# Espacios múltiples
_MULTI_SPACE = re.compile(r" {2,}")


# ──────────────────────────────────────────────────────────────────────────────
# Núcleo de limpieza por celda
# ──────────────────────────────────────────────────────────────────────────────

def clean_cell(value: object) -> tuple[str, bool]:
    """
    Aplica las 7 reglas de limpieza a un valor de celda de texto.

    Parameters
    ----------
    value : cualquier valor de celda (se convierte a str si no es NaN).

    Returns
    -------
    (cleaned_str, changed) donde `changed` es True si el valor se modificó.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ("", False)

    original = str(value)
    text = original

    # Regla 0: restaurar caracteres acentuados corruptos (U+FFFD contextuales)
    text = _fix_unicode_corruption(text)

    # Regla 3: símbolo de reemplazo y variantes residuales
    text = _REPLACEMENT.sub("", text)

    # Regla 2: caracteres de control (preservando \t \n \r)
    text = _CONTROL.sub("", text)

    # Regla 4: caracteres fuera del espacio latino y símbolos comunes
    text = _ALLOWED.sub("", text)

    # Regla 7: comillas / escapes al inicio y final
    # Solo si el valor quedó rodeado de comillas sueltas tras la limpieza
    stripped_test = text.strip()
    if (
        len(stripped_test) >= 2
        and stripped_test[0] in ('"', "'")
        and stripped_test[-1] in ('"', "'")
        and stripped_test[0] == stripped_test[-1]
        # Solo eliminar si el interior no contiene la misma comilla
        and stripped_test[0] not in stripped_test[1:-1]
    ):
        text = stripped_test[1:-1]

    # Regla 5: colapsar espacios múltiples
    text = _MULTI_SPACE.sub(" ", text)

    # Regla 6: strip
    text = text.strip()

    changed = text != original
    return (text, changed)


def _auto_detect_text_cols(df: pd.DataFrame, skip_cols: list[str]) -> list[str]:
    """
    Detecta automáticamente columnas de texto en un DataFrame.

    Una columna es candidata si:
    - dtype == object
    - No está en skip_cols
    - Menos del 90% de sus valores (no nulos) son numéricos.
    """
    candidates = []
    for col in df.columns:
        if col in skip_cols:
            continue
        if df[col].dtype != object:
            continue
        sample = df[col].dropna().head(200).astype(str)
        if len(sample) == 0:
            continue
        numeric_frac = pd.to_numeric(sample, errors="coerce").notna().mean()
        if numeric_frac < 0.9:
            candidates.append(col)
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Limpieza de un DataFrame (un chunk)
# ──────────────────────────────────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    text_cols: list[str],
    filename: str,
    row_offset: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Aplica `clean_cell` a cada columna de texto del DataFrame.

    Parameters
    ----------
    df         : Chunk a limpiar.
    text_cols  : Columnas a procesar.
    filename   : Nombre del archivo fuente (para el log).
    row_offset : Número de fila base (para archivos procesados por chunks).

    Returns
    -------
    (df_clean, log_entries)
    """
    df_clean = df.copy()
    log_entries: list[dict] = []

    cols_present = [c for c in text_cols if c in df_clean.columns]

    for col in cols_present:
        series = df_clean[col].copy()
        for local_idx, (df_idx, val) in enumerate(series.items()):
            cleaned, changed = clean_cell(val)
            if changed:
                log_entries.append(
                    {
                        "archivo"        : filename,
                        "fila"           : row_offset + local_idx + 2,  # +2: 1-based + header
                        "columna"        : col,
                        "valor_original" : str(val),
                        "valor_limpio"   : cleaned,
                    }
                )
                df_clean.at[df_idx, col] = cleaned

    return df_clean, log_entries


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────────────────

class CsvCleaner:
    """
    Limpia y normaliza archivos CSV del proyecto Movilidad Medellín.

    Parameters
    ----------
    raw_dir   : Directorio de archivos fuente.
    clean_dir : Directorio de salida (archivos limpios).
    log_path  : Ruta del CSV de log de cambios.
    dry_run   : Si True, no escribe archivos; solo reporta estadísticas.
    """

    def __init__(
        self,
        raw_dir  : Path = RAW_DIR,
        clean_dir: Path = CLEAN_DIR,
        log_path : Path = LOG_PATH,
        dry_run  : bool = False,
    ) -> None:
        self.raw_dir   = raw_dir
        self.clean_dir = clean_dir
        self.log_path  = log_path
        self.dry_run   = dry_run
        self._log_rows: list[dict] = []

    # ── Métodos de soporte ────────────────────────────────────────────

    def _resolve_input(self, file_cfg: dict) -> Path:
        return self.raw_dir / file_cfg["filename"]

    def _resolve_output(self, file_cfg: dict) -> Path:
        return self.clean_dir / file_cfg["filename"]

    def _chunks(
        self,
        path       : Path,
        file_cfg   : dict,
        text_cols  : list[str],
    ) -> Iterator[tuple[pd.DataFrame, list[dict], int]]:
        """
        Itera sobre el archivo CSV en chunks, limpiando cada uno.

        Yields (cleaned_chunk, log_entries, rows_so_far).
        """
        chunksize = file_cfg.get("chunksize")
        enc       = file_cfg["encoding"]
        sep       = file_cfg["separator"]
        rows_read = 0

        read_kwargs: dict = dict(
            encoding      = enc,
            sep           = sep,
            low_memory    = False,
            keep_default_na=True,
            on_bad_lines  = "warn",
        )

        if chunksize:
            reader = pd.read_csv(path, chunksize=chunksize, **read_kwargs)
            for chunk in reader:
                # Resolver columnas de texto la primera vez (pueden variar entre chunks)
                effective_cols = text_cols if text_cols else _auto_detect_text_cols(
                    chunk, file_cfg.get("skip_cols", [])
                )
                cleaned, logs = clean_dataframe(
                    chunk,
                    effective_cols,
                    file_cfg["filename"],
                    row_offset=rows_read,
                )
                rows_read += len(chunk)
                yield cleaned, logs, rows_read
        else:
            df = pd.read_csv(path, **read_kwargs)
            effective_cols = text_cols if text_cols else _auto_detect_text_cols(
                df, file_cfg.get("skip_cols", [])
            )
            cleaned, logs = clean_dataframe(
                df,
                effective_cols,
                file_cfg["filename"],
                row_offset=0,
            )
            yield cleaned, logs, len(df)

    # ── Limpieza de un archivo ────────────────────────────────────────

    def clean_file(self, key: str) -> dict:
        """
        Limpia un archivo CSV identificado por su clave en FILES.

        Returns
        -------
        dict con métricas: filas_procesadas, celdas_modificadas, columnas_limpias.
        """
        if key not in FILES:
            raise ValueError(f"Clave '{key}' no reconocida. Opciones: {list(FILES)}")

        file_cfg   = FILES[key]
        input_path = self._resolve_input(file_cfg)
        out_path   = self._resolve_output(file_cfg)

        if not input_path.exists():
            logger.warning("Archivo no encontrado: %s — omitiendo.", input_path)
            return {"filas_procesadas": 0, "celdas_modificadas": 0, "columnas_limpias": []}

        text_cols = file_cfg.get("text_cols") or []
        logger.info("▶ Procesando: %s", file_cfg["filename"])
        logger.info("  Columnas de texto: %s", text_cols or "(auto-detectar)")

        total_rows     = 0
        total_changes  = 0
        file_log: list[dict] = []
        first_chunk    = True

        # Escribir en modo streaming para archivos grandes
        out_handle = None
        if not self.dry_run:
            out_handle = open(out_path, "w", encoding="utf-8", newline="")
            csv_writer = None  # Se inicializará con el primer chunk

        try:
            for cleaned_chunk, logs, rows_so_far in self._chunks(
                input_path, file_cfg, text_cols
            ):
                total_rows    = rows_so_far
                total_changes += len(logs)
                file_log.extend(logs)

                if not self.dry_run:
                    # Usar el writer nativo de pandas para consistencia
                    cleaned_chunk.to_csv(
                        out_handle,
                        index  = False,
                        header = first_chunk,
                        # Forzar comillas solo donde sea necesario
                        quoting= csv.QUOTE_MINIMAL,
                    )
                    first_chunk = False

                if rows_so_far % 100_000 == 0 or rows_so_far == total_rows:
                    logger.info("  … %d filas procesadas", rows_so_far)
        finally:
            if out_handle:
                out_handle.close()

        # Acumular log global
        self._log_rows.extend(file_log)

        cols_touched = sorted({e["columna"] for e in file_log})
        logger.info(
            "  ✓ %s → %d filas | %d celdas modificadas | cols: %s",
            file_cfg["filename"],
            total_rows,
            total_changes,
            cols_touched or "ninguna",
        )

        return {
            "filas_procesadas"  : total_rows,
            "celdas_modificadas": total_changes,
            "columnas_limpias"  : cols_touched,
        }

    # ── Limpieza de todos los archivos ────────────────────────────────

    def clean_all(self, keys: list[str] | None = None) -> dict[str, dict]:
        """
        Limpia todos los CSV configurados (o los indicados en `keys`).

        Returns
        -------
        dict { key → métricas }
        """
        targets = keys or list(FILES.keys())
        results: dict[str, dict] = {}

        logger.info("=" * 60)
        logger.info("CSV Cleaner — Movilidad Medellín")
        logger.info("Modo: %s", "DRY-RUN (sin escritura)" if self.dry_run else "ESCRITURA")
        logger.info("Destino: %s", self.clean_dir)
        logger.info("=" * 60)

        for key in targets:
            try:
                results[key] = self.clean_file(key)
            except Exception as exc:
                logger.error("Error en '%s': %s", key, exc)
                results[key] = {"error": str(exc)}

        self._write_log()
        self._print_summary(results)
        return results

    # ── Log de cambios ────────────────────────────────────────────────

    def _write_log(self) -> None:
        """Escribe el log acumulado de cambios en cleaning_log.csv."""
        if not self._log_rows:
            logger.info("Log: sin cambios registrados.")
            return

        if self.dry_run:
            logger.info("DRY-RUN: log no escrito (%d entradas).", len(self._log_rows))
            return

        log_df = pd.DataFrame(
            self._log_rows,
            columns=["archivo", "fila", "columna", "valor_original", "valor_limpio"],
        )
        log_df.to_csv(self.log_path, index=False, encoding="utf-8")
        logger.info("Log de cambios guardado: %s (%d entradas)", self.log_path, len(log_df))

    def _print_summary(self, results: dict[str, dict]) -> None:
        """Imprime tabla resumen al final del proceso."""
        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info("=" * 60)

        total_filas   = 0
        total_cambios = 0

        for key, metrics in results.items():
            if "error" in metrics:
                logger.error("  %-12s  ERROR: %s", key, metrics["error"])
                continue
            f = metrics["filas_procesadas"]
            c = metrics["celdas_modificadas"]
            total_filas   += f
            total_cambios += c
            logger.info(
                "  %-12s  %8d filas  |  %5d celdas modificadas",
                key, f, c,
            )

        logger.info("-" * 60)
        logger.info(
            "  %-12s  %8d filas  |  %5d celdas modificadas",
            "TOTAL", total_filas, total_cambios,
        )
        logger.info("=" * 60)

        if not self.dry_run:
            logger.info("Archivos limpios en: %s", self.clean_dir)
            logger.info("Log detallado en:    %s", self.log_path)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidad: inspeccionar un valor puntual (útil para debug)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_value(value: str) -> None:
    """Muestra el detalle de limpieza de un valor específico (para debugging)."""
    cleaned, changed = clean_cell(value)
    print(f"Original : {repr(value)}")
    print(f"Limpio   : {repr(cleaned)}")
    print(f"Cambió   : {changed}")
    if changed:
        orig_chars = [f"U+{ord(c):04X}({unicodedata.name(c, '?')})" for c in value]
        cln_chars  = [f"U+{ord(c):04X}({unicodedata.name(c, '?')})" for c in cleaned]
        removed    = set(orig_chars) - set(cln_chars)
        if removed:
            print(f"Eliminados: {removed}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpieza y normalización de texto en CSV de Movilidad Medellín."
    )
    parser.add_argument(
        "--file",
        choices=list(FILES.keys()),
        default=None,
        help="Procesar solo este archivo (por defecto: todos).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar estadísticas sin escribir archivos ni log.",
    )
    parser.add_argument(
        "--inspect",
        metavar="VALOR",
        default=None,
        help="Inspeccionar cómo se limpiaría un valor específico y salir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Modo inspect: solo analiza un valor y sale
    if args.inspect:
        inspect_value(args.inspect)
        return

    cleaner = CsvCleaner(dry_run=args.dry_run)

    if args.file:
        cleaner.clean_file(args.file)
        cleaner._write_log()
    else:
        cleaner.clean_all()


if __name__ == "__main__":
    sys.exit(main() or 0)
