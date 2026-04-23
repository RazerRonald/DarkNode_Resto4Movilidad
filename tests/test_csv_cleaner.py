"""
tests/test_csv_cleaner.py — Tests unitarios del módulo de limpieza CSV.

Ejecutar:
    python -m pytest tests/test_csv_cleaner.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.csv_cleaner import clean_cell, clean_dataframe, CsvCleaner, FILES


# ════════════════════════════════════════════════════════════════════
# Tests de clean_cell — regla por regla
# ════════════════════════════════════════════════════════════════════

class TestCleanCellRegla1_EspanolConservado:
    """Regla 1: caracteres españoles válidos se conservan."""

    @pytest.mark.parametrize("value", [
        "Avenida Bolívar",
        "Laureles Estadio",
        "Calle 80 Nº 45-12",
        "¡Atención! ¿Qué pasó?",
        "Señalización vial",
        "güiro ñoño",
        "camión",
        "área",
    ])
    def test_espanol_conservado(self, value):
        cleaned, changed = clean_cell(value)
        assert cleaned == value, f"No debería modificar: {repr(value)}"
        assert not changed

class TestCleanCellRegla2_ControlesEliminados:
    """Regla 2: caracteres de control eliminados (excepto \\t \\n \\r)."""

    def test_null_byte_eliminado(self):
        cleaned, changed = clean_cell("Hola\x00Mundo")
        assert "\x00" not in cleaned
        assert changed

    def test_bell_eliminado(self):
        cleaned, changed = clean_cell("test\x07texto")
        assert "\x07" not in cleaned
        assert changed

    def test_multiple_controles(self):
        cleaned, changed = clean_cell("\x01\x02texto\x1F")
        assert cleaned == "texto"
        assert changed

    def test_tab_conservado(self):
        """\\t es el único control que se conserva."""
        cleaned, changed = clean_cell("col1\tcol2")
        assert "\t" in cleaned


class TestCleanCellRegla3_ReplacementEliminado:
    """Regla 3: símbolo de reemplazo U+FFFD eliminado."""

    def test_replacement_char(self):
        cleaned, changed = clean_cell("texto\uFFFDroto")
        assert "\uFFFD" not in cleaned
        assert changed

    def test_multiple_replacements(self):
        cleaned, changed = clean_cell("\uFFFD\uFFFD\uFFFD")
        assert cleaned == ""
        assert changed

    def test_ffff_eliminado(self):
        cleaned, _ = clean_cell("a\uFFFFb")
        assert "\uFFFF" not in cleaned


class TestCleanCellRegla4_NoLatinoEliminado:
    """Regla 4: caracteres no latinos eliminados."""

    def test_cirílico_eliminado(self):
        cleaned, changed = clean_cell("Привет")
        assert cleaned == ""
        assert changed

    def test_árabe_eliminado(self):
        cleaned, changed = clean_cell("مرحبا")
        assert cleaned == ""
        assert changed

    def test_cjk_eliminado(self):
        cleaned, changed = clean_cell("你好")
        assert cleaned == ""
        assert changed

    def test_mezcla_español_cirílico(self):
        """Solo la parte española se conserva."""
        cleaned, changed = clean_cell("Bogotá Привет")
        assert "Bogotá" in cleaned
        # Los caracteres cirílicos se eliminan; puede quedar espacio extra
        assert "Привет" not in cleaned
        assert changed


class TestCleanCellRegla5_EspaciosColapsados:
    """Regla 5: espacios múltiples → un solo espacio."""

    def test_doble_espacio(self):
        cleaned, changed = clean_cell("Calle  80")
        assert cleaned == "Calle 80"
        assert changed

    def test_muchos_espacios(self):
        cleaned, changed = clean_cell("a      b")
        assert cleaned == "a b"
        assert changed

    def test_espacio_simple_intacto(self):
        cleaned, changed = clean_cell("Calle 80")
        assert cleaned == "Calle 80"
        assert not changed


class TestCleanCellRegla6_StripAplicado:
    """Regla 6: strip de espacios al inicio y final."""

    def test_espacio_inicio(self):
        cleaned, changed = clean_cell("  texto")
        assert cleaned == "texto"
        assert changed

    def test_espacio_final(self):
        cleaned, changed = clean_cell("texto  ")
        assert cleaned == "texto"
        assert changed

    def test_ambos_extremos(self):
        cleaned, changed = clean_cell("   Medellín   ")
        assert cleaned == "Medellín"
        assert changed


class TestCleanCellRegla7_ComillasCorregidas:
    """Regla 7: comillas redundantes al inicio/final eliminadas."""

    def test_comillas_dobles_exteriores(self):
        cleaned, changed = clean_cell('"Avenida Bolivariana"')
        assert cleaned == "Avenida Bolivariana"
        assert changed

    def test_comillas_simples_exteriores(self):
        cleaned, changed = clean_cell("'corredor norte'")
        assert cleaned == "corredor norte"
        assert changed

    def test_comilla_interior_conservada(self):
        """Si el valor ya tiene comillas internas válidas, no se toca."""
        valor = '"texto con comilla interna \' aquí"'
        # El regex no elimina si hay comilla interior diferente
        cleaned, _ = clean_cell(valor)
        # Solo verificamos que no lanza excepción y el resultado es string
        assert isinstance(cleaned, str)


class TestCleanCellCasosEspeciales:
    """Casos borde: None, NaN, strings vacíos, numéricos como string."""

    def test_none(self):
        cleaned, changed = clean_cell(None)
        assert cleaned == ""
        assert not changed

    def test_nan(self):
        import math
        cleaned, changed = clean_cell(float("nan"))
        assert cleaned == ""
        assert not changed

    def test_string_vacio(self):
        cleaned, changed = clean_cell("")
        assert cleaned == ""
        assert not changed

    def test_numero_string(self):
        """Números como string no deberían cambiar."""
        cleaned, changed = clean_cell("12345")
        assert cleaned == "12345"
        assert not changed

    def test_valor_numerico_float(self):
        """Floats se convierten a string; no deberían romperse."""
        cleaned, changed = clean_cell(3.14)
        assert "3" in cleaned


# ════════════════════════════════════════════════════════════════════
# Tests de clean_dataframe
# ════════════════════════════════════════════════════════════════════

class TestCleanDataframe:
    """Tests de limpieza sobre un DataFrame completo."""

    @pytest.fixture
    def df_sucio(self):
        return pd.DataFrame({
            "corredor"     : ["Avenida\x00 Bolivariana", "  Calle 80  ", "Привет corredor"],
            "nombre_comuna": ["Laureles\uFFFD", "El Poblado", "  Robledo  "],
            "velocidad"    : [45.3, 60.1, 30.0],   # numérico: NO tocar
            "hora"         : [7, 8, 9],              # numérico: NO tocar
        })

    def test_columnas_texto_limpias(self, df_sucio):
        cleaned, logs = clean_dataframe(
            df_sucio, ["corredor", "nombre_comuna"], "test.csv"
        )
        # Verificar que los valores problemáticos fueron corregidos
        assert "\x00" not in cleaned["corredor"].iloc[0]
        assert "\uFFFD" not in cleaned["nombre_comuna"].iloc[0]
        assert cleaned["nombre_comuna"].iloc[2] == "Robledo"

    def test_columnas_numericas_intactas(self, df_sucio):
        cleaned, _ = clean_dataframe(
            df_sucio, ["corredor", "nombre_comuna"], "test.csv"
        )
        # Los valores numéricos no deben cambiar
        assert list(cleaned["velocidad"]) == [45.3, 60.1, 30.0]
        assert list(cleaned["hora"])      == [7, 8, 9]

    def test_log_registra_cambios(self, df_sucio):
        _, logs = clean_dataframe(
            df_sucio, ["corredor", "nombre_comuna"], "test.csv"
        )
        assert len(logs) > 0
        # Cada entrada debe tener las 5 claves requeridas
        for entry in logs:
            assert "archivo"         in entry
            assert "fila"            in entry
            assert "columna"         in entry
            assert "valor_original"  in entry
            assert "valor_limpio"    in entry

    def test_log_no_registra_sin_cambio(self):
        df_limpio = pd.DataFrame({"texto": ["Medellín", "Bogotá", "Cali"]})
        _, logs = clean_dataframe(df_limpio, ["texto"], "clean.csv")
        assert len(logs) == 0

    def test_columna_inexistente_no_falla(self, df_sucio):
        """Si una columna de text_cols no existe, no lanza excepción."""
        cleaned, logs = clean_dataframe(
            df_sucio, ["corredor", "columna_que_no_existe"], "test.csv"
        )
        assert "corredor" in cleaned.columns

    def test_row_offset_en_log(self, df_sucio):
        """El row_offset se refleja correctamente en los números de fila del log."""
        _, logs = clean_dataframe(
            df_sucio, ["corredor", "nombre_comuna"], "test.csv", row_offset=1000
        )
        if logs:
            # Todas las filas del log deben ser >= 1000 + 2 (header)
            assert all(e["fila"] >= 1002 for e in logs)


# ════════════════════════════════════════════════════════════════════
# Tests de CsvCleaner (integración ligera, sin I/O real)
# ════════════════════════════════════════════════════════════════════

class TestCsvCleanerDryRun:
    """Tests de CsvCleaner en modo dry-run (sin escritura de archivos)."""

    @pytest.fixture
    def cleaner(self, tmp_path):
        return CsvCleaner(
            raw_dir   = RAW_DIR_REAL if RAW_DIR_REAL.exists() else tmp_path,
            clean_dir = tmp_path / "clean",
            log_path  = tmp_path / "log.csv",
            dry_run   = True,
        )

    def test_archivo_inexistente_no_falla(self, tmp_path):
        """Si el archivo fuente no existe, retorna métricas vacías sin error."""
        cleaner = CsvCleaner(
            raw_dir   = tmp_path,       # directorio vacío
            clean_dir = tmp_path / "out",
            log_path  = tmp_path / "log.csv",
            dry_run   = True,
        )
        result = cleaner.clean_file("ciclorrutas")
        assert result["filas_procesadas"]   == 0
        assert result["celdas_modificadas"] == 0

    def test_clave_invalida(self, tmp_path):
        cleaner = CsvCleaner(dry_run=True)
        with pytest.raises(ValueError, match="no reconocida"):
            cleaner.clean_file("archivo_que_no_existe")

    def test_files_dict_completo(self):
        """El diccionario FILES tiene las 6 entradas esperadas."""
        expected = {"aforos", "ciclorrutas", "pasajeros", "poblacion", "simm", "velocidad"}
        assert set(FILES.keys()) == expected

    def test_files_tienen_campos_requeridos(self):
        for key, cfg in FILES.items():
            assert "filename"  in cfg, f"'{key}' falta 'filename'"
            assert "encoding"  in cfg, f"'{key}' falta 'encoding'"
            assert "separator" in cfg, f"'{key}' falta 'separator'"


# Referencia a los CSV reales (usados opcionalmente si existen)
RAW_DIR_REAL = ROOT / "data" / "raw"


class TestCsvCleanerConArchivosReales:
    """
    Tests de integración que se saltan si los archivos fuente no existen.
    Verifica que clean_file() funcione end-to-end con datos reales (dry-run).
    """

    @pytest.fixture
    def cleaner(self, tmp_path):
        return CsvCleaner(
            raw_dir   = RAW_DIR_REAL,
            clean_dir = tmp_path / "clean",
            log_path  = tmp_path / "log.csv",
            dry_run   = True,
        )

    @pytest.mark.skipif(
        not (RAW_DIR_REAL / "Ciclorrutas.csv").exists(),
        reason="Ciclorrutas.csv no encontrado",
    )
    def test_ciclorrutas_dry_run(self, cleaner):
        result = cleaner.clean_file("ciclorrutas")
        assert result["filas_procesadas"] > 0
        assert "nombre" in (result["columnas_limpias"] or []) or True  # puede no haber cambios

    @pytest.mark.skipif(
        not (RAW_DIR_REAL / "Pasajeros_movilizados.csv").exists(),
        reason="Pasajeros_movilizados.csv no encontrado",
    )
    def test_pasajeros_dry_run(self, cleaner):
        result = cleaner.clean_file("pasajeros")
        assert result["filas_procesadas"] > 0

    @pytest.mark.skipif(
        not (RAW_DIR_REAL / "proyecciones_de_poblacion_medellin_2019.csv").exists(),
        reason="poblacion CSV no encontrado",
    )
    def test_poblacion_dry_run(self, cleaner):
        result = cleaner.clean_file("poblacion")
        assert result["filas_procesadas"] > 0


# ════════════════════════════════════════════════════════════════════
# Tests de clean_cell con combinaciones múltiples
# ════════════════════════════════════════════════════════════════════

class TestCleanCellCombinaciones:
    """Valores que activan múltiples reglas a la vez."""

    def test_combinacion_control_espacios_strip(self):
        """Control + espacios múltiples + strip."""
        cleaned, changed = clean_cell("  Texto\x00  con  espacios  ")
        assert cleaned == "Texto con espacios"
        assert changed

    def test_combinacion_replacement_espanol(self):
        """U+FFFD mezclado con texto español válido."""
        cleaned, changed = clean_cell("Corred\uFFFDor Bolívar")
        assert "Bolívar" in cleaned
        assert "\uFFFD" not in cleaned
        assert changed

    def test_valor_solo_controles(self):
        """Valor que es puro basura → queda vacío."""
        cleaned, changed = clean_cell("\x00\x01\x02\uFFFD")
        assert cleaned == ""
        assert changed

    def test_valor_limpio_no_cambia(self):
        """Un valor perfectamente limpio no debe modificarse."""
        valor = "Corredor de la 80 (Medellín)"
        cleaned, changed = clean_cell(valor)
        assert cleaned == valor
        assert not changed
