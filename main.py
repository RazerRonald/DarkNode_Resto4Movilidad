"""
main.py — Orquestador: ejecuta pipeline y lanza dashboard.

Uso:
  python main.py                  # Solo pipeline
  python main.py --dashboard      # Pipeline + Streamlit
"""

import sys
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run as run_pipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Movilidad Medellín")
    parser.add_argument("--dashboard", action="store_true", help="Lanzar dashboard tras el pipeline")
    args = parser.parse_args()

    config = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    # Ajustar rutas relativas al directorio del proyecto
    config["paths"]["raw"] = str(ROOT / config["paths"]["raw"])
    config["paths"]["processed"] = str(ROOT / config["paths"]["processed"])

    print("Ejecutando pipeline...")
    results = run_pipeline(config)

    print(f"\nMaster:             {len(results['master'])} registros")
    print(f"Corredores críticos:{len(results['corredores'])}")
    print(f"Hotspots:           {len(results['hotspots'])}")
    print("\nPipeline completado.")
    print("Ejecuta: streamlit run src/dashboard.py")

    if args.dashboard:
        dashboard_path = ROOT / "src" / "dashboard.py"
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
