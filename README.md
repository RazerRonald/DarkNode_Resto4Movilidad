# Reto de Movilidad — Medellín

Solución end-to-end de ciencia de datos para analizar corredores críticos, comunas con mayor presión vehicular y horas de mayor congestión en Medellín.

## Estructura del proyecto

```
reto4/
├── data/
│   └── raw/                  ← Copia aquí los 6 CSV
├── notebooks/
│   └── 01_eda.ipynb          ← EDA narrativo
├── src/
│   ├── pipeline.py           ← Ingesta, ETL, EDA, ICV
│   └── dashboard.py          ← Dashboard Streamlit
├── config.yaml               ← Rutas y parámetros centralizados
├── requirements.txt
├── README.md
└── main.py                   ← Punto de entrada
```

## Requisitos previos

- Windows 11
- Python 3.10 o superior
- PowerShell (terminal integrado de VS Code)

## Instalación

### 1. Crear entorno virtual

```powershell
cd C:\Users\WinterOS\Documents\Dashboard\reto4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea la ejecución de scripts, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2. Instalar dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Copiar los CSV a `data/raw/`

Coloca los siguientes archivos **exactamente con estos nombres** en `data/raw/`:

| Archivo | Descripción |
|---------|-------------|
| `Aforos_Vehiculares.csv` | Conteos vehiculares por intersección (2018) |
| `Ciclorrutas.csv` | Red ciclista existente y proyectada |
| `Pasajeros_movilizados.csv` | Pasajeros Metro Medellín 2014-2021 |
| `proyecciones_de_poblacion_medellin_2019.csv` | Población por comuna 2019 |
| `simmtrafficdata.csv` | Sensores CCTV en corredores (2021) |
| `velocidad_e_intensidad_vehicular_en_medellin.csv` | Velocidad e intensidad horaria (2020+) |

```powershell
# Ejemplo: copiar desde el escritorio
Copy-Item "$env:USERPROFILE\Desktop\*.csv" "data\raw\"
```

## Ejecución

### Pipeline completo (ingesta → ETL → EDA → ICV)

```powershell
python main.py
```

### Pipeline + Dashboard interactivo

```powershell
python main.py --dashboard
```

El dashboard se abre automáticamente en `http://localhost:8501`.

### Solo el dashboard (si el pipeline ya se ejecutó antes)

```powershell
streamlit run src\dashboard.py
```

### Notebook de EDA

```powershell
jupyter notebook notebooks\01_eda.ipynb
```

O desde VS Code: abre el archivo `.ipynb` y selecciona el kernel `.venv`.

## Descripción técnica

### `src/pipeline.py`

Contiene toda la lógica de negocio organizada en secciones:

- **Ingesta**: `load_all_data()` — detección automática de encoding con chardet.
- **ETL por dataset**: `etl_aforos()`, `etl_simm()`, `etl_velocidad()`, `etl_ciclorrutas()`, `etl_pasajeros()`, `etl_poblacion()`.
- **EDA**: funciones que retornan DataFrames o Figures matplotlib.
- **ICV**: `build_criticality_index()` — Índice de Criticidad Vial 0-100.

### Índice de Criticidad Vial (ICV)

Fórmula documentada en `build_criticality_index()`:

```
ICV = 0.40 × (1 - vel_normalizada) + 0.40 × int_normalizada + 0.20 × pop_normalizada
```

- Filtro: días entre semana + horas pico (6-9h, 17-20h).
- Normalización min-max por componente.
- Escala final: 0 (sin criticidad) a 100 (máxima criticidad).

### Estrategia de integración entre datasets

| Dataset | Granularidad | Rol en el análisis |
|---------|-------------|-------------------|
| `velocidad` | Horario, por corredor (2020+) | **Fuente primaria del ICV** |
| `simm` | Por sensor CCTV (2021) | Validación + mapa de puntos |
| `aforos` | 15 min, por intersección (2018) | Coordenadas WGS84 para mapas |
| `poblacion` | Anual por comuna (2019) | Ponderación demográfica del ICV |
| `pasajeros` | Semestral/mensual (2014-2021) | Contexto transporte público |
| `ciclorrutas` | Estático | Contexto infraestructura |

> No se fuerzan joins entre datasets con desfase temporal significativo.
> La armonización se realiza por nombre de corredor y nombre de comuna.

## Notas técnicas

- **Encoding**: los CSV usan encodings distintos (UTF-8, UTF-8-sig, latin-1). La detección es automática.
- **Formato numérico**: `Pasajeros_movilizados.csv` usa punto como separador de miles (formato español). Se convierte automáticamente.
- **Coordenadas en `velocidad_e_intensidad`**: están en MAGNA-SIRGAS CTM12 (sistema proyectado), no en WGS84. No se usan para mapas; los mapas usan coordenadas de `simmtrafficdata` y `Aforos_Vehiculares`.
