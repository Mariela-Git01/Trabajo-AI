# =============================================================================
# datos.py - Datos REALES de consumo de agua (4 años)
# Vivienda de Alquiler - 3 pisos - Cajamarca, Peru
# Habitantes: 5 personas (fijo todos los meses y anos)
# Fuente: Reporte de consumo en L/m3 durante 4 anos
# =============================================================================

import numpy as np

np.random.seed(7)

# -----------------------------------------------------------------------------
# CONSTANTES
# -----------------------------------------------------------------------------

PERSONAS = 5          # Habitantes fijos todos los meses y anos
ANIOS    = [2022, 2023, 2024, 2025]

meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

meses_abr = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
             "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# -----------------------------------------------------------------------------
# DATOS POR ANO - directamente de la tabla de la imagen
# Cada lista tiene 12 valores (Enero a Diciembre)
# -----------------------------------------------------------------------------

datos_por_anio = {
    2025: {
        "m3"   : [12, 15, 19, 17, 17, 16, 15, 19, 13, 13, 14, 17],
        "L"    : [12000, 15000, 19000, 17000, 17000, 16000,
                  15000, 19000, 13000, 13000, 14000, 17000],
        "L_dia": [400.00, 500.00, 633.33, 566.67, 566.67, 533.33,
                  500.00, 633.33, 433.33, 433.33, 466.67, 566.67],
    },
    2024: {
        "m3"   : [19, 15, 21, 16, 24, 24, 16, 12, 14, 12, 12, 12],
        "L"    : [19000, 15000, 21000, 16000, 24000, 24000,
                  16000, 12000, 14000, 12000, 12000, 12000],
        "L_dia": [633.33, 500.00, 700.00, 533.33, 800.00, 800.00,
                  533.33, 400.00, 466.67, 400.00, 400.00, 400.00],
    },
    2023: {
        "m3"   : [12, 13, 11, 16, 20, 17, 20, 13, 12, 17, 12, 13],
        "L"    : [12000, 13000, 11000, 16000, 20000, 17000,
                  20000, 13000, 12000, 17000, 12000, 13000],
        "L_dia": [400.00, 433.33, 366.67, 533.33, 666.67, 566.67,
                  666.67, 433.33, 400.00, 566.67, 400.00, 433.33],
    },
    2022: {
        "m3"   : [17, 19, 22, 19, 19, 20, 13, 14, 14, 8, 13, 14],
        "L"    : [17000, 19000, 22000, 19000, 19000, 20000,
                  13000, 14000, 14000, 8000, 13000, 14000],
        "L_dia": [566.67, 633.33, 733.33, 633.33, 633.33, 666.67,
                  433.33, 466.67, 466.67, 266.67, 433.33, 466.67],
    },
}

# -----------------------------------------------------------------------------
# ACCESOS RAPIDOS - ano mas reciente (2025) para graficos del ano actual
# -----------------------------------------------------------------------------

consumo_diario        = datos_por_anio[2025]["L"]
metros_cubicos_mes    = datos_por_anio[2025]["m3"]
litros_por_dia_mes    = datos_por_anio[2025]["L_dia"]
ocupantes_por_dia     = [PERSONAS] * 12
dias                  = list(range(1, 13))

litros_por_persona_mes = [round(l / PERSONAS, 2) for l in consumo_diario]

# -----------------------------------------------------------------------------
# SERIE HISTORICA COMPLETA - 48 puntos en orden cronologico 2022 a 2025
# Esta es la clave para que los modelos predigan bien:
#   - Cada punto representa un mes real con su consumo en litros
#   - El indice lineal (1 a 48) permite al modelo ver la tendencia a lo largo del tiempo
#   - El indice de mes (1 a 12) permite detectar patrones estacionales (cada ano)
# -----------------------------------------------------------------------------

consumo_historico = []  # Litros por mes, orden cronologico
anio_historico    = []  # Ano de cada punto
mes_historico     = []  # Mes (1-12) de cada punto

for anio in [2022, 2023, 2024, 2025]:
    for i, litros in enumerate(datos_por_anio[anio]["L"]):
        consumo_historico.append(litros)
        anio_historico.append(anio)
        mes_historico.append(i + 1)

# Indice lineal 1-48
indice_historico = list(range(1, len(consumo_historico) + 1))

# -----------------------------------------------------------------------------
# CONSUMO POR FRANJA HORARIA - estimado (solo 2025, para analisis interno)
# Franjas: [Madrugada 00-06h, Manana 06-12h, Tarde 12-18h, Noche 18-24h]
# Con 5 personas fijas la distribucion es estable durante todo el ano
# -----------------------------------------------------------------------------

def generar_franja_mensual(total_litros, mes_index):
    props = np.array([0.05, 0.40, 0.30, 0.25])
    ruido = np.random.uniform(-0.01, 0.01, 4)
    props = np.clip(props + ruido, 0.02, 0.60)
    props /= props.sum()
    return np.round(props * total_litros, 1).tolist()

consumo_horario = [
    generar_franja_mensual(consumo_diario[i], i)
    for i in range(12)
]

# -----------------------------------------------------------------------------
# VERIFICACION - ejecutar datos.py directamente para revisar
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  VERIFICACION - 4 anos de consumo real")
    print("  Vivienda de Alquiler, 3 Pisos - Cajamarca, Peru")
    print("  Habitantes: 5 personas (fijo)")
    print("=" * 65)
    for anio in [2022, 2023, 2024, 2025]:
        total = sum(datos_por_anio[anio]["L"])
        prom  = total / 12
        maxi  = max(datos_por_anio[anio]["L"])
        mini  = min(datos_por_anio[anio]["L"])
        mes_m = meses[datos_por_anio[anio]["L"].index(maxi)]
        mes_n = meses[datos_por_anio[anio]["L"].index(mini)]
        print(f"  {anio}  ->  Total: {total:,} L  |  "
              f"Prom: {prom:,.0f} L/mes  |  "
              f"Max: {mes_m} {maxi:,} L  |  Min: {mes_n} {mini:,} L")
    print("=" * 65)
    print(f"  Total historico (48 meses): {sum(consumo_historico):,} L")
    print("=" * 65)
