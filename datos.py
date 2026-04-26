# =============================================================================
# datos.py — Datos REALES de consumo de agua
# Vivienda de Alquiler — 3 pisos — Cajamarca, Perú
# Consumo anual 2025: 245 m³ = 245,000 litros
# Fuente: Tabla de registro mensual del sistema
# =============================================================================

import numpy as np

np.random.seed(7)

# -----------------------------------------------------------------------------
# MESES DEL AÑO
# -----------------------------------------------------------------------------

meses = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"
]

# Índice numérico de los meses (1–12)
dias = list(range(1, 13))

# -----------------------------------------------------------------------------
# DATOS MENSUALES — Extraídos directamente de la tabla 2025
#
# Columnas originales:
#   N°PERSONAS | L/m3 | L (litros totales) | L/DÍA | L*PERSONA
#
# Enero     : 3 pers,  1 m³,  1000 L,   33.33 L/día,  11.11 L/pers
# Febrero   : 7 pers,  5 m³,  5000 L,  166.67 L/día,  23.81 L/pers
# Marzo     : 8 pers,  4 m³,  4000 L,  133.33 L/día,  16.67 L/pers
# Abril     : 10 pers, 9 m³,  9000 L,  300.00 L/día,  30.00 L/pers
# Mayo      : 12 pers,38 m³, 38000 L, 1266.67 L/día, 105.56 L/pers
# Junio     : 13 pers,49 m³, 49000 L, 1633.33 L/día, 125.64 L/pers
# Julio     : 10 pers,18 m³, 18000 L,  600.00 L/día,  60.00 L/pers
# Agosto    : 10 pers, 8 m³,  8000 L,  266.67 L/día,  26.67 L/pers
# Setiembre : 12 pers,17 m³, 17000 L,  566.67 L/día,  47.22 L/pers
# Octubre   : 11 pers,25 m³, 25000 L,  833.33 L/día,  75.76 L/pers
# Noviembre : 10 pers,38 m³, 38000 L, 1266.67 L/día, 126.67 L/pers
# Diciembre : 10 pers,33 m³, 33000 L, 1100.00 L/día, 110.00 L/pers
# -----------------------------------------------------------------------------

# Consumo mensual total en LITROS (una entrada por mes)
consumo_diario = [
    1000,   # Enero
    5000,   # Febrero
    4000,   # Marzo
    9000,   # Abril
    38000,  # Mayo
    49000,  # Junio     ← MÁXIMO del año
    18000,  # Julio
    8000,   # Agosto
    17000,  # Setiembre
    25000,  # Octubre
    38000,  # Noviembre
    33000,  # Diciembre
]

# Número de ocupantes (personas) por mes
ocupantes_por_dia = [
    3,   # Enero
    7,   # Febrero
    8,   # Marzo
    10,  # Abril
    12,  # Mayo
    13,  # Junio
    10,  # Julio
    10,  # Agosto
    12,  # Setiembre
    11,  # Octubre
    10,  # Noviembre
    10,  # Diciembre
]

# Metros cúbicos por mes (L/m3 de la tabla original)
metros_cubicos_mes = [1, 5, 4, 9, 38, 49, 18, 8, 17, 25, 38, 33]

# Litros por día por mes (L/DÍA de la tabla)
litros_por_dia_mes = [
    33.33, 166.67, 133.33, 300.00, 1266.67, 1633.33,
    600.00, 266.67, 566.67, 833.33, 1266.67, 1100.00
]

# Litros por persona por mes (L*PERSONA de la tabla)
litros_por_persona_mes = [
    11.11, 23.81, 16.67, 30.00, 105.56, 125.64,
    60.00, 26.67, 47.22, 75.76, 126.67, 110.00
]

# Verificación del total anual
_total = sum(consumo_diario)
assert _total == 245000, f"Total anual incorrecto: {_total} L (esperado 245,000 L)"

# -----------------------------------------------------------------------------
# CONSUMO POR FRANJA HORARIA — Estimado mensual
# Franjas: [Madrugada 00–06h, Mañana 06–12h, Tarde 12–18h, Noche 18–24h]
#
# Distribución estimada para una vivienda de alquiler con múltiples pisos:
#   Madrugada (00–06h):  5% — mínimo uso; pico = posible fuga
#   Mañana    (06–12h): 40% — ducha, desayuno, cocina
#   Tarde     (12–18h): 30% — almuerzo, limpieza
#   Noche     (18–24h): 25% — cena, ducha nocturna
#
# Meses con pico alto (Mayo, Junio, Nov, Dic) → mañana sube al 45%
# Meses de bajo consumo (Enero, Feb, Mar) → patrón más conservador
# -----------------------------------------------------------------------------

def generar_franja_mensual(total_litros, mes_index):
    """
    Estima el consumo por franja horaria para un mes dado.

    Args:
        total_litros (float): Consumo total del mes en litros.
        mes_index (int): Índice del mes (0 = Enero, 11 = Diciembre).

    Returns:
        list: 4 valores en litros [madrugada, mañana, tarde, noche].
    """
    # Proporciones base
    props = np.array([0.05, 0.40, 0.30, 0.25])

    # Meses de alta ocupación (Mayo, Junio, Octubre, Noviembre, Diciembre)
    if mes_index in [4, 5, 9, 10, 11]:
        props = np.array([0.05, 0.45, 0.30, 0.20])

    # Meses de baja ocupación (Enero, Febrero, Marzo)
    if mes_index in [0, 1, 2]:
        props = np.array([0.04, 0.38, 0.32, 0.26])

    # Ruido pequeño para realismo
    ruido = np.random.uniform(-0.01, 0.01, 4)
    props = np.clip(props + ruido, 0.02, 0.60)
    props /= props.sum()

    return np.round(props * total_litros, 1).tolist()


consumo_horario = [
    generar_franja_mensual(consumo_diario[i], i)
    for i in range(len(consumo_diario))
]

# -----------------------------------------------------------------------------
# BLOQUE DE VERIFICACIÓN (ejecutar datos.py directamente para revisar)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    total = sum(consumo_diario)
    print("=" * 60)
    print("  VERIFICACIÓN DE DATOS — Vivienda de Alquiler, 3 Pisos")
    print("  Cajamarca, Perú — Año 2025")
    print("=" * 60)
    print(f"  Total anual        : {total:,} litros = {total/1000:.0f} m³")
    print(f"  Total m³ (tabla)   : {sum(metros_cubicos_mes)} m³")
    print(f"  Promedio mensual   : {total/12:,.0f} L/mes")
    print(f"  Mes máximo         : Junio — {consumo_diario[5]:,} L ({metros_cubicos_mes[5]} m³)")
    print(f"  Mes mínimo         : Enero — {consumo_diario[0]:,} L ({metros_cubicos_mes[0]} m³)")
    print(f"  Max L/día (Junio)  : {litros_por_dia_mes[5]:,.2f} L/día")
    print(f"  Max L/pers (Nov)   : {litros_por_persona_mes[10]:.2f} L/persona")
    print("=" * 60)
    print(f"  {'MES':<12} {'PERS':>4}  {'m³':>4}  {'LITROS':>8}  {'L/DÍA':>9}  {'L/PERS':>8}")
    print("  " + "-" * 54)
    for i, mes in enumerate(meses):
        print(f"  {mes:<12} {ocupantes_por_dia[i]:>4}  {metros_cubicos_mes[i]:>4}  "
              f"{consumo_diario[i]:>8,}  {litros_por_dia_mes[i]:>9.2f}  "
              f"{litros_por_persona_mes[i]:>8.2f}")
    print("  " + "-" * 54)
    print(f"  {'TOTAL':<12} {'245':>4}  {'245':>4}  {total:>8,}")
    print("=" * 60)
