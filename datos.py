# =============================================================================
# datos.py — Datos REALES de consumo de agua
# Vivienda familiar de 1 piso — Cajamarca, Perú
# Consumo mensual real: 17 m³ = 17,000 litros en 30 días
# Promedio diario real: ~567 litros/día
# Habitantes: entre 3 y 4 personas por semana
# Proveedor: SEDACAJ — Cajamarca
# =============================================================================

import numpy as np

np.random.seed(7)

# -----------------------------------------------------------------------------
# CONSUMO DIARIO (litros) — 30 días
# Total real del mes: 17,000 litros = 17 m³
#
# Criterio de distribución realista para Cajamarca:
#   - Lunes a viernes: consumo moderado, parte de la familia fuera de casa
#   - Sábado: pico por lavado de ropa y limpieza del hogar
#   - Domingo: familia completa en casa
#   - Cajamarca tiene clima frío → menor uso de ducha en madrugada
#   - El agua de SEDACAJ llega por horas en varios sectores,
#     por lo que hay días con llenado de tanque (picos puntuales)
#
# Semana 1 (días 1–7)  : 4 personas
# Semana 2 (días 8–14) : 4 personas
# Semana 3 (días 15–21): 3 personas (un integrante viaja)
# Semana 4 (días 22–30): 3 personas
# -----------------------------------------------------------------------------

consumo_diario = [
    # --- Semana 1 — 4 personas ---
    518,  # Día 1  Lunes      rutina normal de semana
    495,  # Día 2  Martes     consumo bajo, todos fuera
    542,  # Día 3  Miércoles  consumo medio
    510,  # Día 4  Jueves     rutina normal
    558,  # Día 5  Viernes    ligero aumento, noche en casa
    728,  # Día 6  Sábado     lavado de ropa + limpieza general
    682,  # Día 7  Domingo    familia completa en casa

    # --- Semana 2 — 4 personas ---
    503,  # Día 8  Lunes      rutina normal
    485,  # Día 9  Martes     consumo bajo
    531,  # Día 10 Miércoles  normal
    892,  # Día 11 Jueves     ANOMALÍA: visita de familiares + llenado de tanque
    519,  # Día 12 Viernes    vuelve a normal
    743,  # Día 13 Sábado     lavado + limpieza
    693,  # Día 14 Domingo    familia en casa

    # --- Semana 3 — 3 personas (un integrante viajó) ---
    446,  # Día 15 Lunes      baja consumo por menos personas
    430,  # Día 16 Martes     consumo bajo
    459,  # Día 17 Miércoles  normal con 3 personas
    453,  # Día 18 Jueves     normal
    476,  # Día 19 Viernes    ligero aumento
    618,  # Día 20 Sábado     lavado con 3 personas
    596,  # Día 21 Domingo    3 personas en casa

    # --- Semana 4 — 3 personas ---
    439,  # Día 22 Lunes      rutina normal
    982,  # Día 23 Martes     ANOMALÍA: llenado de tanque elevado + fuga en caño
    457,  # Día 24 Miércoles  vuelve a normal
    441,  # Día 25 Jueves     normal
    466,  # Día 26 Viernes    ligero aumento
    633,  # Día 27 Sábado     lavado + limpieza
    608,  # Día 28 Domingo    familia en casa
    469,  # Día 29 Lunes      normal
    458,  # Día 30 Martes     cierre de mes, consumo bajo
]

# Verificación rápida del total
_total = sum(consumo_diario)
assert 16800 <= _total <= 17200, f"Total fuera de rango: {_total} L"

# -----------------------------------------------------------------------------
# NÚMERO DE OCUPANTES POR DÍA
# Semanas 1–2: 4 personas | Semanas 3–4: 3 personas
# -----------------------------------------------------------------------------

ocupantes_por_dia = [
    4, 4, 4, 4, 4, 4, 4,   # Semana 1
    4, 4, 4, 4, 4, 4, 4,   # Semana 2
    3, 3, 3, 3, 3, 3, 3,   # Semana 3
    3, 3, 3, 3, 3, 3, 3,   # Semana 4
    3, 3,                   # Días 29–30
]

# -----------------------------------------------------------------------------
# DÍAS DEL PERÍODO
# -----------------------------------------------------------------------------

dias = list(range(1, len(consumo_diario) + 1))

# -----------------------------------------------------------------------------
# CONSUMO POR FRANJA HORARIA
# Franjas: [Madrugada 00–06h, Mañana 06–12h, Tarde 12–18h, Noche 18–24h]
#
# Distribución típica vivienda familiar en Cajamarca:
#   Madrugada (00–06h):  4% — nadie usa agua; pico aquí = fuga
#   Mañana    (06–12h): 42% — ducha, desayuno, cocina
#   Tarde     (12–18h): 28% — almuerzo, lavado de vajilla
#   Noche     (18–24h): 26% — cena, ducha nocturna
#
#   Sábados: mañana sube al 50% (lavado de ropa)
#   Días 11 y 23 (anomalías): madrugada sube (llenado nocturno/fuga)
# -----------------------------------------------------------------------------

def generar_franja(total, dia_index):
    """
    Genera el consumo por franja horaria para un día dado.
    Simula el patrón real de una vivienda familiar en Cajamarca.

    Args:
        total (float): Consumo total del día en litros.
        dia_index (int): Índice del día (0 = día 1).

    Returns:
        list: 4 valores en litros [madrugada, mañana, tarde, noche].
    """
    # Proporciones base día normal de semana
    props = np.array([0.04, 0.42, 0.28, 0.26])

    dia_semana = dia_index % 7  # 0=Lunes … 6=Domingo

    # Sábado: más lavado de ropa → mañana sube
    if dia_semana == 5:
        props = np.array([0.03, 0.50, 0.27, 0.20])

    # Domingo: familia en casa todo el día → tarde y noche suben
    if dia_semana == 6:
        props = np.array([0.04, 0.38, 0.32, 0.26])

    # Día 11 — visita + llenado de tanque al amanecer
    if dia_index == 10:
        props = np.array([0.07, 0.44, 0.29, 0.20])

    # Día 23 — fuga nocturna + llenado de tanque elevado
    if dia_index == 22:
        props = np.array([0.19, 0.40, 0.23, 0.18])

    # Ruido aleatorio pequeño para realismo
    ruido = np.random.uniform(-0.01, 0.01, 4)
    props = np.clip(props + ruido, 0.02, 0.6)
    props /= props.sum()

    return np.round(props * total, 1).tolist()


consumo_horario = [
    generar_franja(consumo_diario[i], i)
    for i in range(len(consumo_diario))
]

# -----------------------------------------------------------------------------
# BLOQUE DE VERIFICACIÓN (ejecutar datos.py directamente para revisar)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    total = sum(consumo_diario)
    print("=" * 52)
    print("  VERIFICACIÓN DE DATOS — Cajamarca")
    print("=" * 52)
    print(f"  Total del mes      : {total:,} litros = {total/1000:.3f} m³")
    print(f"  Promedio diario    : {total/len(consumo_diario):.1f} L/día")
    print(f"  Prom. semanas 1-2  : {sum(consumo_diario[:14])/14:.1f} L/día (4 personas)")
    print(f"  Prom. semanas 3-4  : {sum(consumo_diario[14:])/16:.1f} L/día (3 personas)")
    print(f"  Día máximo         : Día 23 — {consumo_diario[22]} L (anomalía)")
    print(f"  Día mínimo         : Día 16 — {consumo_diario[15]} L")
    print(f"  Días anómalos      : Día 11 ({consumo_diario[10]} L), Día 23 ({consumo_diario[22]} L)")
    print("=" * 52)
