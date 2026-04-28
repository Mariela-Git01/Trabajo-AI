# =============================================================================
# SISTEMA INTELIGENTE DE OPTIMIZACIÓN DEL USO DE AGUA EN EDIFICACIONES
# Versión 3.0 — Datos reales: Vivienda de Alquiler, 3 pisos — Cajamarca, Perú
# Consumo real anual 2025: 245 m³ = 245,000 litros
# Proveedor: SEDACAJ — Cajamarca
# =============================================================================
# Instalación de dependencias:
#   pip install numpy matplotlib scikit-learn rich
# =============================================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import LinearRegression
from sklearn.preprocessing  import PolynomialFeatures, StandardScaler
from sklearn.pipeline       import make_pipeline
from sklearn.ensemble       import IsolationForest
from sklearn.cluster        import KMeans
from sklearn.neural_network import MLPRegressor

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich         import box

from datos import (
    consumo_diario,           # Consumo mensual en litros (12 meses)
    dias,                     # Índices 1–12
    ocupantes_por_dia,        # Personas por mes
    consumo_horario,          # Franjas horarias estimadas por mes
    meses,                    # Nombres de los meses
    metros_cubicos_mes,       # m³ por mes
    litros_por_dia_mes,       # L/día promedio por mes
    litros_por_persona_mes,   # L/persona por mes
)

console = Console()

# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================

def cargar_config(ruta="config.json"):
    """Carga y retorna la configuración desde el archivo JSON."""
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        console.print(f"[green]✔ Configuración cargada:[/green] [cyan]{ruta}[/cyan]")
        return cfg
    except FileNotFoundError:
        console.print(f"[red]✘ No se encontró {ruta}[/red]")
        raise

# =============================================================================
# 1. TARIFA REAL SEDACAJ — CAJAMARCA (por tramos, aplicada mensualmente)
# =============================================================================

def calcular_costo_sedacaj(total_litros, config):
    """
    Calcula el costo del agua según la tarifa por tramos de SEDACAJ Cajamarca.

    Tramos vigentes (doméstico):
      0–8   m³ : S/. 0.880 / m³
      8–20  m³ : S/. 1.650 / m³
      20+   m³ : S/. 3.120 / m³
      Cargo fijo mensual: S/. 3.20

    Args:
        total_litros (float): Consumo total en litros.
        config (dict): Configuración con tramos tarifarios.

    Returns:
        float: Costo total estimado en soles (S/.).
    """
    total_m3 = total_litros / 1000
    rangos   = config["tarifa"]["rangos"]
    costo    = config["tarifa"]["cargo_fijo_soles"]

    for rango in rangos:
        desde  = rango["desde_m3"]
        hasta  = rango["hasta_m3"]
        precio = rango["precio_sol_por_m3"]
        if total_m3 > desde:
            m3_en_tramo = min(total_m3, hasta) - desde
            costo      += m3_en_tramo * precio
        if total_m3 <= hasta:
            break

    return round(costo, 2)

def calcular_costo_por_mes(config):
    """Retorna el costo SEDACAJ estimado para cada mes."""
    return [calcular_costo_sedacaj(m3 * 1000, config) for m3 in metros_cubicos_mes]

# =============================================================================
# 2. ANÁLISIS ESTADÍSTICO ANUAL
# =============================================================================

def analizar_datos(consumo, ocupantes, litros_persona_dia):
    """
    Calcula estadísticas descriptivas del consumo mensual anual.

    Args:
        consumo (list): Consumo mensual en litros (12 valores).
        ocupantes (list): Personas por mes.
        litros_persona_dia (float): Referencia de consumo por persona/día.

    Returns:
        dict: Estadísticas completas del año.
    """
    datos      = np.array(consumo)
    ocup       = np.array(ocupantes)
    per_capita = datos / ocup  # litros por persona por mes

    return {
        "promedio"          : float(np.mean(datos)),
        "maximo"            : float(np.max(datos)),
        "minimo"            : float(np.min(datos)),
        "desviacion"        : float(np.std(datos)),
        "total"             : float(np.sum(datos)),
        "total_m3"          : float(np.sum(datos) / 1000),
        "per_capita_prom"   : float(np.mean(per_capita)),
        "per_capita_max"    : float(np.max(per_capita)),
        "per_capita_min"    : float(np.min(per_capita)),
        "mes_maximo"        : int(np.argmax(datos)),
        "mes_minimo"        : int(np.argmin(datos)),
        "consumo_per_capita": per_capita.tolist(),
    }

# =============================================================================
# 3. ANÁLISIS POR FRANJA HORARIA — Detección de franjas críticas
# =============================================================================

def analizar_franjas(consumo_horario_mes, meses_lista, umbral_nocturno):
    """
    Analiza el consumo en la franja de madrugada (00–06h) para cada mes.

    Args:
        consumo_horario_mes (list): Lista de [madrugada, mañana, tarde, noche] por mes.
        meses_lista (list): Nombres de los meses.
        umbral_nocturno (float): Litros en madrugada que disparan alerta.

    Returns:
        dict: Promedios por franja y lista de meses con alerta.
    """
    matriz    = np.array(consumo_horario_mes)
    promedios = matriz.mean(axis=0).tolist()
    alertas   = []

    for i, fila in enumerate(matriz):
        if fila[0] > umbral_nocturno:
            alertas.append((meses_lista[i], round(fila[0], 1)))

    return {
        "promedios_franja" : promedios,
        "fugas_nocturnas"  : alertas,
        "matriz"           : matriz,
    }

# =============================================================================
# 4. DETECCIÓN DE ANOMALÍAS — Isolation Forest + Umbral clásico
# =============================================================================

def detectar_anomalias_if(consumo, meses_lista):
    """
    Usa Isolation Forest para detectar meses con consumo inusual.
    Con solo 12 puntos usa contamination baja para no sobre-detectar.

    Returns:
        tuple: (lista de anomalías, array de scores)
    """
    X         = np.array(consumo).reshape(-1, 1)
    # Con 12 muestras, contamination ~0.15 detecta 1–2 outliers reales
    modelo    = IsolationForest(contamination=0.15, random_state=42)
    etiquetas = modelo.fit_predict(X)
    scores    = modelo.decision_function(X)

    anomalias = [
        (meses_lista[i], consumo[i], round(scores[i], 4))
        for i, e in enumerate(etiquetas) if e == -1
    ]
    return anomalias, scores

def detectar_anomalias_umbral(consumo, meses_lista, promedio, factor):
    """Detección clásica: meses cuyo consumo supera factor × promedio."""
    umbral    = promedio * factor
    anomalias = [
        (meses_lista[i], v)
        for i, v in enumerate(consumo) if v > umbral
    ]
    return anomalias, umbral

# =============================================================================
# 5. CLUSTERING K-MEANS — Agrupación de meses por patrón de consumo
# =============================================================================

def clustering_kmeans(consumo_horario_mes, meses_lista, n_clusters):
    """
    Agrupa los meses en clusters según su patrón de consumo por franja.
    Para esta vivienda se esperan 3 grupos:
      - Meses de bajo consumo (Enero–Abril)
      - Meses de consumo medio (Julio–Setiembre)
      - Meses de consumo alto / pico (Mayo, Junio, Octubre–Diciembre)

    Returns:
        dict: Etiquetas, nombres de clusters y promedios.
    """
    X      = np.array(consumo_horario_mes)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    km   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etiq = km.fit_predict(Xs)

    totales = {}
    for i, lbl in enumerate(etiq):
        totales.setdefault(lbl, []).append(sum(consumo_horario_mes[i]))

    prom_c  = {k: np.mean(v) for k, v in totales.items()}
    orden   = sorted(prom_c, key=prom_c.get)
    nombres = {
        orden[i]: n
        for i, n in enumerate(
            ["Consumo bajo", "Consumo medio", "Consumo alto/pico"][:n_clusters]
        )
    }
    return {"etiquetas": etiq.tolist(), "nombres": nombres, "prom_cluster": prom_c}

# =============================================================================
# 6. CLASIFICACIÓN DEL CONSUMO
# =============================================================================

def clasificar_consumo(promedio, bajo, alto):
    if promedio < bajo:    return "BAJO"
    elif promedio <= alto: return "NORMAL"
    else:                  return "ALTO"

def clasificar_mes(valor, bajo, alto):
    if valor < bajo:    return "BAJO"
    elif valor <= alto: return "NORMAL"
    else:               return "ALTO"

# =============================================================================
# 7. REGRESIÓN POLINÓMICA — Tendencia del consumo mensual
# =============================================================================

def predecir_polinomica(indices, consumo, grado, meses_futuros):
    """
    Ajusta una curva polinómica al historial anual y la proyecta.
    Útil para capturar la curva de crecimiento de mitad de año.

    Args:
        indices (list): Índices 1–12.
        grado (int): Grado del polinomio.
        meses_futuros (int): Cuántos meses futuros predecir.

    Returns:
        tuple: (modelo, índices predichos, valores predichos)
    """
    X      = np.array(indices).reshape(-1, 1)
    y      = np.array(consumo)
    modelo = make_pipeline(PolynomialFeatures(degree=grado), LinearRegression())
    modelo.fit(X, y)

    ultimo  = max(indices)
    X_pred  = np.arange(ultimo + 1, ultimo + meses_futuros + 1).reshape(-1, 1)
    vals    = np.clip(modelo.predict(X_pred), 0, None)

    return modelo, X_pred.flatten(), vals

# =============================================================================
# 8. RED NEURONAL MLP — Predicción con ocupantes como variable
# =============================================================================

def predecir_red_neuronal(indices, consumo, ocupantes, cfg_nn, meses_futuros,
                          ocup_futuros=None):
    """
    Red neuronal que aprende la relación (mes, ocupantes) → consumo.

    Args:
        cfg_nn (dict): Configuración de la red.
        ocup_futuros (list): Ocupantes estimados para meses futuros.

    Returns:
        tuple: (modelo, índices predichos, valores predichos)
    """
    X  = np.column_stack([indices, ocupantes])
    y  = np.array(consumo)

    sx = StandardScaler(); sy = StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=(cfg_nn["neuronas_capa1"], cfg_nn["neuronas_capa2"]),
        activation="relu",
        max_iter=cfg_nn["epocas"],
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(Xs, ys)

    ultimo    = max(indices)
    idx_pred  = np.arange(ultimo + 1, ultimo + meses_futuros + 1)

    if ocup_futuros is None:
        ocup_futuros = [int(np.mean(ocupantes))] * meses_futuros

    Xf   = np.column_stack([idx_pred, ocup_futuros])
    Xfs  = sx.transform(Xf)
    vals = np.clip(
        sy.inverse_transform(mlp.predict(Xfs).reshape(-1, 1)).ravel(), 0, None
    )
    return mlp, idx_pred, vals

# =============================================================================
# 9. RECOMENDACIONES AUTOMÁTICAS
# =============================================================================

def generar_recomendaciones(stats, clasificacion, anomalias_if, fugas,
                            config, costos_mes):
    """
    Genera recomendaciones contextualizadas para la vivienda de alquiler
    en Cajamarca, considerando la tarifa SEDACAJ y el patrón anual.
    """
    recs     = []
    ref      = config["umbrales"]["litros_por_persona_dia"]
    costo_t  = sum(costos_mes)
    mes_max  = meses[stats["mes_maximo"]]
    mes_min  = meses[stats["mes_minimo"]]

    # Clasificación general
    if clasificacion == "BAJO":
        recs.append(("green", "✅ Consumo anual BAJO. Excelente gestión del agua."))
    elif clasificacion == "NORMAL":
        recs.append(("cyan",  "✅ Consumo anual dentro del rango NORMAL para la edificación."))
        recs.append(("cyan",  "   → Instale reductores de caudal en grifos y duchas de cada piso."))
    else:
        recs.append(("red",   "⚠️  Consumo anual ALTO. Supera el rango esperado."))
        recs.append(("red",   f"   → Mes pico: {mes_max} con {stats['maximo']:,.0f} L."))
        recs.append(("red",   "   → Revise instalaciones, tuberías y posibles fugas en los 3 pisos."))

    # Per cápita
    pc = stats["per_capita_prom"]
    recs.append(("yellow", f"💧 Consumo per cápita promedio mensual: {pc:,.1f} L/persona"))
    recs.append(("yellow",
        f"   → Mes de mayor consumo: {mes_max} | Mes de menor consumo: {mes_min}"))

    # Anomalías Isolation Forest
    if anomalias_if:
        recs.append(("red",
            f"🔴 Isolation Forest detectó {len(anomalias_if)} mes(es) anómalo(s):"))
        for mes_n, valor, score in anomalias_if:
            recs.append(("red",
                f"   → {mes_n}: {valor:,} L — Revisar ocupación y posibles fugas."))
    else:
        recs.append(("green", "🔴 Sin meses anómalos detectados por Isolation Forest. ✔"))

    # Fugas nocturnas
    if fugas:
        recs.append(("magenta",
            f"🌙 Alerta nocturna detectada en {len(fugas)} mes(es):"))
        for mes_n, litros in fugas:
            recs.append(("magenta",
                f"   → {mes_n}: {litros:,.1f} L en madrugada"))
        recs.append(("magenta",
            "   → Verifique llaves de paso en los 3 pisos durante la madrugada."))
    else:
        recs.append(("green", "🌙 Sin alertas nocturnas detectadas. ✔"))

    # Costo anual
    recs.append(("yellow",
        f"💰 Costo anual estimado SEDACAJ: S/. {costo_t:.2f}"))
    recs.append(("yellow",
        f"   → Promedio mensual: S/. {costo_t/12:.2f}/mes"))
    recs.append(("yellow",
        "   → Instalar medidores por piso podría identificar el piso de mayor consumo."))

    # Tendencia
    pico_meses = ["Mayo", "Junio", "Noviembre", "Diciembre"]
    recs.append(("cyan",
        f"📈 Los meses de mayor consumo son: {', '.join(pico_meses)}."))
    recs.append(("cyan",
        "   → Programar revisión de instalaciones antes de estos meses."))

    return recs

# =============================================================================
# 10. VISUALIZACIÓN — 2 gráficos simples para cualquier público
# =============================================================================

def visualizar_completo(indices, consumo, stats, umbral, anomalias_umbral,
                        anomalias_if, cluster_info, consumo_horario_mes,
                        idx_pred_poly, vals_pred_poly,
                        idx_pred_nn, vals_pred_nn,
                        config, costos_mes):
    """
    Genera 2 gráficos claros y fáciles de entender para cualquier persona,
    sin términos técnicos ni datos confusos.

    Gráfico 1 — Barras de consumo mensual:
      Muestra cuánta agua se usó cada mes, con colores simples
      (verde = poco, naranja = mucho) y el valor exacto encima de cada barra.

    Gráfico 2 — Estimación de los próximos 3 meses:
      Continúa la línea del año 2025 y muestra qué se espera consumir
      en enero, febrero y marzo de 2026.
    """
    meses_abr = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
    meses_fut_abr = ["Ene 26", "Feb 26", "Mar 26"]
    nombre   = config["edificio"]["nombre"]
    ciudad   = config["edificio"]["ciudad"]
    anio     = config["edificio"]["anio"]
    costo_t  = sum(costos_mes)
    meses_if = {m for m, _, _ in anomalias_if}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"{nombre}  —  {ciudad}, Perú  |  Año {anio}\n"
        f"Agua consumida en el año: {stats['total_m3']:.0f} m³  "
        f"(equivale a {stats['total_m3']*1000:,.0f} litros)  •  "
        f"Costo anual estimado: S/. {costo_t:.2f}",
        fontsize=12, fontweight="bold"
    )

    # ------------------------------------------------------------------
    # GRÁFICO 1 — ¿Cuánta agua se usó cada mes?
    # ------------------------------------------------------------------
    # Color: verde si está por debajo del promedio, naranja si está por encima
    colores = [
        "#e74c3c" if mes_n in meses_if           # rojo = mes inusual
        else "#e67e22" if v > stats["promedio"]  # naranja = más de lo normal
        else "#2ecc71"                            # verde = dentro de lo normal
        for mes_n, v in zip(meses, consumo)
    ]

    barras = ax1.bar(meses_abr, consumo, color=colores,
                     edgecolor="white", linewidth=0.8, zorder=2)

    # Valor en litros encima de cada barra (redondeado, fácil de leer)
    for barra, valor, mes_n in zip(barras, consumo, meses):
        etiqueta = f"{valor//1000}k L"  # ej: "49k L"
        color_txt = "#c0392b" if mes_n in meses_if else "#2c3e50"
        ax1.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + max(consumo) * 0.015,
            etiqueta,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color_txt
        )
        # Señalar meses inusuales con una etiqueta de alerta
        if mes_n in meses_if:
            ax1.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + max(consumo) * 0.085,
                "⚠ Mes inusual",
                ha="center", va="bottom",
                fontsize=7.5, color="#c0392b", fontstyle="italic"
            )

    # Línea del promedio anual
    ax1.axhline(
        stats["promedio"], color="#2c3e50",
        linestyle="--", lw=1.8, zorder=3,
        label=f"Promedio del año: {stats['promedio']:,.0f} L/mes"
    )

    # Leyenda con texto simple
    leyenda = [
        mpatches.Patch(color="#2ecc71", label="Consumo normal (por debajo del promedio)"),
        mpatches.Patch(color="#e67e22", label="Consumo alto (por encima del promedio)"),
        mpatches.Patch(color="#e74c3c", label="Mes inusual — revisar posibles fugas"),
    ]
    ax1.legend(handles=leyenda, fontsize=8.5, loc="upper left",
               framealpha=0.9)

    ax1.set_title("¿Cuánta agua se usó cada mes?",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Mes del año", fontsize=10)
    ax1.set_ylabel("Litros de agua usados", fontsize=10)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k")
    )
    ax1.set_ylim(0, max(consumo) * 1.28)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.set_axisbelow(True)

    # ------------------------------------------------------------------
    # GRÁFICO 2 — ¿Cuánta agua se usará los próximos meses?
    # ------------------------------------------------------------------
    # Historial 2025
    ax2.plot(meses_abr, consumo, "o-",
             color="#3498db", lw=2.2, markersize=7,
             label="Consumo real 2025", zorder=3)

    # Sombrear zona de predicción
    n_fut = len(idx_pred_nn)
    todos_x  = meses_abr + meses_fut_abr[:n_fut]
    todos_y  = list(consumo) + [None] * n_fut

    # Línea de predicción — promedio de ambos modelos para mostrar UNA sola línea
    vals_prom = [(p + n) / 2 for p, n in zip(vals_pred_poly, vals_pred_nn)]
    x_fut_pos = list(range(len(meses_abr), len(meses_abr) + n_fut))

    # Conectar el último mes real con la predicción
    ax2.plot(
        [len(meses_abr) - 1] + x_fut_pos,
        [consumo[-1]] + vals_prom,
        "o--", color="#e67e22", lw=2.2, markersize=7,
        label="Estimación próximos meses", zorder=3
    )

    # Valores encima de los puntos de predicción
    for xi, vp in zip(x_fut_pos, vals_prom):
        ax2.text(xi, vp + max(consumo) * 0.04,
                 f"{vp/1000:.0f}k L",
                 ha="center", fontsize=9, fontweight="bold", color="#e67e22")

    # Sombrear la zona futura
    ax2.axvspan(
        len(meses_abr) - 1 + 0.5, len(meses_abr) + n_fut - 0.5,
        alpha=0.08, color="#e67e22", label="Período estimado"
    )

    # Etiquetas en el eje X unificadas
    ax2.set_xticks(range(len(meses_abr) + n_fut))
    ax2.set_xticklabels(meses_abr + meses_fut_abr[:n_fut], fontsize=8.5)

    # Línea del promedio
    ax2.axhline(
        stats["promedio"], color="#2c3e50",
        linestyle=":", lw=1.5, alpha=0.6,
        label=f"Promedio 2025: {stats['promedio']:,.0f} L/mes"
    )

    ax2.set_title("¿Cuánta agua se usará los próximos meses?",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Mes", fontsize=10)
    ax2.set_ylabel("Litros de agua estimados", fontsize=10)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k")
    )
    ax2.set_ylim(0, max(consumo) * 1.28)
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    nombre_png = config["graficos"]["nombre_archivo"]
    plt.savefig(nombre_png, dpi=150, bbox_inches="tight")
    console.print(f"\n[green]📁 Gráfico guardado:[/green] [cyan]{nombre_png}[/cyan]")
    plt.show()

# =============================================================================
# 11. SALIDA ENRIQUECIDA CON RICH
# =============================================================================

def imprimir_encabezado(config, stats, costos_mes):
    ed      = config["edificio"]
    costo_t = sum(costos_mes)
    console.print(Panel(
        f"[bold white]Edificación :[/bold white] [cyan]{ed['nombre']}[/cyan]\n"
        f"[bold white]Ciudad      :[/bold white] [cyan]{ed['ciudad']} — {ed['departamento']}, Perú[/cyan]\n"
        f"[bold white]Tipo        :[/bold white] {ed['tipo']}  |  "
        f"[bold white]Pisos:[/bold white] {ed['pisos']}\n"
        f"[bold white]Período     :[/bold white] Año [yellow]{ed['anio']}[/yellow]\n"
        f"[bold white]Consumo anual real:[/bold white] "
        f"[yellow]{ed['consumo_anual_real_m3']} m³  "
        f"({ed['consumo_anual_real_litros']:,} litros)[/yellow]\n"
        f"[bold white]Proveedor   :[/bold white] [magenta]{config['tarifa']['proveedor']}[/magenta]  |  "
        f"[bold white]Costo anual estimado:[/bold white] [green]S/. {costo_t:.2f}[/green]\n"
        f"[bold white]Promedio mensual:[/bold white] [cyan]{stats['promedio']:,.0f} L/mes[/cyan]  |  "
        f"[bold white]Per cápita prom.:[/bold white] [cyan]{stats['per_capita_prom']:,.1f} L/persona/mes[/cyan]",
        title=(
            "[bold yellow]💧 SISTEMA INTELIGENTE DE OPTIMIZACIÓN DEL USO DE AGUA"
            " — v3.0  |  Cajamarca, Perú[/bold yellow]"
        ),
        border_style="blue", padding=(1, 3)
    ))

def imprimir_tabla_anual(config, costos_mes):
    """Tabla con los 12 meses tal como aparece en la imagen original."""
    console.rule("[bold blue]📋 TABLA DE CONSUMO ANUAL 2025[/bold blue]")
    t = Table(box=box.ROUNDED, header_style="bold cyan", min_width=82)
    t.add_column("MES",        style="bold white", min_width=11)
    t.add_column("N°PERS",     justify="right",    min_width=7)
    t.add_column("m³",         justify="right",    min_width=5)
    t.add_column("LITROS",     justify="right",    min_width=8)
    t.add_column("L/DÍA",      justify="right",    min_width=9)
    t.add_column("L/PERSONA",  justify="right",    min_width=10)
    t.add_column("COSTO S/.",  justify="right",    min_width=10)

    total_pers = sum(ocupantes_por_dia)
    total_m3   = sum(metros_cubicos_mes)
    total_l    = sum(consumo_diario)
    total_s    = sum(costos_mes)

    for i, mes_n in enumerate(meses):
        # Color según nivel de consumo
        v = consumo_diario[i]
        bajo = config["umbrales"]["consumo_bajo_litros"]
        alto = config["umbrales"]["consumo_alto_litros"]
        if v >= alto:   col = "red"
        elif v < bajo:  col = "green"
        else:           col = "cyan"

        t.add_row(
            mes_n,
            str(ocupantes_por_dia[i]),
            str(metros_cubicos_mes[i]),
            f"[{col}]{consumo_diario[i]:,}[/{col}]",
            f"{litros_por_dia_mes[i]:,.2f}",
            f"{litros_por_persona_mes[i]:,.2f}",
            f"S/. {costos_mes[i]:.2f}",
        )

    t.add_row(
        "[bold]TOTAL AÑO[/bold]",
        f"[bold]{total_pers}[/bold]",
        f"[bold]{total_m3}[/bold]",
        f"[bold yellow]{total_l:,}[/bold yellow]",
        "—",
        "—",
        f"[bold green]S/. {total_s:.2f}[/bold green]",
    )
    console.print(t)

def imprimir_estadisticas(stats, config):
    console.rule("[bold blue]📊 [1] ANÁLISIS ESTADÍSTICO ANUAL[/bold blue]")
    ref = config["umbrales"]["litros_por_persona_dia"]

    t = Table(box=box.ROUNDED, header_style="bold cyan", min_width=60)
    t.add_column("Métrica",              style="bold white", min_width=35)
    t.add_column("Valor",                justify="right",    min_width=23)
    t.add_row("Período analizado",       "12 meses (Año 2025)")
    t.add_row("Consumo total anual",
              f"[bold]{stats['total']:,.0f} L  =  {stats['total_m3']:.0f} m³[/bold]")
    t.add_row("Promedio mensual",
              f"[cyan]{stats['promedio']:,.1f} L/mes[/cyan]")
    t.add_row("Mes de mayor consumo",
              f"[red]{meses[stats['mes_maximo']]} — {stats['maximo']:,.0f} L[/red]")
    t.add_row("Mes de menor consumo",
              f"[green]{meses[stats['mes_minimo']]} — {stats['minimo']:,.0f} L[/green]")
    t.add_row("Desviación estándar",
              f"{stats['desviacion']:,.1f} L")
    t.add_row("─" * 33,                "─" * 21)
    t.add_row("Per cápita prom. mensual",
              f"[yellow]{stats['per_capita_prom']:,.1f} L/persona/mes[/yellow]")
    t.add_row("Per cápita máximo",
              f"{stats['per_capita_max']:,.1f} L/persona/mes")
    t.add_row("Per cápita mínimo",
              f"{stats['per_capita_min']:,.1f} L/persona/mes")
    t.add_row(f"Ref. OMS/Perú ({ref} L/pers/día)",
              f"= {ref * 30:,} L/persona/mes estimado")
    console.print(t)

def imprimir_tarifa(stats, costos_mes, config):
    console.rule("[bold yellow]💰 [2] DESGLOSE TARIFARIO ANUAL — SEDACAJ CAJAMARCA[/bold yellow]")
    total_s = sum(costos_mes)

    t = Table(box=box.SIMPLE_HEAVY, header_style="bold yellow")
    t.add_column("Mes",         min_width=12)
    t.add_column("m³",          justify="right", min_width=6)
    t.add_column("Costo S/.",   justify="right", min_width=10)
    t.add_column("Tramo ppal.", min_width=20)

    for i, mes_n in enumerate(meses):
        m3   = metros_cubicos_mes[i]
        costo = costos_mes[i]
        if m3 <= 8:   tramo = "0–8 m³ (social)"
        elif m3 <= 20: tramo = "8–20 m³ (básico)"
        else:          tramo = "20+ m³ (excedente)"
        t.add_row(mes_n, str(m3), f"S/. {costo:.2f}", tramo)

    t.add_row("─" * 10, "─" * 4, "─" * 8, "─" * 18)
    t.add_row(
        "[bold]TOTAL ANUAL[/bold]",
        f"[bold]{sum(metros_cubicos_mes)}[/bold]",
        f"[bold green]S/. {total_s:.2f}[/bold green]",
        f"Prom. S/. {total_s/12:.2f}/mes",
    )
    console.print(t)

def imprimir_anomalias(anomalias_if, anomalias_umb, umbral, factor, stats):
    console.rule("[bold red]🔍 [3] DETECCIÓN DE ANOMALÍAS[/bold red]")

    console.print(f"\n  [bold]Isolation Forest[/bold] — {len(anomalias_if)} mes(es) anómalo(s):")
    if anomalias_if:
        t = Table(box=box.SIMPLE, header_style="bold red")
        t.add_column("Mes",             justify="center")
        t.add_column("Consumo",         justify="right")
        t.add_column("Score IA",        justify="right")
        t.add_column("Exceso vs prom.", justify="right")
        for mes_n, valor, score in anomalias_if:
            exceso = valor - stats["promedio"]
            t.add_row(mes_n,
                      f"[red]{valor:,} L[/red]",
                      f"{score:.4f}",
                      f"+{exceso:,.0f} L")
        console.print(t)
    else:
        console.print("  [green]Sin meses anómalos detectados. ✔[/green]")

    console.print(
        f"\n  [bold]Umbral clásico[/bold] "
        f"(×{factor} = {umbral:,.0f} L) — {len(anomalias_umb)} mes(es):"
    )
    for mes_n, valor in anomalias_umb:
        console.print(f"    ⚠  {mes_n}: [red]{valor:,} L[/red]")
    if not anomalias_umb:
        console.print("  [green]Ningún mes supera el umbral ×1.5. ✔[/green]")

def imprimir_franjas(res_franjas, umbral_noch):
    console.rule("[bold magenta]🕐 [4] ANÁLISIS POR FRANJA HORARIA (promedio mensual)[/bold magenta]")
    nombres = ["Madrugada 00–06h", "Mañana 06–12h", "Tarde 12–18h", "Noche 18–24h"]
    obs     = [
        "Consumo mínimo. Alerta si supera umbral (posible fuga).",
        "Pico principal: ducha, desayuno, cocina.",
        "Almuerzo, limpieza de habitaciones.",
        "Cena, ducha nocturna.",
    ]
    total_p = sum(res_franjas["promedios_franja"])

    t = Table(box=box.ROUNDED, header_style="bold magenta")
    t.add_column("Franja",            min_width=20)
    t.add_column("Promedio L/mes",    justify="right", min_width=14)
    t.add_column("% del mes",         justify="right", min_width=9)
    t.add_column("Observación",       min_width=42)

    for nombre, prom, ob in zip(nombres, res_franjas["promedios_franja"], obs):
        pct   = prom / total_p * 100
        color = "red" if "Madrugada" in nombre and prom > umbral_noch else "white"
        t.add_row(nombre, f"[{color}]{prom:,.1f}[/{color}]", f"{pct:.1f}%", ob)
    console.print(t)

    fugas = res_franjas["fugas_nocturnas"]
    if fugas:
        console.print(f"\n  [bold red]🌙 Alertas nocturnas ({len(fugas)} mes(es)):[/bold red]")
        for mes_n, litros in fugas:
            console.print(
                f"    → {mes_n}: [red]{litros:,.1f} L[/red] en madrugada "
                f"(umbral: {umbral_noch:,} L)"
            )
        console.print(
            "  [magenta]   → Revisar llaves de paso y tuberías en cada piso.[/magenta]"
        )
    else:
        console.print(
            f"\n  [green]🌙 Sin alertas nocturnas. "
            f"Consumo en madrugada dentro del umbral ({umbral_noch:,} L). ✔[/green]"
        )

def imprimir_clustering(cluster_info, indices, consumo_mes, n_clusters):
    console.rule("[bold green]🔵 [5] CLUSTERING K-MEANS — Agrupación de Meses[/bold green]")
    etiq    = cluster_info["etiquetas"]
    nombres = cluster_info["nombres"]

    t = Table(box=box.ROUNDED, header_style="bold green")
    t.add_column("Cluster",        min_width=22)
    t.add_column("Meses",          min_width=42)
    t.add_column("Prom. consumo",  justify="right")
    for k in sorted(set(etiq)):
        meses_k  = [meses[i]       for i, e in enumerate(etiq) if e == k]
        prom_k   = np.mean([consumo_mes[i] for i, e in enumerate(etiq) if e == k])
        nombre   = nombres.get(k, f"Grupo {k}")
        t.add_row(
            f"[bold]{nombre}[/bold]",
            ", ".join(meses_k),
            f"{prom_k:,.0f} L",
        )
    console.print(t)

def imprimir_predicciones(idx_poly, vals_poly, idx_nn, vals_nn, grado, config):
    console.rule("[bold cyan]🤖 [6] PREDICCIONES CON IA — Próximos 3 meses (2026)[/bold cyan]")
    bajo = config["umbrales"]["consumo_bajo_litros"]
    alto = config["umbrales"]["consumo_alto_litros"]
    oc   = config["red_neuronal"]
    meses_pred = ["Enero 2026", "Febrero 2026", "Marzo 2026"]

    console.print(f"\n  [bold magenta]Regresión Polinómica[/bold magenta] — grado {grado}:")
    t1 = Table(box=box.SIMPLE, header_style="bold magenta")
    t1.add_column("Mes",            min_width=14)
    t1.add_column("Litros est.",    justify="right")
    t1.add_column("m³ est.",        justify="right")
    t1.add_column("Categoría")
    for i, (idx, v) in enumerate(zip(idx_poly, vals_poly)):
        cat   = clasificar_mes(v, bajo, alto)
        color = {"BAJO": "green", "NORMAL": "cyan", "ALTO": "red"}.get(cat, "white")
        nombre_m = meses_pred[i] if i < len(meses_pred) else f"Mes {idx}"
        t1.add_row(nombre_m, f"{v:,.0f} L", f"{v/1000:.2f} m³",
                   f"[{color}]{cat}[/{color}]")
    console.print(t1)

    console.print(
        f"\n  [bold green]Red Neuronal MLP[/bold green] — "
        f"capas ({oc['neuronas_capa1']}→{oc['neuronas_capa2']} neuronas, "
        f"{oc['epocas']} épocas):"
    )
    t2 = Table(box=box.SIMPLE, header_style="bold green")
    t2.add_column("Mes",            min_width=14)
    t2.add_column("Litros est.",    justify="right")
    t2.add_column("m³ est.",        justify="right")
    t2.add_column("Categoría")
    for i, (idx, v) in enumerate(zip(idx_nn, vals_nn)):
        cat   = clasificar_mes(v, bajo, alto)
        color = {"BAJO": "green", "NORMAL": "cyan", "ALTO": "red"}.get(cat, "white")
        nombre_m = meses_pred[i] if i < len(meses_pred) else f"Mes {idx}"
        t2.add_row(nombre_m, f"{v:,.0f} L", f"{v/1000:.2f} m³",
                   f"[{color}]{cat}[/{color}]")
    console.print(t2)

def imprimir_recomendaciones(recomendaciones):
    console.rule("[bold yellow]💡 [7] RECOMENDACIONES[/bold yellow]")
    for color, msg in recomendaciones:
        console.print(f"  [{color}]{msg}[/{color}]")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    config = cargar_config("config.json")

    factor       = config["umbrales"]["factor_anomalia"]
    bajo         = config["umbrales"]["consumo_bajo_litros"]
    alto         = config["umbrales"]["consumo_alto_litros"]
    ref_persona  = config["umbrales"]["litros_por_persona_dia"]
    meses_fut    = config["prediccion"]["dias_futuros"]   # meses futuros a predecir
    grado_poly   = config["prediccion"]["grado_polinomio"]
    n_clusters   = config["clustering"]["num_clusters"]
    umbral_noch  = config["alertas"]["fuga_nocturna_umbral_litros"]

    # ── Estadísticas ────────────────────────────────────────────────────────
    stats = analizar_datos(consumo_diario, ocupantes_por_dia, ref_persona)

    # ── Costos SEDACAJ por mes ───────────────────────────────────────────────
    costos_mes = calcular_costo_por_mes(config)

    # ── Encabezado ───────────────────────────────────────────────────────────
    imprimir_encabezado(config, stats, costos_mes)

    # ── Tabla anual completa (fiel a la imagen) ───────────────────────────────
    imprimir_tabla_anual(config, costos_mes)

    # [1] Estadísticas
    imprimir_estadisticas(stats, config)

    # [2] Tarifa desglosada
    imprimir_tarifa(stats, costos_mes, config)

    # [3] Anomalías
    anomalias_if, _        = detectar_anomalias_if(consumo_diario, meses)
    anomalias_umb, umbral  = detectar_anomalias_umbral(
        consumo_diario, meses, stats["promedio"], factor
    )
    imprimir_anomalias(anomalias_if, anomalias_umb, umbral, factor, stats)

    # [4] Franjas horarias
    res_franjas = analizar_franjas(consumo_horario, meses, umbral_noch)
    imprimir_franjas(res_franjas, umbral_noch)

    # [5] Clustering
    cluster_info = clustering_kmeans(consumo_horario, meses, n_clusters)
    imprimir_clustering(cluster_info, dias, consumo_diario, n_clusters)

    # [6] Predicciones
    _, idx_poly, vals_poly = predecir_polinomica(
        dias, consumo_diario, grado_poly, meses_fut
    )
    _, idx_nn, vals_nn = predecir_red_neuronal(
        dias, consumo_diario, ocupantes_por_dia,
        config["red_neuronal"], meses_fut
    )
    imprimir_predicciones(idx_poly, vals_poly, idx_nn, vals_nn, grado_poly, config)

    # [7] Recomendaciones
    clasificacion = clasificar_consumo(stats["promedio"], bajo, alto)
    recs = generar_recomendaciones(
        stats, clasificacion, anomalias_if,
        res_franjas["fugas_nocturnas"], config, costos_mes
    )
    imprimir_recomendaciones(recs)

    # [8] Gráfico
    console.rule("[bold blue]📈 [8] GENERANDO GRÁFICO[/bold blue]")
    visualizar_completo(
        dias, consumo_diario, stats,
        umbral, anomalias_umb, anomalias_if,
        cluster_info, consumo_horario,
        idx_poly, vals_poly,
        idx_nn, vals_nn,
        config, costos_mes,
    )

    console.print(Panel(
        f"[bold green]✔ Análisis completado — "
        f"Vivienda de Alquiler, 3 Pisos — Cajamarca, Perú[/bold green]\n"
        f"Consumo anual: {stats['total_m3']:.0f} m³  •  "
        f"Costo anual SEDACAJ: S/. {sum(costos_mes):.2f}  •  "
        f"Promedio mensual: {stats['promedio']:,.0f} L/mes",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
