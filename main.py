# =============================================================================
# SISTEMA DE OPTIMIZACIÓN DEL USO DE AGUA EN EDIFICACIONES
# Versión 2.1 — Datos reales: Vivienda familiar, Cajamarca, Perú
# Consumo real: 17 m³/mes | Habitantes: 3–4 personas
# Proveedor: SEDACAJ — Cajamarca
# =============================================================================
# pip install numpy matplotlib scikit-learn rich
# =============================================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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

from datos import consumo_diario, dias, ocupantes_por_dia, consumo_horario

console = Console()

# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================

def cargar_config(ruta="config.json"):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        console.print(f"[green]✔ Configuración cargada:[/green] [cyan]{ruta}[/cyan]")
        return cfg
    except FileNotFoundError:
        console.print(f"[red]✘ No se encontró {ruta}[/red]")
        raise

# =============================================================================
# 1. TARIFA REAL SEDACAJ — CAJAMARCA (por tramos)
# =============================================================================

def calcular_costo_sedacaj(total_litros, config):
    """
    Calcula el costo real del agua según la tarifa por tramos de SEDACAJ
    Cajamarca. Los tramos y precios están definidos en config.json.

    Tramos vigentes SEDACAJ doméstico:
      0–8   m³ : S/. 0.880 / m³
      8–20  m³ : S/. 1.650 / m³
      20+   m³ : S/. 3.120 / m³
      Cargo fijo mensual: S/. 3.20

    Args:
        total_litros (float): Consumo total del período en litros.
        config (dict): Configuración con tramos tarifarios.

    Returns:
        float: Costo total estimado en soles (S/.).
    """
    total_m3   = total_litros / 1000
    rangos     = config["tarifa"]["rangos"]
    costo      = config["tarifa"]["cargo_fijo_soles"]

    for rango in rangos:
        desde  = rango["desde_m3"]
        hasta  = rango["hasta_m3"]
        precio = rango["precio_sol_por_m3"]

        if total_m3 > desde:
            m3_en_tramo = min(total_m3, hasta) - desde
            costo += m3_en_tramo * precio

        if total_m3 <= hasta:
            break

    return round(costo, 2)

# =============================================================================
# 2. ANÁLISIS ESTADÍSTICO
# =============================================================================

def analizar_datos(consumo, ocupantes, litros_persona_dia):
    """
    Calcula estadísticas descriptivas del consumo real e incluye
    análisis per cápita comparado con la referencia OMS/Perú
    (150 L/persona/día para zona urbana).

    Args:
        consumo (list): Consumo diario en litros.
        ocupantes (list): Número de personas por día.
        litros_persona_dia (float): Referencia de consumo por persona.

    Returns:
        dict: Estadísticas completas del período.
    """
    datos      = np.array(consumo)
    ocup       = np.array(ocupantes)
    per_capita = datos / ocup

    return {
        "promedio"          : np.mean(datos),
        "maximo"            : np.max(datos),
        "minimo"            : np.min(datos),
        "desviacion"        : np.std(datos),
        "total"             : np.sum(datos),
        "total_m3"          : np.sum(datos) / 1000,
        "per_capita_prom"   : np.mean(per_capita),
        "per_capita_max"    : np.max(per_capita),
        "per_capita_min"    : np.min(per_capita),
        "dias_sobre_ref"    : int(np.sum(per_capita > litros_persona_dia)),
        "consumo_per_capita": per_capita.tolist(),
    }

# =============================================================================
# 3. ANÁLISIS POR FRANJA HORARIA — Detección de fugas nocturnas
# =============================================================================

def analizar_franjas(consumo_horario, dias, umbral_nocturno):
    """
    Analiza el consumo en la franja de madrugada (00–06h).
    En Cajamarca el suministro de SEDACAJ llega por horas en algunos
    sectores, por lo que el llenado de tanque a la madrugada puede
    confundirse con fuga. Se reportan ambos casos para que el usuario
    decida.

    Args:
        consumo_horario (list): Consumo por franja por día (4 franjas).
        dias (list): Lista de números de día.
        umbral_nocturno (float): Litros en madrugada que disparan alerta.

    Returns:
        dict: Promedios por franja y lista de noches con alerta.
    """
    matriz    = np.array(consumo_horario)
    promedios = matriz.mean(axis=0).tolist()
    alertas   = []

    for i, fila in enumerate(matriz):
        if fila[0] > umbral_nocturno:
            alertas.append((dias[i], round(fila[0], 1)))

    return {"promedios_franja": promedios, "fugas_nocturnas": alertas, "matriz": matriz}

# =============================================================================
# 4. DETECCIÓN DE ANOMALÍAS — Isolation Forest
# =============================================================================

def detectar_anomalias_if(consumo, dias):
    """
    Usa Isolation Forest para detectar días con consumo inusual sin
    necesidad de un umbral fijo. Detecta visitas inesperadas, llenados
    de tanque o fugas puntuales de forma automática.

    Returns:
        tuple: (lista de anomalías, array de scores)
    """
    X         = np.array(consumo).reshape(-1, 1)
    modelo    = IsolationForest(contamination=0.08, random_state=42)
    etiquetas = modelo.fit_predict(X)
    scores    = modelo.decision_function(X)

    anomalias = [
        (dias[i], consumo[i], round(scores[i], 4))
        for i, e in enumerate(etiquetas) if e == -1
    ]
    return anomalias, scores

def detectar_anomalias_umbral(consumo, dias, promedio, factor):
    """Detección clásica: días cuyo consumo supera factor × promedio."""
    umbral    = promedio * factor
    anomalias = [(dias[i], v) for i, v in enumerate(consumo) if v > umbral]
    return anomalias, umbral

# =============================================================================
# 5. CLUSTERING K-MEANS
# =============================================================================

def clustering_kmeans(consumo_horario, dias, n_clusters):
    """
    Agrupa los días en clusters según su patrón de consumo por franja.
    Para esta vivienda en Cajamarca se esperan 3 grupos naturales:
      - Días de semana normales (consumo moderado)
      - Fines de semana / lavado (consumo alto)
      - Días anómalos (picos por visitas, llenado de tanque o fuga)

    Returns:
        dict: Etiquetas, nombres de clusters y promedios.
    """
    X      = np.array(consumo_horario)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etiq     = km.fit_predict(Xs)

    totales  = {}
    for i, lbl in enumerate(etiq):
        totales.setdefault(lbl, []).append(sum(consumo_horario[i]))

    prom_c  = {k: np.mean(v) for k, v in totales.items()}
    orden   = sorted(prom_c, key=prom_c.get)
    nombres = {orden[i]: n for i, n in enumerate(
        ["Día normal semana", "Fin de semana", "Día anómalo/pico"][:n_clusters]
    )}

    return {"etiquetas": etiq.tolist(), "nombres": nombres, "prom_cluster": prom_c}

# =============================================================================
# 6. CLASIFICACIÓN
# =============================================================================

def clasificar_consumo(promedio, bajo, alto):
    if promedio < bajo:    return "BAJO"
    elif promedio <= alto: return "NORMAL"
    else:                  return "ALTO"

def clasificar_dia(valor, bajo, alto):
    if valor < bajo:    return "BAJO"
    elif valor <= alto: return "NORMAL"
    else:               return "ALTO"

# =============================================================================
# 7. REGRESIÓN POLINÓMICA
# =============================================================================

def predecir_polinomica(dias, consumo, grado, dias_futuros):
    """
    Ajusta una curva polinómica al historial y la proyecta hacia adelante.
    Captura la caída de consumo al pasar de 4 a 3 personas (semana 3),
    que no sería visible con una línea recta.

    Args:
        grado (int): Grado del polinomio (configurado en config.json).

    Returns:
        tuple: (modelo, días predichos, valores predichos)
    """
    X      = np.array(dias).reshape(-1, 1)
    y      = np.array(consumo)
    modelo = make_pipeline(PolynomialFeatures(degree=grado), LinearRegression())
    modelo.fit(X, y)

    ultimo   = max(dias)
    X_pred   = np.arange(ultimo + 1, ultimo + dias_futuros + 1).reshape(-1, 1)
    vals     = np.clip(modelo.predict(X_pred), 0, None)

    return modelo, X_pred.flatten(), vals

# =============================================================================
# 8. RED NEURONAL MLP
# =============================================================================

def predecir_red_neuronal(dias, consumo, ocupantes, cfg_nn, dias_futuros, ocup_futuros=None):
    """
    Red neuronal que aprende la relación (día, ocupantes) → consumo.
    Más realista que la regresión porque considera cuántas personas
    estarán en casa los días predichos.

    Args:
        cfg_nn (dict): Configuración de la red (épocas, neuronas).
        ocup_futuros (list): Estimación de ocupantes para días futuros.

    Returns:
        tuple: (modelo, días predichos, valores predichos)
    """
    X  = np.column_stack([dias, ocupantes])
    y  = np.array(consumo)

    sx = StandardScaler();  sy = StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=(cfg_nn["neuronas_capa1"], cfg_nn["neuronas_capa2"]),
        activation="relu",
        max_iter=cfg_nn["epocas"],
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(Xs, ys)

    ultimo    = max(dias)
    dias_pred = np.arange(ultimo + 1, ultimo + dias_futuros + 1)

    if ocup_futuros is None:
        ocup_futuros = [int(np.mean(ocupantes))] * dias_futuros

    Xf   = np.column_stack([dias_pred, ocup_futuros])
    Xfs  = sx.transform(Xf)
    vals = np.clip(
        sy.inverse_transform(mlp.predict(Xfs).reshape(-1, 1)).ravel(), 0, None
    )
    return mlp, dias_pred, vals

# =============================================================================
# 9. RECOMENDACIONES AUTOMÁTICAS
# =============================================================================

def generar_recomendaciones(stats, clasificacion, anomalias_if, fugas, config, costo):
    """
    Genera recomendaciones contextualizadas a la realidad de una
    vivienda familiar en Cajamarca, considerando la tarifa SEDACAJ
    y el patrón de suministro por horas de la ciudad.
    """
    recs = []
    ref  = config["umbrales"]["litros_por_persona_dia"]

    # Clasificación general
    if clasificacion == "BAJO":
        recs.append(("green",   "✅ Consumo general BAJO. Excelente gestión del agua."))
        recs.append(("green",   "   → Continúe con los hábitos actuales."))
    elif clasificacion == "NORMAL":
        recs.append(("cyan",    "✅ Consumo dentro del rango NORMAL para la vivienda."))
        recs.append(("cyan",    "   → Considere instalar ahorradores de agua en grifos y ducha."))
    else:
        recs.append(("red",     "⚠️  Consumo ALTO. Supera el rango esperado para el hogar."))
        recs.append(("red",     "   → Revise grifos, inodoros y conexiones en busca de fugas."))

    # Per cápita
    pc = stats["per_capita_prom"]
    recs.append(("yellow", f"💧 Consumo per cápita promedio: {pc:.1f} L/persona/día"))
    if pc > ref:
        recs.append(("yellow",
            f"   → Supera la referencia de {ref} L/persona/día en +{pc - ref:.1f} L."))
        recs.append(("yellow",
            "   → Reduzca el tiempo de ducha y lave ropa con carga completa."))
    else:
        recs.append(("green",
            f"   → Dentro de la referencia de {ref} L/persona/día. ✔"))

    # Anomalías
    if anomalias_if:
        recs.append(("red",
            f"🔴 Isolation Forest detectó {len(anomalias_if)} día(s) anómalo(s):"))
        causas = {11: "Visita de familiares + llenado de tanque",
                  23: "Fuga en caño + llenado de tanque nocturno"}
        for dia, valor, score in anomalias_if:
            causa = causas.get(dia, "Uso extraordinario")
            recs.append(("red",
                f"   → Día {dia}: {valor} L  [{causa}]"))

    # Fugas nocturnas
    if fugas:
        recs.append(("magenta",
            f"🌙 Alerta nocturna detectada en {len(fugas)} noche(s):"))
        for dia, litros in fugas:
            recs.append(("magenta",
                f"   → Noche del día {dia}: {litros} L en madrugada (00–06h)"))
        recs.append(("magenta",
            "   → En Cajamarca el agua de SEDACAJ puede llegar de madrugada."))
        recs.append(("magenta",
            "   → Verifique si fue llenado de tanque o fuga cerrando la llave de paso."))
    else:
        recs.append(("green",
            "🌙 Sin alertas nocturnas. Consumo en madrugada dentro del umbral. ✔"))

    # Costo
    recs.append(("yellow",
        f"💰 Costo estimado del mes (SEDACAJ Cajamarca): S/. {costo:.2f}"))
    recs.append(("yellow",
        "   → Incluye cargo fijo + consumo por tramos tarifarios vigentes."))

    return recs

# =============================================================================
# 10. VISUALIZACIÓN — 4 paneles integrados
# =============================================================================

def visualizar_completo(dias, consumo, stats, umbral, anomalias_umbral,
                        anomalias_if, cluster_info, consumo_horario,
                        dias_pred_poly, vals_pred_poly,
                        dias_pred_nn, vals_pred_nn, config, costo):

    bajo    = config["umbrales"]["consumo_bajo_litros"]
    alto    = config["umbrales"]["consumo_alto_litros"]
    nombre  = config["edificio"]["nombre"]
    ciudad  = config["edificio"]["ciudad"]
    franjas = ["Madrugada\n00–06h", "Mañana\n06–12h",
               "Tarde\n12–18h",    "Noche\n18–24h"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Sistema de Optimización del Uso de Agua — {nombre}  |  {ciudad}, Perú\n"
        f"Consumo real: {stats['total_m3']:.3f} m³  •  "
        f"Promedio diario: {stats['promedio']:.0f} L/día  •  "
        f"Costo estimado SEDACAJ: S/. {costo:.2f}",
        fontsize=12, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.33)

    dias_if = {d for d, _, _ in anomalias_if}

    # ------------------------------------------------------------------
    # PANEL 1 — Consumo diario + anomalías + predicciones
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    colores = []
    for i, v in enumerate(consumo):
        if dias[i] in dias_if:      colores.append("#e74c3c")
        elif v >= alto:             colores.append("#e67e22")
        elif v < bajo:              colores.append("#2ecc71")
        else:                       colores.append("#3498db")

    ax1.bar(dias, consumo, color=colores, alpha=0.85, zorder=2)
    ax1.axhline(stats["promedio"], color="#2c3e50", linestyle="--",
                lw=1.7, label=f"Promedio: {stats['promedio']:.0f} L", zorder=3)
    ax1.axhline(umbral, color="#c0392b", linestyle=":",
                lw=1.4, label=f"Umbral ×{config['umbrales']['factor_anomalia']}: {umbral:.0f} L", zorder=3)

    for dia, valor, _ in anomalias_if:
        ax1.annotate(f"⚠{valor}L",
                     xy=(dia, valor), xytext=(dia, valor + 50),
                     ha="center", fontsize=7.5, color="#c0392b", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

    ax1.plot(dias_pred_poly, vals_pred_poly, "s--", color="#8e44ad",
             lw=1.6, markersize=5, label="Predicción Polinómica")
    ax1.plot(dias_pred_nn, vals_pred_nn, "D--", color="#16a085",
             lw=1.6, markersize=5, label="Predicción Red Neuronal")

    # Sombrear período de 4 personas vs 3 personas
    ax1.axvspan(0.5,  14.5, alpha=0.05, color="#3498db")
    ax1.axvspan(14.5, 30.5, alpha=0.05, color="#2ecc71")
    ax1.text(7,  max(consumo) * 1.18, "4 personas",  ha="center", fontsize=8, color="#2980b9")
    ax1.text(22, max(consumo) * 1.18, "3 personas",  ha="center", fontsize=8, color="#27ae60")

    p_an = mpatches.Patch(color="#e74c3c", label="Anomalía (IF)")
    p_al = mpatches.Patch(color="#e67e22", label=f"Alto (≥{alto} L)")
    p_nm = mpatches.Patch(color="#3498db", label=f"Normal ({bajo}–{alto} L)")
    p_bj = mpatches.Patch(color="#2ecc71", label=f"Bajo (<{bajo} L)")
    hdls, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=hdls + [p_an, p_al, p_nm, p_bj],
               fontsize=7, ncol=3, loc="upper right")
    ax1.set_title("Consumo Diario + Anomalías + Predicciones", fontweight="bold")
    ax1.set_xlabel("Día del mes")
    ax1.set_ylabel("Litros / día")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(consumo) * 1.28)
    ax1.set_xticks(list(dias) + list(dias_pred_poly))

    # ------------------------------------------------------------------
    # PANEL 2 — Heatmap por franja horaria
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    matriz = np.array(consumo_horario).T   # (4 franjas × 30 días)

    im = ax2.imshow(matriz, aspect="auto", cmap="YlOrRd",
                    extent=[0.5, len(dias) + 0.5, 3.5, -0.5])
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Litros por franja", fontsize=9)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(franjas, fontsize=8)
    ax2.set_xlabel("Día del mes")
    ax2.set_title("Consumo por Franja Horaria (Heatmap)", fontweight="bold")

    for dia, _, _ in anomalias_if:
        ax2.axvline(x=dia, color="#e74c3c", lw=1.5, alpha=0.7, linestyle="--")
    for d in dias:
        if d % 7 in (6, 0):
            ax2.axvline(x=d, color="#3498db", lw=0.6, alpha=0.25)

    # ------------------------------------------------------------------
    # PANEL 3 — Clustering K-Means
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    etiq      = cluster_info["etiquetas"]
    nombres_c = cluster_info["nombres"]
    colores_c = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    for i, (dia, val) in enumerate(zip(dias, consumo)):
        lbl = etiq[i]
        ax3.scatter(dia, val,
                    color=colores_c[lbl % len(colores_c)],
                    s=65, zorder=3, edgecolors="white", linewidths=0.6)

    ax3.axhline(stats["promedio"], color="#2c3e50", linestyle="--",
                lw=1.2, alpha=0.5)
    handles_c = [
        mpatches.Patch(color=colores_c[k % len(colores_c)],
                       label=f"Cluster {k}: {nombres_c.get(k, f'Grupo {k}')}")
        for k in sorted(set(etiq))
    ]
    ax3.legend(handles=handles_c, fontsize=8, loc="upper right")
    ax3.set_title("Clustering K-Means — Patrones de Consumo", fontweight="bold")
    ax3.set_xlabel("Día del mes")
    ax3.set_ylabel("Litros / día")
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, max(consumo) * 1.2)

    # ------------------------------------------------------------------
    # PANEL 4 — Comparación de predicciones
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(dias, consumo, "o-", color="#7f8c8d",
             lw=1.3, markersize=4, label="Consumo real", alpha=0.75)
    ax4.plot(dias_pred_poly, vals_pred_poly, "s--", color="#8e44ad",
             lw=2, markersize=6, label="Regresión Polinómica")
    ax4.plot(dias_pred_nn, vals_pred_nn, "D--", color="#16a085",
             lw=2, markersize=6, label="Red Neuronal MLP")
    ax4.axhline(stats["promedio"], color="#2c3e50", linestyle=":",
                lw=1.2, alpha=0.5, label=f"Promedio {stats['promedio']:.0f} L")
    ax4.axvspan(max(dias) + 0.5, max(dias_pred_poly) + 0.5,
                alpha=0.07, color="purple", label="Zona predicción")
    ax4.set_title("Predicción: Polinómica vs Red Neuronal", fontweight="bold")
    ax4.set_xlabel("Día")
    ax4.set_ylabel("Litros / día")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    nombre_png = config["graficos"]["nombre_archivo"]
    plt.savefig(nombre_png, dpi=150, bbox_inches="tight")
    console.print(f"\n[green]📁 Gráfico guardado:[/green] [cyan]{nombre_png}[/cyan]")
    plt.show()

# =============================================================================
# 11. SALIDA ENRIQUECIDA CON RICH
# =============================================================================

def imprimir_encabezado(config, stats, costo):
    ed = config["edificio"]
    console.print(Panel(
        f"[bold white]Vivienda:[/bold white]  [cyan]{ed['nombre']}[/cyan]\n"
        f"[bold white]Ciudad  :[/bold white]  [cyan]{ed['ciudad']} — {ed['departamento']}, Perú[/cyan]\n"
        f"[bold white]Tipo    :[/bold white]  {ed['tipo']}  |  "
        f"[bold white]Pisos:[/bold white] {ed['pisos']}\n"
        f"[bold white]Consumo real del mes:[/bold white] "
        f"[yellow]{ed['consumo_mensual_real_m3']} m³  "
        f"({ed['consumo_mensual_real_litros']:,} litros)[/yellow]\n"
        f"[bold white]Proveedor:[/bold white] [magenta]{config['tarifa']['proveedor']}[/magenta]  |  "
        f"[bold white]Costo estimado:[/bold white] [green]S/. {costo:.2f}[/green]\n"
        f"[bold white]Promedio diario:[/bold white] [cyan]{stats['promedio']:.0f} L/día[/cyan]  |  "
        f"[bold white]Per cápita prom.:[/bold white] [cyan]{stats['per_capita_prom']:.1f} L/persona/día[/cyan]",
        title="[bold yellow]💧 SISTEMA INTELIGENTE DE OPTIMIZACIÓN DEL USO DE AGUA  v2.1[/bold yellow]",
        border_style="blue", padding=(1, 3)
    ))

def imprimir_estadisticas(stats, config):
    console.rule("[bold blue]📊 [1] ANÁLISIS ESTADÍSTICO[/bold blue]")
    ref = config["umbrales"]["litros_por_persona_dia"]

    t = Table(box=box.ROUNDED, header_style="bold cyan", min_width=56)
    t.add_column("Métrica",                 style="bold white", min_width=32)
    t.add_column("Valor",                   justify="right",    min_width=22)
    t.add_row("Días analizados",            "30 días")
    t.add_row("Consumo total real",
              f"[bold]{stats['total']:,.0f} L  =  {stats['total_m3']:.3f} m³[/bold]")
    t.add_row("Promedio diario",            f"[cyan]{stats['promedio']:.1f} L/día[/cyan]")
    t.add_row("Día de mayor consumo",       f"[red]{stats['maximo']:.0f} L[/red]")
    t.add_row("Día de menor consumo",       f"[green]{stats['minimo']:.0f} L[/green]")
    t.add_row("Desviación estándar",        f"{stats['desviacion']:.1f} L")
    t.add_row("─" * 30,                    "─" * 20)
    t.add_row("Per cápita promedio",
              f"[yellow]{stats['per_capita_prom']:.1f} L/persona/día[/yellow]")
    t.add_row("Per cápita máximo",          f"{stats['per_capita_max']:.1f} L/persona/día")
    t.add_row("Per cápita mínimo",          f"{stats['per_capita_min']:.1f} L/persona/día")
    t.add_row(f"Días sobre ref. ({ref} L/p)",
              f"[red]{stats['dias_sobre_ref']} días[/red]")
    console.print(t)

def imprimir_tarifa(stats, costo, config):
    console.rule("[bold yellow]💰 [2] DESGLOSE TARIFARIO — SEDACAJ CAJAMARCA[/bold yellow]")
    total_m3 = stats["total_m3"]
    rangos   = config["tarifa"]["rangos"]
    cargo    = config["tarifa"]["cargo_fijo_soles"]

    t = Table(box=box.SIMPLE_HEAVY, header_style="bold yellow")
    t.add_column("Tramo",           min_width=16)
    t.add_column("S/./m³",          justify="right")
    t.add_column("m³ consumidos",   justify="right")
    t.add_column("Subtotal S/.",    justify="right")

    t.add_row("Cargo fijo mensual", "—", "—", f"S/. {cargo:.2f}")
    for rango in rangos:
        desde  = rango["desde_m3"]
        hasta  = rango["hasta_m3"]
        precio = rango["precio_sol_por_m3"]
        if total_m3 > desde:
            m3_t     = round(min(total_m3, hasta) - desde, 3)
            subtotal = round(m3_t * precio, 2)
            t.add_row(f"{desde}–{hasta} m³",
                      f"S/. {precio:.3f}",
                      f"{m3_t:.3f} m³",
                      f"[yellow]S/. {subtotal:.2f}[/yellow]")
        if total_m3 <= hasta:
            break

    t.add_row("─" * 14, "─" * 10, "─" * 13, "─" * 13)
    t.add_row("[bold]TOTAL MES[/bold]", "",
              f"[bold]{total_m3:.3f} m³[/bold]",
              f"[bold green]S/. {costo:.2f}[/bold green]")
    console.print(t)

def imprimir_anomalias(anomalias_if, anomalias_umb, umbral, factor, stats):
    console.rule("[bold red]🔍 [3] DETECCIÓN DE ANOMALÍAS[/bold red]")

    console.print(f"\n  [bold]Isolation Forest[/bold] — {len(anomalias_if)} anomalía(s):")
    causas = {11: "Visita familiares + llenado tanque",
              23: "Fuga en caño + llenado nocturno"}
    if anomalias_if:
        t = Table(box=box.SIMPLE, header_style="bold red")
        t.add_column("Día",                 justify="center")
        t.add_column("Consumo",             justify="right")
        t.add_column("Score IA",            justify="right")
        t.add_column("Exceso vs promedio",  justify="right")
        t.add_column("Posible causa")
        for dia, valor, score in anomalias_if:
            exceso = valor - stats["promedio"]
            t.add_row(str(dia),
                      f"[red]{valor} L[/red]",
                      f"{score:.4f}",
                      f"+{exceso:.0f} L",
                      causas.get(dia, "Uso extraordinario"))
        console.print(t)

    console.print(f"\n  [bold]Umbral clásico[/bold] (×{factor} = {umbral:.0f} L) — "
                  f"{len(anomalias_umb)} día(s):")
    for dia, valor in anomalias_umb:
        console.print(f"    ⚠  Día [bold]{dia:02d}[/bold]: {valor} L")

def imprimir_franjas(res_franjas, umbral_noch):
    console.rule("[bold magenta]🕐 [4] ANÁLISIS POR FRANJA HORARIA[/bold magenta]")
    nombres = ["Madrugada 00–06h", "Mañana 06–12h", "Tarde 12–18h", "Noche 18–24h"]
    obs     = ["Bajo uso normal. Alerta si supera umbral (fuga/llenado tanque).",
               "Pico principal: ducha, desayuno, cocina.",
               "Almuerzo + lavado de vajilla.",
               "Cena + ducha nocturna."]
    total_p = sum(res_franjas["promedios_franja"])

    t = Table(box=box.ROUNDED, header_style="bold magenta")
    t.add_column("Franja",           min_width=20)
    t.add_column("Promedio (L)",     justify="right")
    t.add_column("% del día",        justify="right")
    t.add_column("Observación",      min_width=42)

    for nombre, prom, ob in zip(nombres, res_franjas["promedios_franja"], obs):
        pct   = prom / total_p * 100
        color = "red" if "Madrugada" in nombre and prom > umbral_noch else "white"
        t.add_row(nombre,
                  f"[{color}]{prom:.1f}[/{color}]",
                  f"{pct:.1f}%",
                  ob)
    console.print(t)

    fugas = res_franjas["fugas_nocturnas"]
    if fugas:
        console.print(f"\n  [bold red]🌙 Alertas nocturnas ({len(fugas)} noche(s)):[/bold red]")
        for dia, litros in fugas:
            console.print(
                f"    → Día [bold]{dia}[/bold]: [red]{litros} L[/red] en madrugada "
                f"(umbral: {umbral_noch} L) — posible fuga o llenado de tanque SEDACAJ"
            )
    else:
        console.print(
            f"\n  [green]🌙 Sin alertas nocturnas. "
            f"Consumo en madrugada dentro del umbral ({umbral_noch} L). ✔[/green]"
        )

def imprimir_clustering(cluster_info, dias, consumo_diario, n_clusters):
    console.rule("[bold green]🔵 [5] CLUSTERING K-MEANS[/bold green]")
    etiq    = cluster_info["etiquetas"]
    nombres = cluster_info["nombres"]

    t = Table(box=box.ROUNDED, header_style="bold green")
    t.add_column("Cluster",         min_width=22)
    t.add_column("Días del mes",    min_width=42)
    t.add_column("Prom. consumo",   justify="right")
    for k in sorted(set(etiq)):
        dias_k   = [dias[i]         for i, e in enumerate(etiq) if e == k]
        prom_k   = np.mean([consumo_diario[i] for i, e in enumerate(etiq) if e == k])
        nombre   = nombres.get(k, f"Grupo {k}")
        dias_str = ", ".join(str(d) for d in dias_k)
        t.add_row(f"[bold]{nombre}[/bold]", dias_str, f"{prom_k:.0f} L")
    console.print(t)

def imprimir_predicciones(dias_poly, vals_poly, dias_nn, vals_nn, grado, config):
    console.rule("[bold cyan]🤖 [6] PREDICCIONES CON IA[/bold cyan]")
    bajo = config["umbrales"]["consumo_bajo_litros"]
    alto = config["umbrales"]["consumo_alto_litros"]
    oc   = config["red_neuronal"]

    console.print(f"\n  [bold magenta]Regresión Polinómica[/bold magenta] — grado {grado}:")
    t1 = Table(box=box.SIMPLE, header_style="bold magenta")
    t1.add_column("Día futuro",        justify="center")
    t1.add_column("Litros estimados",  justify="right")
    t1.add_column("Categoría")
    for d, v in zip(dias_poly, vals_poly):
        cat   = clasificar_dia(v, bajo, alto)
        color = {"BAJO": "green", "NORMAL": "cyan", "ALTO": "red"}.get(cat, "white")
        t1.add_row(str(d), f"{v:.0f} L", f"[{color}]{cat}[/{color}]")
    console.print(t1)

    console.print(
        f"\n  [bold green]Red Neuronal MLP[/bold green] — "
        f"capas ({oc['neuronas_capa1']}→{oc['neuronas_capa2']} neuronas, {oc['epocas']} épocas):"
    )
    t2 = Table(box=box.SIMPLE, header_style="bold green")
    t2.add_column("Día futuro",        justify="center")
    t2.add_column("Litros estimados",  justify="right")
    t2.add_column("Categoría")
    for d, v in zip(dias_nn, vals_nn):
        cat   = clasificar_dia(v, bajo, alto)
        color = {"BAJO": "green", "NORMAL": "cyan", "ALTO": "red"}.get(cat, "white")
        t2.add_row(str(d), f"{v:.0f} L", f"[{color}]{cat}[/{color}]")
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

    factor      = config["umbrales"]["factor_anomalia"]
    bajo        = config["umbrales"]["consumo_bajo_litros"]
    alto        = config["umbrales"]["consumo_alto_litros"]
    ref_persona = config["umbrales"]["litros_por_persona_dia"]
    dias_fut    = config["prediccion"]["dias_futuros"]
    grado_poly  = config["prediccion"]["grado_polinomio"]
    n_clusters  = config["clustering"]["num_clusters"]
    umbral_noch = config["alertas"]["fuga_nocturna_umbral_litros"]

    # Estadísticas
    stats = analizar_datos(consumo_diario, ocupantes_por_dia, ref_persona)

    # Costo real SEDACAJ Cajamarca
    costo = calcular_costo_sedacaj(stats["total"], config)

    # Encabezado
    imprimir_encabezado(config, stats, costo)

    # [1] Estadísticas
    imprimir_estadisticas(stats, config)

    # [2] Tarifa desglosada
    imprimir_tarifa(stats, costo, config)

    # [3] Anomalías
    anomalias_if, _      = detectar_anomalias_if(consumo_diario, dias)
    anomalias_umb, umbral = detectar_anomalias_umbral(
        consumo_diario, dias, stats["promedio"], factor)
    imprimir_anomalias(anomalias_if, anomalias_umb, umbral, factor, stats)

    # [4] Franjas horarias
    res_franjas = analizar_franjas(consumo_horario, dias, umbral_noch)
    config["_fugas_nocturnas"] = res_franjas["fugas_nocturnas"]
    imprimir_franjas(res_franjas, umbral_noch)

    # [5] Clustering
    cluster_info = clustering_kmeans(consumo_horario, dias, n_clusters)
    imprimir_clustering(cluster_info, dias, consumo_diario, n_clusters)

    # [6] Predicciones
    _, dias_poly, vals_poly = predecir_polinomica(
        dias, consumo_diario, grado_poly, dias_fut)
    _, dias_nn, vals_nn = predecir_red_neuronal(
        dias, consumo_diario, ocupantes_por_dia,
        config["red_neuronal"], dias_fut)
    imprimir_predicciones(dias_poly, vals_poly, dias_nn, vals_nn, grado_poly, config)

    # [7] Recomendaciones
    clasificacion = clasificar_consumo(stats["promedio"], bajo, alto)
    recs = generar_recomendaciones(
        stats, clasificacion, anomalias_if,
        res_franjas["fugas_nocturnas"], config, costo)
    imprimir_recomendaciones(recs)

    # [8] Gráfico
    console.rule("[bold blue]📈 [8] GENERANDO GRÁFICO[/bold blue]")
    visualizar_completo(
        dias, consumo_diario, stats,
        umbral, anomalias_umb, anomalias_if,
        cluster_info, consumo_horario,
        dias_poly, vals_poly,
        dias_nn, vals_nn,
        config, costo
    )

    console.print(Panel(
        f"[bold green]✔ Análisis completado — Vivienda familiar, Cajamarca, Perú[/bold green]\n"
        f"Consumo real: {stats['total_m3']:.3f} m³  •  "
        f"Costo SEDACAJ: S/. {costo:.2f}  •  "
        f"Promedio: {stats['promedio']:.0f} L/día",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
