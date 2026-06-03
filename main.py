# =============================================================================
# SISTEMA INTELIGENTE DE OPTIMIZACION DEL USO DE AGUA EN EDIFICACIONES
# Version 4.0 - Datos reales 4 anos: Vivienda de Alquiler, Cajamarca, Peru
# Habitantes: 5 personas (fijo) | Periodo: 2022-2025
# =============================================================================
# Instalacion:
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
    # Ano actual (2025)
    consumo_diario, metros_cubicos_mes, litros_por_dia_mes,
    litros_por_persona_mes, ocupantes_por_dia, dias,
    consumo_horario, meses, meses_abr,
    # Historico 4 anos (48 puntos)
    consumo_historico, anio_historico, mes_historico, indice_historico,
    datos_por_anio, PERSONAS, ANIOS,
)

console = Console()

# =============================================================================
# 0. CONFIGURACION
# =============================================================================

def cargar_config(ruta="config.json"):
    with open(ruta, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    console.print(f"[green]Configuracion cargada:[/green] [cyan]{ruta}[/cyan]")
    return cfg

# =============================================================================
# 1. TARIFA SEDACAJ - costo mensual por tramos
# =============================================================================

def calcular_costo_sedacaj(litros, config):
    m3     = litros / 1000
    rangos = config["tarifa"]["rangos"]
    costo  = config["tarifa"]["cargo_fijo_soles"]
    for r in rangos:
        if m3 > r["desde_m3"]:
            costo += (min(m3, r["hasta_m3"]) - r["desde_m3"]) * r["precio_sol_por_m3"]
        if m3 <= r["hasta_m3"]:
            break
    return round(costo, 2)

def costos_por_anio(config):
    resultado = {}
    for anio in ANIOS:
        resultado[anio] = [
            calcular_costo_sedacaj(l, config)
            for l in datos_por_anio[anio]["L"]
        ]
    return resultado

# =============================================================================
# 2. ESTADISTICAS - por ano y general
# =============================================================================

def estadisticas_anio(anio):
    datos = np.array(datos_por_anio[anio]["L"])
    return {
        "total"    : int(np.sum(datos)),
        "promedio" : float(np.mean(datos)),
        "maximo"   : int(np.max(datos)),
        "minimo"   : int(np.min(datos)),
        "desv"     : float(np.std(datos)),
        "mes_max"  : int(np.argmax(datos)),
        "mes_min"  : int(np.argmin(datos)),
    }

def estadisticas_historicas():
    h = np.array(consumo_historico)
    return {
        "promedio_global" : float(np.mean(h)),
        "maximo_global"   : int(np.max(h)),
        "minimo_global"   : int(np.min(h)),
        "total_4_anios"   : int(np.sum(h)),
        "tendencia"       : float(np.polyfit(range(len(h)), h, 1)[0]),
    }

# =============================================================================
# 3. DETECCION DE ANOMALIAS - Isolation Forest sobre los 48 puntos
# =============================================================================

def detectar_anomalias(config):
    X      = np.array(consumo_historico).reshape(-1, 1)
    modelo = IsolationForest(contamination=0.10, random_state=42)
    etiq   = modelo.fit_predict(X)
    factor = config["umbrales"]["factor_anomalia"]
    prom   = np.mean(consumo_historico)
    umbral = prom * factor

    anomalias = []
    for i, e in enumerate(etiq):
        if e == -1:
            anomalias.append({
                "anio"  : anio_historico[i],
                "mes"   : meses[mes_historico[i] - 1],
                "litros": consumo_historico[i],
                "tipo"  : "alto" if consumo_historico[i] > prom else "bajo",
            })
    return anomalias, umbral

# =============================================================================
# 4. CLASIFICACION DEL CONSUMO
# =============================================================================

def clasificar(valor, bajo, alto):
    if valor < bajo:    return "BAJO"
    elif valor <= alto: return "NORMAL"
    else:               return "ALTO"

# =============================================================================
# 5. PREDICCION - usando los 48 puntos historicos
#
# La logica de prediccion usa el promedio mensual de los 4 anos como base
# y luego aplica la tendencia general (subida o bajada) para ajustar.
# Esto es mas honesto y sensato que extrapolar una curva fuera del rango.
#
# Por ejemplo: para predecir Enero 2026 promedia los 4 eneros reales
# (2022, 2023, 2024, 2025) y aplica la tendencia de los 48 meses.
# =============================================================================

def predecir_proximos_meses(config):
    n_fut  = config["prediccion"]["meses_futuros"]
    grado  = config["prediccion"]["grado_polinomio"]

    # --- Modelo 1: promedio mensual + tendencia lineal ---
    # Para cada mes futuro (Ene, Feb, Mar 2026):
    #   base = promedio de ese mismo mes en los 4 anos anteriores
    #   ajuste = tendencia anual * avance de tiempo
    tendencia_anual = estadisticas_historicas()["tendencia"] * 12
    predicciones_prom = []
    meses_pred_nombres = []

    for i in range(n_fut):
        mes_idx = i          # 0=Ene, 1=Feb, 2=Mar
        valores_ese_mes = [
            datos_por_anio[a]["L"][mes_idx] for a in ANIOS
        ]
        base    = np.mean(valores_ese_mes)
        ajuste  = tendencia_anual * (i + 1) / 12
        pred    = max(base + ajuste, 0)
        predicciones_prom.append(round(pred))
        meses_pred_nombres.append(f"{meses[mes_idx]} 2026")

    # --- Modelo 2: regresion polinomica sobre los 48 puntos ---
    X      = np.array(indice_historico).reshape(-1, 1)
    y      = np.array(consumo_historico)
    modelo = make_pipeline(PolynomialFeatures(degree=grado), LinearRegression())
    modelo.fit(X, y)
    idx_fut   = np.arange(49, 49 + n_fut).reshape(-1, 1)
    vals_poly = np.clip(modelo.predict(idx_fut), 5000, 30000).tolist()

    # --- Modelo 3: red neuronal MLP ---
    # Entradas: (indice_lineal, mes_del_anio) -> consumo
    X_nn   = np.column_stack([indice_historico, mes_historico])
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_nn)

    sy     = StandardScaler()
    ys     = sy.fit_transform(y.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu",
        max_iter=1000, random_state=42,
        early_stopping=True, validation_fraction=0.15
    )
    mlp.fit(Xs, ys)

    idx_fut_lineal = list(range(49, 49 + n_fut))
    meses_fut_idx  = [i + 1 for i in range(n_fut)]   # 1=Ene, 2=Feb, 3=Mar
    Xf     = np.column_stack([idx_fut_lineal, meses_fut_idx])
    Xfs    = scaler.transform(Xf)
    vals_nn = np.clip(
        sy.inverse_transform(mlp.predict(Xfs).reshape(-1, 1)).ravel(),
        5000, 30000
    ).tolist()

    return {
        "nombres"      : meses_pred_nombres,
        "prom_hist"    : predicciones_prom,
        "polinomica"   : [round(v) for v in vals_poly],
        "red_neuronal" : [round(v) for v in vals_nn],
    }

# =============================================================================
# 6. CLUSTERING - agrupar los 48 meses por nivel de consumo
# =============================================================================

def clustering(config):
    X      = np.array(consumo_historico).reshape(-1, 1)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    km     = KMeans(n_clusters=config["clustering"]["num_clusters"],
                    random_state=42, n_init=10)
    etiq   = km.fit_predict(Xs)

    # Ordenar clusters: 0=bajo, 1=medio, 2=alto segun su promedio
    centros = km.cluster_centers_.ravel()
    orden   = np.argsort(centros)
    mapa    = {orden[i]: i for i in range(len(orden))}
    etiq    = [mapa[e] for e in etiq]

    nombres_cluster = ["Consumo bajo", "Consumo normal", "Consumo alto"]
    return etiq, nombres_cluster

# =============================================================================
# 7. RECOMENDACIONES
# =============================================================================

def generar_recomendaciones(stats_actual, anomalias, pred, costos, config):
    bajo = config["umbrales"]["consumo_bajo_litros"]
    alto = config["umbrales"]["consumo_alto_litros"]
    recs = []

    # Clasificacion del consumo 2025
    cls = clasificar(stats_actual["promedio"], bajo, alto)
    if cls == "BAJO":
        recs.append(("green", "Consumo 2025 BAJO. Excelente uso del agua en la vivienda."))
    elif cls == "NORMAL":
        recs.append(("cyan", "Consumo 2025 dentro del rango NORMAL para 5 personas."))
        recs.append(("cyan", "  -> Instale reductores de caudal para mejorar aun mas."))
    else:
        recs.append(("red", "Consumo 2025 ALTO. Supera el rango esperado para 5 personas."))
        recs.append(("red", f"  -> Mes mas alto: {meses[stats_actual['mes_max']]} con {stats_actual['maximo']:,} L."))

    # Tendencia historica
    tend = estadisticas_historicas()["tendencia"]
    if tend > 0:
        recs.append(("yellow", f"Tendencia: el consumo sube ~{abs(tend*12):,.0f} L/anio en promedio. Vigilar."))
    elif tend < 0:
        recs.append(("green", f"Tendencia: el consumo baja ~{abs(tend*12):,.0f} L/anio. Buena senal."))
    else:
        recs.append(("cyan", "Tendencia estable en los 4 anos analizados."))

    # Anomalias
    if anomalias:
        recs.append(("red", f"Se detectaron {len(anomalias)} mes(es) con consumo inusual en los 4 anos:"))
        for a in anomalias:
            recs.append(("red", f"  -> {a['mes']} {a['anio']}: {a['litros']:,} L ({a['tipo']})"))
    else:
        recs.append(("green", "Sin meses con consumo inusual detectado en los 4 anos."))

    # Costo
    costo_2025 = sum(costos[2025])
    recs.append(("yellow", f"Costo anual estimado SEDACAJ 2025: S/. {costo_2025:.2f}"))
    recs.append(("yellow", f"  -> Promedio mensual: S/. {costo_2025/12:.2f}/mes"))

    # Prediccion
    recs.append(("cyan", "Estimacion proximos 3 meses (Ene-Mar 2026) basada en 4 anos reales:"))
    for nombre, val in zip(pred["nombres"], pred["prom_hist"]):
        cls_p = clasificar(val, bajo, alto)
        recs.append(("cyan", f"  -> {nombre}: ~{val:,} L ({cls_p})"))

    return recs

# =============================================================================
# 8. VISUALIZACION - 2 graficos claros para cualquier publico
# =============================================================================

def visualizar(stats_actual, anomalias, pred, costos, config):
    bajo     = config["umbrales"]["consumo_bajo_litros"]
    alto     = config["umbrales"]["consumo_alto_litros"]
    nombre   = config["edificio"]["nombre"]
    ciudad   = config["edificio"]["ciudad"]
    personas = config["edificio"]["personas"]
    costo_25 = sum(costos[2025])

    meses_pred_abr = ["Ene 26", "Feb 26", "Mar 26"]
    anomalias_set  = {(a["anio"], a["mes"]) for a in anomalias}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"{nombre}  -  {ciudad}, Peru  |  {personas} personas\n"
        f"Consumo 2025: {stats_actual['total']:,} L  "
        f"({stats_actual['total']//1000} m3)  |  "
        f"Costo anual estimado: S/. {costo_25:.2f}",
        fontsize=12, fontweight="bold"
    )

    # ------------------------------------------------------------------
    # GRAFICO 1 - Cuanta agua se uso cada mes en 2025?
    # Barras por mes con colores intuitivos y valor encima
    # ------------------------------------------------------------------
    colores_barra = []
    for i, (v, mes_n) in enumerate(zip(consumo_diario, meses)):
        if (2025, mes_n) in anomalias_set:
            colores_barra.append("#e74c3c")   # rojo: mes inusual
        elif v > alto:
            colores_barra.append("#e67e22")   # naranja: consumo alto
        elif v < bajo:
            colores_barra.append("#2ecc71")   # verde: consumo bajo
        else:
            colores_barra.append("#3498db")   # azul: consumo normal

    barras = ax1.bar(meses_abr, consumo_diario, color=colores_barra,
                     edgecolor="white", linewidth=0.8, zorder=2)

    # Valor en litros encima de cada barra
    for barra, valor, mes_n in zip(barras, consumo_diario, meses):
        color_txt = "#c0392b" if (2025, mes_n) in anomalias_set else "#2c3e50"
        ax1.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + max(consumo_diario) * 0.015,
            f"{valor//1000}k L",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color_txt
        )
        if (2025, mes_n) in anomalias_set:
            ax1.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + max(consumo_diario) * 0.08,
                "Mes inusual",
                ha="center", fontsize=7.5, color="#c0392b", fontstyle="italic"
            )

    # Linea del promedio
    ax1.axhline(stats_actual["promedio"], color="#2c3e50", linestyle="--",
                lw=1.8, zorder=3,
                label=f"Promedio 2025: {stats_actual['promedio']:,.0f} L/mes")

    # Leyenda en lenguaje simple
    leyenda = [
        mpatches.Patch(color="#2ecc71", label="Consumo bajo"),
        mpatches.Patch(color="#3498db", label="Consumo normal"),
        mpatches.Patch(color="#e67e22", label="Consumo alto"),
        mpatches.Patch(color="#e74c3c", label="Mes inusual - revisar"),
    ]
    ax1.legend(handles=leyenda, fontsize=8.5, loc="upper right", framealpha=0.9)
    hdl, lbl = ax1.get_legend_handles_labels()
    ax1.legend(handles=leyenda + hdl, labels=[p.get_label() for p in leyenda] + lbl,
               fontsize=8, loc="upper right", framealpha=0.9)

    ax1.set_title("Cuanta agua se uso cada mes en 2025?",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Mes del ano", fontsize=10)
    ax1.set_ylabel("Litros de agua", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax1.set_ylim(0, max(consumo_diario) * 1.30)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.set_axisbelow(True)

    # ------------------------------------------------------------------
    # GRAFICO 2 - Comparacion de los 4 anos + estimacion 2026
    # Una linea por cada ano para ver la evolucion real,
    # y una zona sombreada para los 3 meses estimados de 2026.
    # ------------------------------------------------------------------
    colores_anio = {2022: "#95a5a6", 2023: "#3498db", 2024: "#e67e22", 2025: "#2c3e50"}
    estilos      = {2022: "--",       2023: "--",       2024: "--",       2025: "-"}
    grosores     = {2022: 1.2,        2023: 1.2,        2024: 1.2,        2025: 2.2}

    for anio in ANIOS:
        y_vals = datos_por_anio[anio]["L"]
        ax2.plot(meses_abr, y_vals,
                 marker="o", markersize=4 if anio != 2025 else 6,
                 color=colores_anio[anio],
                 linestyle=estilos[anio],
                 linewidth=grosores[anio],
                 label=str(anio),
                 zorder=3 if anio == 2025 else 2,
                 alpha=0.7 if anio != 2025 else 1.0)

    # Estimacion 2026 - usando promedio historico (mas confiable)
    # Se conecta desde el ultimo punto real de 2025 (Diciembre)
    x_conexion  = [len(meses_abr) - 1] + [len(meses_abr), len(meses_abr)+1, len(meses_abr)+2]
    y_conexion  = [consumo_diario[-1]] + pred["prom_hist"]

    ax2.plot(x_conexion, y_conexion,
             "o--", color="#8e44ad", lw=2, markersize=7,
             label="Estimacion Ene-Mar 2026", zorder=4)

    # Valor encima de cada punto estimado
    for xi, vp, nombre_m in zip(
        x_conexion[1:], pred["prom_hist"], meses_pred_abr
    ):
        ax2.text(xi, vp + max(consumo_historico) * 0.04,
                 f"{vp//1000}k L",
                 ha="center", fontsize=9, fontweight="bold", color="#8e44ad")

    # Zona sombreada de prediccion
    ax2.axvspan(len(meses_abr) - 0.5, len(meses_abr) + 2.5,
                alpha=0.07, color="#8e44ad")
    ax2.text(len(meses_abr) + 1, max(consumo_historico) * 0.95,
             "Estimacion\n2026", ha="center", fontsize=8,
             color="#8e44ad", fontstyle="italic")

    # Eje X con todos los meses + prediccion
    todos_x = list(range(len(meses_abr) + 3))
    ax2.set_xticks(todos_x)
    ax2.set_xticklabels(meses_abr + meses_pred_abr, fontsize=8)

    ax2.set_title("Consumo por ano (2022-2025) y estimacion 2026",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Mes", fontsize=10)
    ax2.set_ylabel("Litros de agua", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax2.set_ylim(0, max(consumo_historico) * 1.30)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    nombre_png = config["graficos"]["nombre_archivo"]
    plt.savefig(nombre_png, dpi=150, bbox_inches="tight")
    console.print(f"\n[green]Grafico guardado:[/green] [cyan]{nombre_png}[/cyan]")
    plt.show()

# =============================================================================
# 9. SALIDA EN CONSOLA CON RICH
# =============================================================================

def imprimir_encabezado(config, stats_actual, costos):
    ed = config["edificio"]
    console.print(Panel(
        f"[bold white]Edificacion :[/bold white] [cyan]{ed['nombre']}[/cyan]\n"
        f"[bold white]Ciudad      :[/bold white] [cyan]{ed['ciudad']}, Peru[/cyan]\n"
        f"[bold white]Habitantes  :[/bold white] {ed['personas']} personas (fijo)\n"
        f"[bold white]Periodo     :[/bold white] [yellow]2022 - 2025  (48 meses reales)[/yellow]\n"
        f"[bold white]Consumo 2025:[/bold white] "
        f"[yellow]{stats_actual['total']:,} L  ({stats_actual['total']//1000} m3)[/yellow]  |  "
        f"[bold white]Costo anual:[/bold white] [green]S/. {sum(costos[2025]):.2f}[/green]",
        title="[bold yellow]SISTEMA DE OPTIMIZACION DEL USO DE AGUA - v4.0[/bold yellow]",
        border_style="blue", padding=(1, 3)
    ))

def imprimir_tabla_4_anios(costos):
    console.rule("[bold blue]TABLA DE CONSUMO - 4 ANOS (litros por mes)[/bold blue]")
    t = Table(box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Mes",         style="bold white", min_width=12)
    t.add_column("2022",        justify="right", min_width=9)
    t.add_column("2023",        justify="right", min_width=9)
    t.add_column("2024",        justify="right", min_width=9)
    t.add_column("2025",        justify="right", min_width=9)
    t.add_column("Promedio",    justify="right", min_width=10)

    for i, mes_n in enumerate(meses):
        vals   = [datos_por_anio[a]["L"][i] for a in ANIOS]
        prom_m = int(np.mean(vals))
        maxi   = max(vals)
        t.add_row(
            mes_n,
            *[f"[red]{v:,}[/red]" if v == maxi else f"{v:,}" for v in vals],
            f"[yellow]{prom_m:,}[/yellow]",
        )

    # Fila de totales
    tots = [sum(datos_por_anio[a]["L"]) for a in ANIOS]
    t.add_row(
        "[bold]TOTAL[/bold]",
        *[f"[bold]{t_:,}[/bold]" for t_ in tots],
        f"[bold yellow]{int(np.mean(tots)):,}[/bold yellow]",
    )
    console.print(t)

def imprimir_estadisticas_comparadas(costos):
    console.rule("[bold blue]ESTADISTICAS COMPARADAS POR ANO[/bold blue]")
    t = Table(box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Ano",         justify="center", min_width=6)
    t.add_column("Total L",     justify="right",  min_width=10)
    t.add_column("Prom/mes",    justify="right",  min_width=10)
    t.add_column("Mes max",     justify="center", min_width=15)
    t.add_column("Mes min",     justify="center", min_width=15)
    t.add_column("Costo S/.",   justify="right",  min_width=10)

    for anio in ANIOS:
        s = estadisticas_anio(anio)
        t.add_row(
            str(anio),
            f"{s['total']:,}",
            f"{s['promedio']:,.0f}",
            f"{meses[s['mes_max']]} ({s['maximo']:,} L)",
            f"{meses[s['mes_min']]} ({s['minimo']:,} L)",
            f"S/. {sum(costos[anio]):.2f}",
        )

    hist = estadisticas_historicas()
    tend_str = (f"+{hist['tendencia']*12:,.0f} L/anio"
                if hist["tendencia"] >= 0
                else f"{hist['tendencia']*12:,.0f} L/anio")
    console.print(t)
    console.print(f"  [yellow]Tendencia 4 anos: {tend_str}[/yellow]  |  "
                  f"[cyan]Total historico: {hist['total_4_anios']:,} L "
                  f"({hist['total_4_anios']//1000} m3)[/cyan]")

def imprimir_anomalias(anomalias, umbral):
    console.rule("[bold red]DETECCION DE MESES INUSUALES (Isolation Forest)[/bold red]")
    if not anomalias:
        console.print("[green]Sin meses inusuales en los 4 anos analizados.[/green]")
        return
    t = Table(box=box.SIMPLE, header_style="bold red")
    t.add_column("Ano",    justify="center")
    t.add_column("Mes",    justify="center")
    t.add_column("Litros", justify="right")
    t.add_column("Tipo",   justify="center")
    for a in anomalias:
        color = "red" if a["tipo"] == "alto" else "cyan"
        t.add_row(str(a["anio"]), a["mes"],
                  f"[{color}]{a['litros']:,}[/{color}]", a["tipo"].upper())
    console.print(t)

def imprimir_prediccion(pred, config):
    console.rule("[bold cyan]ESTIMACION - Enero, Febrero y Marzo 2026[/bold cyan]")
    bajo = config["umbrales"]["consumo_bajo_litros"]
    alto = config["umbrales"]["consumo_alto_litros"]

    console.print(
        "\n  [dim]Metodo: promedio de los mismos meses en los 4 anos reales "
        "+ tendencia historica.[/dim]\n"
        "  [dim]Este metodo es el mas confiable porque se basa en patrones "
        "reales observados.[/dim]\n"
    )
    t = Table(box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Mes estimado",    min_width=16)
    t.add_column("Litros est.",     justify="right")
    t.add_column("m3 est.",         justify="right")
    t.add_column("Nivel")
    t.add_column("Base del calculo")

    for i, (nombre_m, val) in enumerate(zip(pred["nombres"], pred["prom_hist"])):
        cls   = clasificar(val, bajo, alto)
        color = {"BAJO": "green", "NORMAL": "cyan", "ALTO": "red"}[cls]
        # Mostrar los valores reales de ese mes en cada ano
        vals_reales = [datos_por_anio[a]["L"][i] for a in ANIOS]
        base_str = "  |  ".join(
            f"{a}: {v//1000}k" for a, v in zip(ANIOS, vals_reales)
        )
        t.add_row(
            nombre_m,
            f"{val:,} L",
            f"{val/1000:.1f}",
            f"[{color}]{cls}[/{color}]",
            base_str,
        )
    console.print(t)

def imprimir_recomendaciones(recs):
    console.rule("[bold yellow]RECOMENDACIONES[/bold yellow]")
    for color, msg in recs:
        console.print(f"  [{color}]{msg}[/{color}]")

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    config = cargar_config("config.json")

    # Calcular costos para los 4 anos
    costos = costos_por_anio(config)

    # Estadisticas del ano actual
    stats_actual = estadisticas_anio(2025)

    # Encabezado
    imprimir_encabezado(config, stats_actual, costos)

    # Tabla comparativa 4 anos
    imprimir_tabla_4_anios(costos)

    # Estadisticas comparadas
    imprimir_estadisticas_comparadas(costos)

    # Anomalias sobre los 48 puntos
    anomalias, umbral = detectar_anomalias(config)
    imprimir_anomalias(anomalias, umbral)

    # Prediccion proximos 3 meses
    pred = predecir_proximos_meses(config)
    imprimir_prediccion(pred, config)

    # Recomendaciones
    recs = generar_recomendaciones(stats_actual, anomalias, pred, costos, config)
    imprimir_recomendaciones(recs)

    # Graficos
    console.rule("[bold blue]GENERANDO GRAFICOS[/bold blue]")
    visualizar(stats_actual, anomalias, pred, costos, config)

    console.print(Panel(
        f"[bold green]Analisis completado - {config['edificio']['nombre']}[/bold green]\n"
        f"48 meses analizados (2022-2025)  |  "
        f"Costo 2025: S/. {sum(costos[2025]):.2f}",
        border_style="green"
    ))


if __name__ == "__main__":
    main()