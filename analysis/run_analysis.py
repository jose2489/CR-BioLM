#!/usr/bin/env python
# analysis/run_analysis.py
#
# CR-BioLM — Script de análisis estadístico pre-registrado.
# Ejecutar DESPUÉS del run completo de 100 especies.
#
# Lee: experiment/runs/{exp_id}/results.csv
# Produce: analysis/results/{stats_all.json, report_results.md, figures/}
#
# Uso:
#   python analysis/run_analysis.py --exp-id EXP-20260416-005-botanico
#   python analysis/run_analysis.py --exp-id EXP-20260416-005-botanico --stratum A
#   python analysis/run_analysis.py --exp-id EXP-20260416-005-botanico --pilot   # piloto, N<10

import os
import sys
import json
import argparse
import random

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join("analysis", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TIERS       = ["T0", "T1", "T2", "T3"]
SEED        = 42
N_BOOTSTRAP = 10_000

random.seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(exp_dir: str) -> pd.DataFrame:
    csv = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"No se encontró results.csv en {exp_dir}")
    df = pd.read_csv(csv)
    print(f"[INFO] Cargados {len(df)} registros de {csv}")
    return df


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap percentile CI para la media de 'data'."""
    rng = np.random.default_rng(seed)
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return float("nan"), float("nan")
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return round(lo, 4), round(hi, 4)


def kendalls_w(friedman_stat, n_subjects, k_conditions):
    """Kendall's W como effect size del test de Friedman."""
    return friedman_stat / (k_conditions * (n_subjects - 1))


def wilcoxon_r(z_stat, n):
    """r = Z / sqrt(N) como effect size de Wilcoxon."""
    return abs(z_stat) / np.sqrt(n)


# ── Tests ─────────────────────────────────────────────────────────────────────

def friedman_test(df_pivot: pd.DataFrame, tiers=TIERS):
    """
    Test de Friedman sobre el DataFrame pivot (index=especie, columns=tier).
    Retorna dict con chi2, p, df, y Kendall's W.
    """
    available = [t for t in tiers if t in df_pivot.columns]
    if len(available) < 2:
        return None
    data_arrays = [df_pivot[t].dropna().values for t in available]
    # Alinear: solo filas completas
    aligned = pd.concat([df_pivot[t] for t in available], axis=1).dropna()
    if len(aligned) < 5:
        print(f"  [WARN] N={len(aligned)} — pocas muestras para Friedman.")
    arrays = [aligned[t].values for t in available]
    chi2, p = stats.friedmanchisquare(*arrays)
    n = len(aligned)
    k = len(available)
    w = kendalls_w(chi2, n, k)
    return {
        "test": "Friedman",
        "chi2": round(chi2, 4),
        "p": round(p, 6),
        "df": k - 1,
        "n_species": n,
        "k_tiers": k,
        "kendalls_W": round(w, 4),
        "tiers": available,
    }


def wilcoxon_pairwise(df_pivot: pd.DataFrame, tiers=TIERS, alpha=0.05):
    """
    Wilcoxon signed-rank para todos los pares de tiers, con corrección Bonferroni.
    Retorna lista de dicts.
    """
    from itertools import combinations
    available = [t for t in tiers if t in df_pivot.columns]
    pairs = list(combinations(available, 2))
    n_comparisons = len(pairs)
    alpha_corrected = alpha / n_comparisons
    results = []
    for t1, t2 in pairs:
        aligned = df_pivot[[t1, t2]].dropna()
        if len(aligned) < 5:
            continue
        d = aligned[t1].values - aligned[t2].values
        if np.all(d == 0):
            continue
        try:
            stat, p = stats.wilcoxon(aligned[t1].values, aligned[t2].values, alternative="two-sided")
        except Exception as e:
            print(f"  [WARN] Wilcoxon {t1}–{t2}: {e}")
            continue
        # Approximate Z from W statistic for effect size
        n = len(aligned)
        # Z approximation: W is the Wilcoxon test statistic
        z_approx = (stat - (n * (n + 1) / 4)) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        r = wilcoxon_r(z_approx, n)
        results.append({
            "pair": f"{t1}–{t2}",
            "tier_1": t1, "tier_2": t2,
            "W": round(stat, 2),
            "p_raw": round(p, 6),
            "p_bonferroni": round(min(p * n_comparisons, 1.0), 6),
            "significant": bool(p * n_comparisons < alpha),
            "alpha_bonferroni": round(alpha_corrected, 6),
            "r_effect_size": round(r, 4),
            "n": n,
            "mean_diff": round(float(aligned[t1].mean() - aligned[t2].mean()), 4),
        })
    return results


# ── Análisis principal ─────────────────────────────────────────────────────────

def analizar_estrato(df: pd.DataFrame, label: str, tiers=TIERS) -> dict:
    """
    Corre Friedman + Wilcoxon + bootstrap CIs para un subconjunto del DataFrame.
    df debe tener columnas: especie, tier, score_compuesto, modelo_generador.
    """
    print(f"\n  [ESTRATO {label}] N={len(df)} registros")
    resultados = {"label": label, "n_total": len(df)}

    # Medias y CIs por tier × generador
    tier_stats = {}
    for tier in tiers:
        sub = df[df["tier"] == tier]["score_compuesto"].dropna()
        if len(sub) == 0:
            continue
        lo, hi = bootstrap_ci(sub.values)
        tier_stats[tier] = {
            "mean": round(sub.mean(), 4),
            "std":  round(sub.std(), 4),
            "n":    len(sub),
            "ci95_lo": lo,
            "ci95_hi": hi,
        }
    resultados["tier_stats"] = tier_stats

    # Pivot para tests apareados (por especie)
    pivot = df.pivot_table(index="especie", columns="tier", values="score_compuesto", aggfunc="mean")

    # Friedman
    friedman = friedman_test(pivot, tiers)
    resultados["friedman"] = friedman

    # Wilcoxon post-hoc (solo si Friedman es significativo o siempre para descriptivo)
    wilcoxon = wilcoxon_pairwise(pivot, tiers)
    resultados["wilcoxon_pairwise"] = wilcoxon

    return resultados


def analizar_metricas(df: pd.DataFrame) -> dict:
    """Análisis descriptivo de cada métrica por tier."""
    metrics = {
        "M5_profundidad_analitica": "0-3",
        "M1_precision_geografica": "0-3",
        "M3_relevancia_respuesta": "0-3",
        "M2_precision_altitudinal": "0-2 (T3 only)",
        "M4_variable_climatica": "0-2 (T3 only)",
    }
    resultado = {}
    for metric, rango in metrics.items():
        if metric not in df.columns:
            continue
        resultado[metric] = {"rango": rango}
        for tier in TIERS:
            sub = df[df["tier"] == tier][metric]
            # Excluir "N/A" strings
            numeric = pd.to_numeric(sub, errors="coerce").dropna()
            if len(numeric) == 0:
                continue
            lo, hi = bootstrap_ci(numeric.values)
            resultado[metric][tier] = {
                "mean": round(numeric.mean(), 4),
                "std":  round(numeric.std(), 4),
                "n":    len(numeric),
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
    return resultado


def analizar_generadores(df: pd.DataFrame) -> dict:
    """Comparación Claude vs GPT-4o por tier via Wilcoxon pareado por especie."""
    resultado = {}
    modelos = df["modelo_generador"].dropna().unique()
    if len(modelos) < 2:
        return {"nota": "Solo un generador en los datos."}

    for tier in TIERS:
        sub = df[df["tier"] == tier]
        pivot = sub.pivot_table(index="especie", columns="modelo_generador",
                                values="score_compuesto", aggfunc="mean")
        if len(pivot.columns) < 2:
            continue
        cols = list(pivot.columns)
        aligned = pivot[cols].dropna()
        if len(aligned) < 5:
            continue
        try:
            stat, p = stats.wilcoxon(aligned[cols[0]], aligned[cols[1]], alternative="two-sided")
        except Exception:
            continue
        resultado[tier] = {
            "modelo_1": cols[0],
            "modelo_2": cols[1],
            "mean_1": round(aligned[cols[0]].mean(), 4),
            "mean_2": round(aligned[cols[1]].mean(), 4),
            "W": round(stat, 2),
            "p": round(p, 6),
            "n": len(aligned),
        }
    return resultado


# ── Figuras ───────────────────────────────────────────────────────────────────

def plot_tier_means(stats_by_stratum: dict, output_dir: str):
    """Gráfico de barras: score medio por tier × estrato con CIs bootstrap."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        labels_estrato = {"A": "Estrato A\n(tier_min=T1)", "B": "Estrato B\n(tier_min=T2)", "C": "Estrato C\n(tier_min=T3)"}
        colors = {"T0": "#9E9E9E", "T1": "#2196F3", "T2": "#4CAF50", "T3": "#FF9800"}

        for ax, (strat, data) in zip(axes, stats_by_stratum.items()):
            tier_stats = data.get("tier_stats", {})
            tiers = [t for t in TIERS if t in tier_stats]
            means = [tier_stats[t]["mean"] for t in tiers]
            cis_lo = [tier_stats[t]["ci95_lo"] for t in tiers]
            cis_hi = [tier_stats[t]["ci95_hi"] for t in tiers]
            yerr = [[m - lo for m, lo in zip(means, cis_lo)],
                    [hi - m  for m, hi in zip(means, cis_hi)]]
            bar_colors = [colors.get(t, "#666") for t in tiers]
            bars = ax.bar(tiers, means, color=bar_colors, yerr=yerr,
                          capsize=5, alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_title(labels_estrato.get(strat, f"Estrato {strat}"), fontsize=11)
            ax.set_xlabel("Tier")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score compuesto (0–1)" if ax is axes[0] else "")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle("CR-BioLM — Score compuesto por tier y estrato (bootstrap 95% CI)", fontsize=12)
        plt.tight_layout()
        out = os.path.join(output_dir, "tier_means_by_stratum.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIG] {out}")
    except Exception as e:
        print(f"  [WARN] No se pudo generar tier_means_by_stratum.png: {e}")


def plot_metric_heatmap(metric_stats: dict, output_dir: str):
    """Heatmap: media de cada métrica por tier."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        metrics = ["M5_profundidad_analitica", "M1_precision_geografica",
                   "M3_relevancia_respuesta", "M2_precision_altitudinal", "M4_variable_climatica"]
        metric_labels = {"M5_profundidad_analitica": "M5 Prof. Analítica (0-3)",
                         "M1_precision_geografica": "M1 Prec. Geográfica (0-3)",
                         "M3_relevancia_respuesta": "M3 Relevancia Resp. (0-3)",
                         "M2_precision_altitudinal": "M2 Altitud (0-2, T3)",
                         "M4_variable_climatica": "M4 Var. Climática (0-2, T3)"}
        data = []
        for m in metrics:
            row = []
            for tier in TIERS:
                val = metric_stats.get(m, {}).get(tier, {}).get("mean", float("nan"))
                row.append(val)
            data.append(row)

        df_heat = pd.DataFrame(data, index=[metric_labels.get(m, m) for m in metrics], columns=TIERS)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=3, ax=ax, linewidths=0.5)
        ax.set_title("CR-BioLM — Media de cada métrica por tier")
        plt.tight_layout()
        out = os.path.join(output_dir, "metric_heatmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIG] {out}")
    except Exception as e:
        print(f"  [WARN] No se pudo generar metric_heatmap.png: {e}")


def plot_m5_distribution(df: pd.DataFrame, output_dir: str):
    """Distribución de M5 por tier para verificar H4 (T0 M5 ≥ T1 en preguntas C)."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
        colors = {"T0": "#9E9E9E", "T1": "#2196F3", "T2": "#4CAF50", "T3": "#FF9800"}
        for ax, tier in zip(axes, TIERS):
            sub = pd.to_numeric(df[df["tier"] == tier].get("M5_profundidad_analitica", pd.Series()), errors="coerce").dropna()
            if len(sub) == 0:
                ax.set_title(f"{tier}\n(sin datos)")
                continue
            counts = sub.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color=colors.get(tier, "#666"),
                   alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{tier}\nn={len(sub)}, μ={sub.mean():.2f}")
            ax.set_xlabel("M5 Score")
            ax.set_ylabel("Frecuencia" if ax is axes[0] else "")
        fig.suptitle("CR-BioLM — Distribución de M5 (Profundidad Analítica) por tier", fontsize=12)
        plt.tight_layout()
        out = os.path.join(output_dir, "m5_distribution.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [FIG] {out}")
    except Exception as e:
        print(f"  [WARN] No se pudo generar m5_distribution.png: {e}")


# ── Reporte Markdown ──────────────────────────────────────────────────────────

def generar_reporte(all_stats: dict, df: pd.DataFrame, output_dir: str):
    """Genera report_results.md con todos los estadísticos."""
    lines = [
        "# CR-BioLM — Resultados del Análisis Estadístico",
        "",
        f"**Fecha de análisis**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**N especies**: {df['especie'].nunique()}",
        f"**N registros totales**: {len(df)}",
        f"**Tiers evaluados**: {', '.join(TIERS)}",
        "",
        "---",
        "",
        "## 1. Score compuesto por tier (todos los estratos)",
        "",
    ]
    global_pivot = df.pivot_table(index="especie", columns="tier", values="score_compuesto", aggfunc="mean")
    for tier in TIERS:
        if tier not in global_pivot.columns:
            continue
        col = global_pivot[tier].dropna()
        lo, hi = bootstrap_ci(col.values)
        lines.append(f"- **{tier}**: M={col.mean():.3f} (SD={col.std():.3f}, n={len(col)}, 95% CI [{lo:.3f}, {hi:.3f}])")
    lines.append("")

    lines += ["---", "", "## 2. Análisis por estrato", ""]
    for strat, data in all_stats.get("por_estrato", {}).items():
        lines.append(f"### Estrato {strat}")
        tier_stats = data.get("tier_stats", {})
        lines.append("")
        lines.append("| Tier | Media | SD | N | CI 95% Lo | CI 95% Hi |")
        lines.append("|------|-------|-----|---|-----------|-----------|")
        for tier in TIERS:
            if tier not in tier_stats:
                continue
            s = tier_stats[tier]
            lines.append(f"| {tier} | {s['mean']:.3f} | {s['std']:.3f} | {s['n']} | {s['ci95_lo']:.3f} | {s['ci95_hi']:.3f} |")
        lines.append("")

        frd = data.get("friedman")
        if frd:
            sig = "**Significativo**" if frd["p"] < 0.05 else "No significativo"
            lines.append(f"**Friedman**: χ²({frd['df']})={frd['chi2']}, p={frd['p']}, Kendall's W={frd['kendalls_W']} — {sig}")
            lines.append("")

        wlx = data.get("wilcoxon_pairwise", [])
        if wlx:
            lines.append("**Wilcoxon pairwise (Bonferroni)**:")
            lines.append("")
            lines.append("| Par | W | p (raw) | p (Bonf.) | Sig. | r (effect) |")
            lines.append("|-----|---|---------|-----------|------|------------|")
            for w in wlx:
                sig_mark = "✓" if w["significant"] else ""
                lines.append(f"| {w['pair']} | {w['W']} | {w['p_raw']} | {w['p_bonferroni']} | {sig_mark} | {w['r_effect_size']} |")
            lines.append("")

    lines += ["---", "", "## 3. Comparación de generadores", ""]
    gen_stats = all_stats.get("generadores", {})
    if gen_stats and "nota" not in gen_stats:
        lines.append("| Tier | Modelo 1 | Modelo 2 | M1 | M2 | W | p |")
        lines.append("|------|----------|----------|-----|-----|---|---|")
        for tier, gs in gen_stats.items():
            lines.append(f"| {tier} | {gs['modelo_1']} | {gs['modelo_2']} | {gs['mean_1']:.3f} | {gs['mean_2']:.3f} | {gs['W']} | {gs['p']} |")
        lines.append("")

    report_path = os.path.join(output_dir, "report_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [REPORT] {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CR-BioLM — Análisis estadístico pre-registrado")
    parser.add_argument("--exp-id", type=str, required=True,
                        help="ID del experimento (ej: EXP-20260416-005-botanico)")
    parser.add_argument("--stratum", type=str, default=None,
                        help="Filtrar por estrato A, B o C (default: todos)")
    parser.add_argument("--pilot", action="store_true",
                        help="Modo piloto: relajar chequeos de N mínimo")
    args = parser.parse_args()

    exp_dir = os.path.join("experiment", "runs", args.exp_id)
    if not os.path.isdir(exp_dir):
        print(f"[ERROR] No existe el directorio: {exp_dir}")
        sys.exit(1)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[ANÁLISIS] {args.exp_id}")
    print(f"{'='*60}")

    df = load_results(exp_dir)

    # Filtrar por estrato si se pide
    if args.stratum:
        if "stratum" in df.columns:
            df = df[df["stratum"] == args.stratum]
            print(f"[INFO] Filtrado a estrato={args.stratum}: {len(df)} registros")
        else:
            print("[WARN] Columna 'stratum' no encontrada en results.csv — ignorando filtro")

    # Determinar estratos disponibles
    if "stratum" in df.columns:
        estratos = sorted(df["stratum"].dropna().unique())
    else:
        estratos = ["todos"]
        df["stratum"] = "todos"

    all_stats = {}

    # Análisis por estrato
    stats_by_stratum = {}
    for strat in estratos:
        sub = df[df["stratum"] == strat] if strat != "todos" else df
        stats_by_stratum[strat] = analizar_estrato(sub, strat)
    all_stats["por_estrato"] = stats_by_stratum

    # Análisis de métricas individuales
    all_stats["metricas"] = analizar_metricas(df)

    # Comparación de generadores
    all_stats["generadores"] = analizar_generadores(df)

    # Guardar JSON
    stats_path = os.path.join(RESULTS_DIR, "stats_all.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[STATS] {stats_path}")

    # Figuras
    print("\n[FIGURAS]")
    plot_tier_means(stats_by_stratum, FIGURES_DIR)
    plot_metric_heatmap(all_stats["metricas"], FIGURES_DIR)
    plot_m5_distribution(df, FIGURES_DIR)

    # Reporte Markdown
    generar_reporte(all_stats, df, RESULTS_DIR)

    print(f"\n{'='*60}")
    print(f"[DONE] Resultados en: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
