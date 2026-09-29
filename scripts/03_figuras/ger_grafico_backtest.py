# -*- coding: utf-8 -*-
"""
VALIDAÇÃO DO MODELO GFOM - 2021 (WALK-FORWARD)

- Monta a base mensal SIN
- Faz backtest temporal dentro de 2021
- Plota GFOM Real vs GFOM Prevista
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)


BASE_DIR = Path(
    r"./dados_brutos"  # Aponte para a pasta local com os dados brutos do ONS (ver secao "Dados" do README) — caminho original removido por conter identificador pessoal
)

GT_PATH = BASE_DIR / "gt_historico.csv"
EAR_PATH = BASE_DIR / "ear_historico.csv"
CREF_PATH = BASE_DIR / "cref_2021_2026.xlsx"
OUT_DIR = BASE_DIR / "graficos_validacao"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    except AttributeError:
        pass
    return dt.dt.to_period("M").dt.to_timestamp()


def normalizar_subsistema(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()

    mapa = {
        "SUDESTE": "SE/CO",
        "SE": "SE/CO",
        "SE/CO": "SE/CO",
        "SECO": "SE/CO",
        "SUDESTE/CO": "SE/CO",
        "SUDESTE / CO": "SE/CO",
        "SUDESTE/CENTRO-OESTE": "SE/CO",
        "SUDESTE / CENTRO-OESTE": "SE/CO",
        "SUDESTE-CENTRO-OESTE": "SE/CO",
        "CENTRO-OESTE": "SE/CO",
        "SUL": "S",
        "S": "S",
        "NORDESTE": "NE",
        "NE": "NE",
        "NORTE": "N",
        "N": "N",
    }
    return s.replace(mapa)


def load_gfom() -> pd.DataFrame:
    df = pd.read_csv(GT_PATH, sep=";")

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data_hora"])
    df["gfom"] = pd.to_numeric(df["val_verifgarantiaenergetica"], errors="coerce")

    df = df[df["mes_dt"].dt.year.between(2021, 2025)].copy()

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom=("gfom", "mean"))
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )
    return out


def load_ear() -> pd.DataFrame:
    df = pd.read_csv(EAR_PATH, sep=";")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data"])

    df = df.sort_values(["subsistema", "data"]).copy()
    df = df.groupby(["subsistema", "mes_dt"], as_index=False).first()

    df["ear"] = pd.to_numeric(df["ear_fim_per"], errors="coerce")
    df = df[df["mes_dt"].dt.year.between(2021, 2025)].copy()

    return df[["mes_dt", "subsistema", "ear"]]


def load_cref() -> pd.DataFrame:
    df = pd.read_excel(CREF_PATH, sheet_name="Consolidado")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])
    df["cref"] = pd.to_numeric(df["curva_amarela"], errors="coerce")

    out = (
        df[["mes_dt", "cref"]]
        .dropna()
        .drop_duplicates()
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return out[out["mes_dt"].dt.year.between(2021, 2025)].copy()


def build_base_sin() -> pd.DataFrame:
    gfom = load_gfom()
    ear = load_ear()
    cref = load_cref()

    base = (
        gfom.merge(ear, on=["mes_dt", "subsistema"], how="inner")
        .merge(cref, on="mes_dt", how="left")
        .dropna(subset=["gfom", "ear", "cref"])
        .copy()
    )

    base_sin = (
        base.groupby("mes_dt", as_index=False)
        .agg(
            gfom=("gfom", "sum"),
            ear=("ear", "mean"),
            cref=("cref", "mean"),
        )
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    base_sin["gap"] = (base_sin["cref"] - base_sin["ear"]).clip(lower=0.0)
    base_sin["gatilho"] = (base_sin["ear"] <= base_sin["cref"]).astype(int)

    base_sin["mes_num"] = base_sin["mes_dt"].dt.month
    base_sin["mes_sin"] = np.sin(2 * np.pi * base_sin["mes_num"] / 12.0)
    base_sin["mes_cos"] = np.cos(2 * np.pi * base_sin["mes_num"] / 12.0)

    return base_sin


def get_features() -> list[str]:
    return ["ear", "cref", "gap", "gatilho", "mes_num", "mes_sin", "mes_cos"]


def run_backtest_2021_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("mes_dt").reset_index(drop=True).copy()

    df_2021 = df[df["mes_dt"].dt.year == 2021].copy()
    if df_2021.empty:
        raise ValueError("Não há dados de 2021 na base.")

    meses_2021 = sorted(df_2021["mes_dt"].unique())
    results = []

    features = get_features()

    for mes_ref in meses_2021:
        treino = df[df["mes_dt"] < mes_ref].copy()
        teste = df[df["mes_dt"] == mes_ref].copy()

        if treino.empty or teste.empty:
            continue

        treino = treino.dropna(subset=features + ["gfom"]).copy()
        teste = teste.dropna(subset=features + ["gfom"]).copy()

        if treino.empty or teste.empty:
            continue

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=8,
            min_samples_leaf=1,
            n_jobs=-1,
        )

        model.fit(treino[features], treino["gfom"])
        teste["gfom_prev"] = model.predict(teste[features]).clip(min=0.0)

        results.append(teste[["mes_dt", "gfom", "gfom_prev", "ear", "cref", "gap", "gatilho"]])

    if not results:
        raise ValueError("Nenhuma previsão foi gerada no walk-forward de 2021.")

    out = pd.concat(results, ignore_index=True).sort_values("mes_dt").reset_index(drop=True)
    return out


def calc_metrics(df_bt: pd.DataFrame) -> dict[str, float]:
    y_true = df_bt["gfom"].to_numpy(dtype=float)
    y_pred = df_bt["gfom_prev"].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if len(np.unique(y_true)) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    return {"mae": mae, "rmse": rmse, "r2": r2}

def calc_classification_metrics(
    df_bt: pd.DataFrame,
    threshold: float = 50.0
) -> dict[str, float]:
    """
    Converte GFOM em evento binário e calcula métricas de classificação.

    Regra:
    - 1 = evento relevante de GFOM
    - 0 = não evento
    """
    y_true = (df_bt["gfom"] >= threshold).astype(int)
    y_pred = (df_bt["gfom_prev"] >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def plot_backtest_2021(df_bt: pd.DataFrame) -> None:
    metrics = calc_metrics(df_bt)
    class_metrics = calc_classification_metrics(df_bt, threshold=50.0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    })

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)

    ax.plot(
        df_bt["mes_dt"],
        df_bt["gfom"],
        linewidth=3,
        label="Observed GFOM",
        zorder=3,
    )

    ax.plot(
        df_bt["mes_dt"],
        df_bt["gfom_prev"],
        linewidth=3,
        linestyle="--",
        label= "Predicted GFOM",
        zorder=3,
    )

    ax.set_ylabel("GFOM (MW average)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.grid(False, axis="x")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
    )

    txt = (
        f"MAE = {metrics['mae']:.1f} | RMSE = {metrics['rmse']:.1f} | R² = {metrics['r2']:.2f}\n"
        f"Precision = {class_metrics['precision']:.2f} | "
        f"Recall = {class_metrics['recall']:.2f} | "
        f"F1-score = {class_metrics['f1']:.2f}"
    )
    ax.text(
        0.01,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            alpha=0.85,
            edgecolor="none"
        ),
    )

    out_dir = ensure_dir(OUT_DIR)
    png_path = out_dir / "validacao_gfom_2021_walkforward.png"
    pdf_path = png_path.with_suffix(".pdf")
    xlsx_path = out_dir / "validacao_gfom_2021_walkforward.xlsx"

    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.show()

    df_bt.to_excel(xlsx_path, index=False)

    print(f"PNG salvo em: {png_path}")
    print(f"PDF salvo em: {pdf_path}")
    print(f"Base salva em: {xlsx_path}")
    print(txt)



def main() -> None:
    df = build_base_sin()
    print("Base SIN:")
    print(df.head())
    print(df.tail())

    df_bt = run_backtest_2021_walkforward(df)

    metrics = calc_metrics(df_bt)
    class_metrics = calc_classification_metrics(df_bt, threshold=50.0)

    print("\nMétricas de regressão:")
    print(metrics)

    print("\nMétricas de classificação (threshold = 50 MWmed):")
    print(class_metrics)

    print("\nBacktest 2021:")
    print(df_bt)

    plot_backtest_2021(df_bt)


if __name__ == "__main__":
    main()