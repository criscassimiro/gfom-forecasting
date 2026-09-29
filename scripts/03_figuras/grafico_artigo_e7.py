# -*- coding: utf-8 -*-
"""
GRÁFICOS PARA ARTIGO - ETAPA E7

Figuras geradas:
1. GFOM histórica total + EAR + CRef
2. GFOM histórica por subsistema

Regras:
- GFOM histórica agregada por média mensal
- EAR histórica capturada no primeiro instante do mês
- CRef mensal
- Período: jan/2021 a dez/2025
- Exportação em PNG e PDF
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


BASE_DIR = Path(
    r"./dados_brutos"  # Aponte para a pasta local com os dados brutos do ONS (ver secao "Dados" do README) — caminho original removido por conter identificador pessoal
)


@dataclass
class Config:
    base_dir: Path = BASE_DIR
    gt_historico_csv: Path = BASE_DIR / "gt_historico.csv"
    ear_historico_csv: Path = BASE_DIR / "ear_historico.csv"
    cref_xlsx: Path = BASE_DIR / "cref_2021_2026.xlsx"
    out_dir: Path = BASE_DIR / "graficos"


CFG = Config()


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


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
    })


def save_figure(fig: plt.Figure, caminho_saida: Path) -> None:
    fig.savefig(
        caminho_saida,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        caminho_saida.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white"
    )


def format_time_axis(
    ax: plt.Axes,
    start: str,
    end: str,
    month_interval: int = 6,
    rotation: int = 35
) -> None:
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", fontsize=18)


def format_primary_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=18)


def format_secondary_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="y", which="major", labelsize=18)


def add_top_legend(ax: plt.Axes, handles, labels, ncol: int = 3) -> None:
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=ncol,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.3,
        handletextpad=0.6,
        borderaxespad=0.0,
        fontsize=18,
    )


#def load_cref_historico(cfg: Config) -> pd.DataFrame:
    #df = pd.read_excel(cfg.cref_xlsx, sheet_name="Consolidado")

    #df["data"] = pd.to_datetime(df["data"], errors="coerce")
    #df["mes_dt"] = to_month_start(df["data"])
    #df["cref_amarela"] = pd.to_numeric(df["curva_amarela"], errors="coerce")

   # df = (
       # df[["mes_dt", "cref_amarela"]]
        #.dropna()
        #.drop_duplicates()
        #.sort_values("mes_dt")
        #.reset_index(drop=True)
   # )

   # df = df[df["mes_dt"].dt.year.between(2021, 2025)].copy()
   # return df


#def load_ear_historico(cfg: Config) -> pd.DataFrame:
   # df = pd.read_csv(cfg.ear_historico_csv, sep=";")

   # df["data"] = pd.to_datetime(df["data"], errors="coerce")
    #df["subsistema"] = normalizar_subsistema(df["subsistema"])
   # df["mes_dt"] = to_month_start(df["data"])

  #  df = df.sort_values(["subsistema", "data"]).copy()

  #  df = (
       # df.groupby(["subsistema", "mes_dt"], as_index=False)
       # .first()
 #   )

   # df["ear_pct_sub"] = pd.to_numeric(df["ear_fim_per"], errors="coerce")

   # df = df[["mes_dt", "subsistema", "ear_pct_sub"]].copy()
   # df = df[df["mes_dt"].dt.year.between(2021, 2025)].reset_index(drop=True)

   # return df


def load_gfom_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";")

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data_hora"])
    df["val_verifgarantiaenergetica"] = pd.to_numeric(
        df["val_verifgarantiaenergetica"], errors="coerce"
    )

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom_hist_mwmed=("val_verifgarantiaenergetica", "mean"))
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year.between(2021, 2025)].copy()
    return out


def build_plot_base(
    cfg: Config,
    subsistemas_gfom: list[str] | None = None,
    subsistema_ear: str = "SE/CO",
) -> pd.DataFrame:
    gfom = load_gfom_historico(cfg)
    #ear = load_ear_historico(cfg)
    #cref = load_cref_historico(cfg)

    if subsistemas_gfom is not None:
        gfom = gfom[gfom["subsistema"].isin(subsistemas_gfom)].copy()

    gfom_total = (
        gfom.groupby("mes_dt", as_index=False)
        .agg(gfom_hist_mwmed=("gfom_hist_mwmed", "sum"))
        .sort_values("mes_dt")
    )

    #ear_sub = (
        #ear[ear["subsistema"] == subsistema_ear]
        #.groupby("mes_dt", as_index=False)
        #.agg(ear_pct_sub=("ear_pct_sub", "mean"))
        #.sort_values("mes_dt")
    #)

   # df = (
      #  gfom_total.merge(ear_sub, on="mes_dt", how="inner")
       # .merge(cref, on="mes_dt", how="left")
      #  .dropna(subset=["gfom_hist_mwmed", "ear_pct_sub", "cref_amarela"])
      #  .sort_values("mes_dt")
       # .reset_index(drop=True)
  # )

    df = df[
        (df["mes_dt"] >= pd.Timestamp("2021-01-01")) &
        (df["mes_dt"] <= pd.Timestamp("2025-12-01"))
    ].copy()

    return df


def build_gfom_por_subsistema(cfg: Config) -> pd.DataFrame:
    gfom = load_gfom_historico(cfg).copy()

    ordem_subs = ["SE/CO", "S", "NE", "N"]
    gfom = gfom[gfom["subsistema"].isin(ordem_subs)].copy()

    gfom = (
        gfom.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom_hist_mwmed=("gfom_hist_mwmed", "sum"))
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    gfom = gfom[
        (gfom["mes_dt"] >= pd.Timestamp("2021-01-01")) &
        (gfom["mes_dt"] <= pd.Timestamp("2025-12-01"))
    ].copy()

    return gfom


def plot_gfom_ear_cref(
    df: pd.DataFrame,
    titulo: str | None = None,
    salvar: bool = True,
    caminho_saida: Path | None = None,
    figsize: tuple[float, float] = (13.5, 7.2),
    month_interval: int = 6,
) -> None:
    if df.empty:
        raise ValueError("A base final do gráfico GFOM x EAR x CRef ficou vazia.")

    apply_publication_style()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax2 = ax.twinx()

    bars = ax2.bar(
        df["mes_dt"],
        df["gfom_hist_mwmed"],
        width=22,
        color="#B3B3B3",
        alpha=0.65,
        edgecolor="none",
        label="GFOM",
        zorder=1,
    )

    line1, = ax.plot(
        df["mes_dt"],
        df["ear_pct_sub"],
        color="#4C78A8",
        linewidth=2.4,
        label="Stored energy",
        zorder=3,
    )

    line2, = ax.plot(
        df["mes_dt"],
        df["cref_amarela"],
        color="#F28E2B",
        linewidth=2.2,
        linestyle="--",
        label="Reference curve",
        zorder=3,
    )

    if titulo:
        ax.set_title(titulo, pad=8)

    format_primary_axis(ax, "GFOM (MW médio)")
   # format_secondary_axis(ax2, "Energia Armazenada / Curva de Referência Amarela (%)")
    format_time_axis(ax, "2020-12-01", "2025-12-31", month_interval=month_interval)

    add_top_legend(
        ax,
        handles=[line1, line2, bars],
        labels=["GFOM"],
        ncol=3,
    )

    if salvar and caminho_saida is not None:
        save_figure(fig, caminho_saida)

    plt.show()


def plot_gfom_por_subsistema(
    df: pd.DataFrame,
    titulo: str | None = None,
    salvar: bool = True,
    caminho_saida: Path | None = None,
    figsize: tuple[float, float] = (13.5, 7.2),
    month_interval: int = 6,
) -> None:
    if df.empty:
        raise ValueError("A base de GFOM por subsistema ficou vazia.")

    apply_publication_style()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cores = {
        "SE/CO": "#4C78A8",
        "S": "#F58518",
        "NE": "#54A24B",
        "N": "#B279A2",
    }

    nomes_legenda = {
        "SE/CO": "Sudeste/Centro-Oeste",
        "S": "Sul",
        "NE": "Nordeste",
        "N": "Norte",
    }

    ordem_subs = ["SE/CO", "S", "NE", "N"]

    for subsistema in ordem_subs:
        base_sub = df[df["subsistema"] == subsistema].copy()
        if base_sub.empty:
            continue

        ax.plot(
            base_sub["mes_dt"],
            base_sub["gfom_hist_mwmed"],
            linewidth=2.6,
            label=nomes_legenda[subsistema],
            color=cores[subsistema],
            zorder=3,
        )

    if titulo:
        ax.set_title(titulo, pad=8)

    format_primary_axis(ax, "GFOM (MW médio)")
    format_time_axis(ax, "2021-01-01", "2025-12-31", month_interval=month_interval)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=4,
        frameon=False,
        fontsize=18,
    )

    if salvar and caminho_saida is not None:
        save_figure(fig, caminho_saida)

    plt.show()


def main(cfg: Config = CFG) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    caminho_figura_7 = out_dir / "figure_7_gfom_ear_cref_2021_2025.png"
    caminho_figura_8 = out_dir / "figure_8_gfom_por_subsistema_2021_2025.png"

    df_figura_7 = build_plot_base(
        cfg=cfg,
        subsistemas_gfom=["SE/CO", "S", "NE", "N"],
        subsistema_ear="SE/CO",
    )

    df_figura_8 = build_gfom_por_subsistema(cfg)

    print("Base Figura 7:")
    print(df_figura_7.head())
    print(df_figura_7.tail())
    print()

    print("Base Figura 8:")
    print(df_figura_8.head())
    print(df_figura_8.tail())
    print()

    plot_gfom_ear_cref(
        df=df_figura_7,
        titulo=None,
        salvar=True,
        caminho_saida=caminho_figura_7,
        figsize=(13.5, 7.2),
        month_interval=6,
    )

    plot_gfom_por_subsistema(
        df=df_figura_8,
        titulo=None,
        salvar=True,
        caminho_saida=caminho_figura_8,
        figsize=(13.5, 7.2),
        month_interval=6,
    )

    print(f"PNG Figura 7 salvo em: {caminho_figura_7}")
    print(f"PDF Figura 7 salvo em: {caminho_figura_7.with_suffix('.pdf')}")
    print(f"PNG Figura 8 salvo em: {caminho_figura_8}")
    print(f"PDF Figura 8 salvo em: {caminho_figura_8.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()