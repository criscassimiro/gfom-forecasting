# -*- coding: utf-8 -*-
"""
GRÁFICO E7 - RELAÇÃO ENTRE GFOM, EAR E CREF AMARELA EM 2021

Figura gerada:
1. GFOM histórica total + EAR + CRef Amarela

Regras:
- GFOM histórica agregada por média mensal e somada entre subsistemas
- EAR histórica capturada no primeiro instante do mês
- CRef Amarela mensal
- Período: jan/2021 a dez/2021
- Exportação em PNG e PDF
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    rotation: int = 35,
) -> None:
    meses_pt = {
        1: "jan/2021",
        2: "fev/2021",
        3: "mar/2021",
        4: "abr/2021",
        5: "mai/2021",
        6: "jun/2021",
        7: "jul/2021",
        8: "ago/2021",
        9: "set/2021",
        10: "out/2021",
        11: "nov/2021",
        12: "dez/2021",
    }

    ticks = pd.date_range(start="2021-01-01", end="2021-12-01", freq="MS")

    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [meses_pt[t.month] for t in ticks],
        rotation=rotation,
        ha="right",
        fontsize=18
    )


def format_left_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=18, color="#1F4E79")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="y", colors="#1F4E79")


def format_right_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=18, color="#F28E2B")
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="y", which="major", labelsize=18, colors="#F28E2B")


def load_cref_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.cref_xlsx, sheet_name="Consolidado")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])

    df["cref_amarela"] = pd.to_numeric(
        df["curva_amarela"],
        errors="coerce"
    )

    df = (
        df[["mes_dt", "cref_amarela"]]
        .dropna()
        .drop_duplicates()
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return df[df["mes_dt"].dt.year == 2021].copy()


def load_ear_historico(
    cfg: Config,
    subsistema_ear: str = "SE/CO",
) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_historico_csv, sep=";")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data"])

    df = df.sort_values(["subsistema", "data"]).copy()

    df = (
        df.groupby(["subsistema", "mes_dt"], as_index=False)
        .first()
    )

    df["ear_pct_sub"] = pd.to_numeric(
        df["ear_fim_per"],
        errors="coerce"
    )

    df = df[
        (df["subsistema"] == subsistema_ear) &
        (df["mes_dt"].dt.year == 2021)
    ].copy()

    return (
        df[["mes_dt", "ear_pct_sub"]]
        .dropna()
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )


def load_gfom_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";")

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data_hora"])

    df["val_verifgarantiaenergetica"] = pd.to_numeric(
        df["val_verifgarantiaenergetica"],
        errors="coerce"
    )

    gfom = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom_hist_mwmed=("val_verifgarantiaenergetica", "mean"))
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    return gfom[gfom["mes_dt"].dt.year == 2021].copy()


def build_plot_base_2021(
    cfg: Config,
    subsistemas_gfom: list[str] | None = None,
    subsistema_ear: str = "SE/CO",
) -> pd.DataFrame:
    gfom = load_gfom_historico(cfg)
    ear = load_ear_historico(cfg, subsistema_ear=subsistema_ear)
    cref = load_cref_historico(cfg)

    if subsistemas_gfom is not None:
        gfom = gfom[gfom["subsistema"].isin(subsistemas_gfom)].copy()

    gfom_total = (
        gfom.groupby("mes_dt", as_index=False)
        .agg(gfom_hist_mwmed=("gfom_hist_mwmed", "sum"))
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    df = (
        gfom_total
        .merge(ear, on="mes_dt", how="inner")
        .merge(cref, on="mes_dt", how="left")
        .dropna(subset=["gfom_hist_mwmed", "ear_pct_sub", "cref_amarela"])
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    df = df[
        (df["mes_dt"] >= pd.Timestamp("2021-01-01")) &
        (df["mes_dt"] <= pd.Timestamp("2021-12-01"))
    ].copy()

    return df


def plot_gfom_ear_cref_2021(
    df: pd.DataFrame,
    titulo: str | None = None,
    salvar: bool = True,
    caminho_saida: Path | None = None,
    figsize: tuple[float, float] = (14.5, 7.5),
) -> None:
    if df.empty:
        raise ValueError("A base final do gráfico GFOM x EAR x CRef em 2021 ficou vazia.")

    apply_publication_style()

    fig, ax_ear = plt.subplots(figsize=figsize, constrained_layout=True)
    ax_gfom = ax_ear.twinx()

    ax_ear.set_zorder(ax_gfom.get_zorder() + 1)
    ax_ear.patch.set_visible(False)

    bars = ax_gfom.bar(
        df["mes_dt"],
        df["gfom_hist_mwmed"],
        width=22,
        color="#F28E2B",
        alpha=0.85,
        edgecolor="none",
        label="GFOM (MW médio)",
        zorder=1,
    )

    line_ear, = ax_ear.plot(
        df["mes_dt"],
        df["ear_pct_sub"],
        color="#1F4E79",
        linewidth=2.8,
        marker="o",
        markersize=5.5,
        label="EAR (%)",
        zorder=4,
    )

    line_cref, = ax_ear.plot(
        df["mes_dt"],
        df["cref_amarela"],
        color="#F2C94C",
        linewidth=2.8,
        linestyle="--",
        marker="o",
        markersize=5.5,
        label="CRef Amarela (%)",
        zorder=4,
    )

    if titulo:
        ax_ear.set_title(titulo, pad=12)

    format_left_axis(ax_ear, "EAR e CRef Amarela (%)")
    format_right_axis(ax_gfom, "GFOM (MW médio)")

    format_time_axis(
        ax_ear,
        "2021-01-01",
        "2021-12-31",
        rotation=35,
    )

    ax_ear.legend(
        handles=[line_ear, line_cref, bars],
        labels=["EAR (%)", "CRef Amarela (%)", "GFOM (MW médio)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        fontsize=18,
    )

    if salvar and caminho_saida is not None:
        save_figure(fig, caminho_saida)

    plt.show()


def main(cfg: Config = CFG) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    caminho_figura = out_dir / "figure_gfom_ear_cref_2021_pt.png"

    df_figura = build_plot_base_2021(
        cfg=cfg,
        subsistemas_gfom=["SE/CO", "S", "NE", "N"],
        subsistema_ear="SE/CO",
    )

    print("Base do gráfico GFOM x EAR x CRef - 2021:")
    print(df_figura)

    plot_gfom_ear_cref_2021(
        df=df_figura,
        titulo=None,
        salvar=True,
        caminho_saida=caminho_figura,
        figsize=(14.5, 7.5),
    )

    print(f"PNG salvo em: {caminho_figura}")
    print(f"PDF salvo em: {caminho_figura.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()