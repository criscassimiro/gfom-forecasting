# -*- coding: utf-8 -*-

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
    gt_historico_csv: Path = BASE_DIR / "gt_historico.csv"
    ear_historico_csv: Path = BASE_DIR / "ear_historico.csv"
    out_dir: Path = BASE_DIR / "graficos"


CFG = Config()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
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
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
    })


def save_figure(fig: plt.Figure, caminho_saida: Path) -> None:
    fig.savefig(caminho_saida, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(caminho_saida.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def carregar_gfom_2021(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";")

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data_hora"])

    df["val_verifgarantiaenergetica"] = pd.to_numeric(
        df["val_verifgarantiaenergetica"],
        errors="coerce"
    )

    df = df[
        (df["mes_dt"] >= pd.Timestamp("2021-01-01")) &
        (df["mes_dt"] <= pd.Timestamp("2021-12-01")) &
        (df["subsistema"].isin(["SE/CO", "S", "NE", "N"]))
    ].copy()

    gfom = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom_mwmed=("val_verifgarantiaenergetica", "mean"))
    )

    gfom_total = (
        gfom.groupby("mes_dt", as_index=False)
        .agg(gfom_mwmed=("gfom_mwmed", "sum"))
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return gfom_total


def carregar_ear_seco_2021(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_historico_csv, sep=";")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["subsistema"] = normalizar_subsistema(df["subsistema"])
    df["mes_dt"] = to_month_start(df["data"])

    df = df.sort_values(["subsistema", "data"]).copy()

    df = (
        df.groupby(["subsistema", "mes_dt"], as_index=False)
        .first()
    )

    df["ear_seco_pct"] = pd.to_numeric(
        df["ear_fim_per"],
        errors="coerce"
    )

    ear = df[
        (df["subsistema"] == "SE/CO") &
        (df["mes_dt"] >= pd.Timestamp("2021-01-01")) &
        (df["mes_dt"] <= pd.Timestamp("2021-12-01"))
    ].copy()

    ear = (
        ear[["mes_dt", "ear_seco_pct"]]
        .dropna()
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return ear


def montar_base_grafico(cfg: Config) -> pd.DataFrame:
    gfom = carregar_gfom_2021(cfg)
    ear = carregar_ear_seco_2021(cfg)

    df = (
        gfom
        .merge(ear, on="mes_dt", how="inner")
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return df


def plot_ear_seco_gfom_2021(
    df: pd.DataFrame,
    caminho_saida: Path,
    figsize: tuple[float, float] = (13.5, 7.2),
) -> None:
    if df.empty:
        raise ValueError("A base do gráfico EAR SE/CO x GFOM 2021 ficou vazia.")

    apply_publication_style()

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

    fig, ax_ear = plt.subplots(figsize=figsize, constrained_layout=True)
    ax_gfom = ax_ear.twinx()

    ax_ear.set_zorder(ax_gfom.get_zorder() + 1)
    ax_ear.patch.set_visible(False)

    bars = ax_gfom.bar(
        df["mes_dt"],
        df["gfom_mwmed"],
        width=22,
        color="#F28E2B",
        alpha=0.85,
        edgecolor="none",
        label="GFOM (MW médio)",
        zorder=1,
    )

    line_ear, = ax_ear.plot(
        df["mes_dt"],
        df["ear_seco_pct"],
        color="#1F4E79",
        linewidth=2.8,
        marker="o",
        markersize=5.5,
        label="EAR SE/CO (%)",
        zorder=4,
    )

    ticks = pd.date_range(start="2021-01-01", end="2021-12-01", freq="MS")

    ax_ear.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
    ax_ear.set_xticks(ticks)
    ax_ear.set_xticklabels(
        [meses_pt[t.month] for t in ticks],
        rotation=35,
        ha="right",
        fontsize=18
    )

    ax_ear.set_ylabel("EAR SE/CO (%)", color="#1F4E79")
    ax_gfom.set_ylabel("GFOM (MW médio)", color="#F28E2B")

    ax_ear.tick_params(axis="y", colors="#1F4E79")
    ax_gfom.tick_params(axis="y", colors="#F28E2B")

    ax_ear.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax_gfom.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax_ear.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax_ear.grid(False, axis="x")

    ax_ear.spines["top"].set_visible(False)
    ax_gfom.spines["top"].set_visible(False)

    ax_ear.legend(
        handles=[line_ear, bars],
        labels=["EAR SE/CO (%)", "GFOM (MW médio)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
        fontsize=18,
    )

    save_figure(fig, caminho_saida)
    plt.show()


def main(cfg: Config = CFG) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    caminho_figura = out_dir / "figure_ear_seco_gfom_2021.png"

    df_grafico = montar_base_grafico(cfg)

    print("Base do gráfico EAR SE/CO x GFOM - 2021:")
    print(df_grafico)

    plot_ear_seco_gfom_2021(
        df=df_grafico,
        caminho_saida=caminho_figura,
        figsize=(13.5, 7.2),
    )

    print(f"PNG salvo em: {caminho_figura}")
    print(f"PDF salvo em: {caminho_figura.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()