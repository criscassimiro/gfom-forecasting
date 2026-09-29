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

    gfom_2021 = (
        df.groupby("mes_dt", as_index=False)
        .agg(gfom_mwmed=("val_verifgarantiaenergetica", "mean"))
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return gfom_2021


def plot_gfom_2021(
    df: pd.DataFrame,
    caminho_saida: Path,
    figsize: tuple[float, float] = (13.5, 7.2),
) -> None:
    if df.empty:
        raise ValueError("A base de GFOM 2021 ficou vazia.")

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

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.bar(
        df["mes_dt"],
        df["gfom_mwmed"],
        width=22,
        color="#F28E2B",
        alpha=0.85,
        edgecolor="none",
        label="GFOM",
        zorder=3,
    )

    ticks = pd.date_range(start="2021-01-01", end="2021-12-01", freq="MS")

    ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [meses_pt[t.month] for t in ticks],
        rotation=35,
        ha="right",
        fontsize=18
    )

    ax.set_ylabel("GFOM (MW médio)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=1,
        frameon=False,
        fontsize=18,
    )

    save_figure(fig, caminho_saida)
    plt.show()


def main(cfg: Config = CFG) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    caminho_figura = out_dir / "figure_gfom_2021.png"

    df_gfom_2021 = carregar_gfom_2021(cfg)

    print("Base GFOM 2021:")
    print(df_gfom_2021)

    plot_gfom_2021(
        df=df_gfom_2021,
        caminho_saida=caminho_figura,
    )

    print(f"PNG salvo em: {caminho_figura}")
    print(f"PDF salvo em: {caminho_figura.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()