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
    cref_xlsx: Path = BASE_DIR / "cref_2021_2026.xlsx"
    out_dir: Path = BASE_DIR / "graficos"


CFG = Config()


def to_month_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def normalizar_subsistema(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip().replace({
        "SUDESTE": "SE/CO",
        "SE": "SE/CO",
        "SE/CO": "SE/CO",
        "SUL": "S",
        "NORDESTE": "NE",
        "NORTE": "N",
    })


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def carregar_gfom_sin_2021(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";")

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data_hora"])
    df["subsistema"] = normalizar_subsistema(df["subsistema"])

    df["gfom_mwmed"] = pd.to_numeric(
        df["val_verifgarantiaenergetica"],
        errors="coerce"
    )

    df = df[
        (df["mes_dt"].dt.year == 2021) &
        (df["subsistema"].isin(["SE/CO", "S", "NE", "N"]))
    ].copy()

    gfom_sub = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom_mwmed=("gfom_mwmed", "mean"))
    )

    gfom_sin = (
        gfom_sub.groupby("mes_dt", as_index=False)
        .agg(gfom_sin_mwmed=("gfom_mwmed", "sum"))
        .sort_values("mes_dt")
    )

    return gfom_sin


def carregar_ear_seco_2021(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_historico_csv, sep=";")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])
    df["subsistema"] = normalizar_subsistema(df["subsistema"])

    df = df.sort_values(["subsistema", "data"])

    df = (
        df.groupby(["subsistema", "mes_dt"], as_index=False)
        .first()
    )

    df["ear_pct"] = pd.to_numeric(df["ear_fim_per"], errors="coerce")

    return (
        df[
            (df["mes_dt"].dt.year == 2021) &
            (df["subsistema"] == "SE/CO")
        ][["mes_dt", "ear_pct"]]
        .dropna()
        .sort_values("mes_dt")
    )


def carregar_cref_2021(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.cref_xlsx, sheet_name="Consolidado")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])

    df["cref_amarela"] = pd.to_numeric(
        df["curva_amarela"],
        errors="coerce"
    )

    return (
        df[df["mes_dt"].dt.year == 2021]
        [["mes_dt", "cref_amarela"]]
        .dropna()
        .drop_duplicates()
        .sort_values("mes_dt")
    )


def montar_base(cfg: Config) -> pd.DataFrame:
    gfom = carregar_gfom_sin_2021(cfg)
    ear = carregar_ear_seco_2021(cfg)
    cref = carregar_cref_2021(cfg)

    df = (
        gfom
        .merge(ear, on="mes_dt", how="inner")
        .merge(cref, on="mes_dt", how="inner")
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    df["gap_cref_ear"] = df["cref_amarela"] - df["ear_pct"]

    return df


def plotar_gfom_gap_2021(df: pd.DataFrame, caminho_saida: Path) -> None:
    meses_pt = {
        1: "jan/2021", 2: "fev/2021", 3: "mar/2021", 4: "abr/2021",
        5: "mai/2021", 6: "jun/2021", 7: "jul/2021", 8: "ago/2021",
        9: "set/2021", 10: "out/2021", 11: "nov/2021", 12: "dez/2021",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "savefig.dpi": 600,
    })

    fig, ax_gap = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
    ax_gfom = ax_gap.twinx()

    ax_gap.set_zorder(ax_gfom.get_zorder() + 1)
    ax_gap.patch.set_visible(False)

    barras = ax_gfom.bar(
        df["mes_dt"],
        df["gfom_sin_mwmed"],
        width=22,
        color="#F28E2B",
        alpha=0.85,
        edgecolor="none",
        label="GFOM SIN (MW médio)",
        zorder=1,
    )

    linha, = ax_gap.plot(
        df["mes_dt"],
        df["gap_cref_ear"],
        color="#1F4E79",
        linewidth=2.8,
        marker="o",
        markersize=5.5,
        label="Gap Below (%) (CRef Amarela - EAR SE/CO)",
        zorder=4,
    )

    ax_gap.axhline(
        0,
        color="#4D4D4D",
        linewidth=1.0,
        linestyle="--",
        alpha=0.80,
    )

    ticks = pd.date_range("2021-01-01", "2021-12-01", freq="MS")

    ax_gap.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
    ax_gap.set_xticks(ticks)
    ax_gap.set_xticklabels(
        [meses_pt[t.month] for t in ticks],
        rotation=35,
        ha="right"
    )

    ax_gap.set_ylabel("CRef Amarela - EAR SE/CO (%)", color="#1F4E79")
    ax_gfom.set_ylabel("GFOM (MW médio)", color="#F28E2B")

    ax_gap.tick_params(axis="y", colors="#1F4E79")
    ax_gfom.tick_params(axis="y", colors="#F28E2B")

    ax_gap.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))
    ax_gfom.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax_gap.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.30)
    ax_gap.grid(False, axis="x")

    ax_gap.spines["top"].set_visible(False)
    ax_gfom.spines["top"].set_visible(False)

    ax_gap.legend(
        handles=[linha, barras],
        labels=["Gap Below (%) (CRef Amarela - EAR SE/CO)", "GFOM (MW médio)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
    )

    fig.savefig(caminho_saida, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(caminho_saida.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    plt.show()


def main(cfg: Config = CFG) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    caminho_saida = out_dir / "figure_gfom_sin_gap_cref_ear_2021.png"

    df = montar_base(cfg)

    print("Base do gráfico:")
    print(df)

    plotar_gfom_gap_2021(df, caminho_saida)

    print(f"PNG salvo em: {caminho_saida}")
    print(f"PDF salvo em: {caminho_saida.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()