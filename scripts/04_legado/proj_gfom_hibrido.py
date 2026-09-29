# -*- coding: utf-8 -*-
"""
PROJEÇÃO HÍBRIDA DE GFOM POR SUBSISTEMA
- Gatilho: EAR_subsistema <= CRef amarela
- Magnitude: RandomForestRegressor
- Saída final: id_rodada, cenario, mes_dt, subsistema, gfom_mwmed
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(
    r"./dados_brutos"  # Aponte para a pasta local com os dados brutos do ONS (ver secao "Dados" do README) — caminho original removido por conter identificador pessoal
)


@dataclass
class Config:
    base_dir: Path = BASE_DIR

    # Entradas
    gt_historico_csv: Path = BASE_DIR / "gt_historico.csv"
    geracao_historico_csv: Path = BASE_DIR / "geracao_historico.csv"
    ear_historico_csv: Path = BASE_DIR / "ear_historico.csv"
    ear_2026_csv: Path = BASE_DIR / "ear.csv"
    gt_2026_csv: Path = BASE_DIR / "ute.csv"
    cref_xlsx: Path = BASE_DIR / "cref_2021_2026.xlsx"

    # Saídas
    out_dir: Path = BASE_DIR / "ute_gfom"

    # Modelo
    random_state: int = 42
    n_estimators: int = 500
    max_depth: int | None = 8
    min_samples_leaf: int = 2
    min_samples_split: int = 4
    test_months: int = 12

    # Regra do gatilho
    trigger_inclusive: bool = True  # True -> EAR <= CRef

    # Colunas finais
    output_cols: tuple = ("id_rodada", "cenario", "mes_dt", "subsistema", "gfom_mwmed")


CFG = Config()

# =============================================================================
# LOG
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("gfom_hibrido_rf")


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    val = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(val))


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["mes_num"] = pd.to_datetime(out[date_col]).dt.month
    out["mes_sin"] = np.sin(2 * np.pi * out["mes_num"] / 12.0)
    out["mes_cos"] = np.cos(2 * np.pi * out["mes_num"] / 12.0)
    return out


def build_trigger(df: pd.DataFrame, ear_col: str, cref_col: str, inclusive: bool = True) -> pd.DataFrame:
    out = df.copy()
    if inclusive:
        out["gatilho"] = (out[ear_col] <= out[cref_col]).astype(int)
    else:
        out["gatilho"] = (out[ear_col] < out[cref_col]).astype(int)

    out["gap_below"] = (out[cref_col] - out[ear_col]).clip(lower=0.0)
    out["intensidade"] = (
        out["gap_below"] / out[cref_col].replace(0, np.nan)
    ).fillna(0.0)

    return out


# =============================================================================
# CREF
# =============================================================================
def load_cref(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.cref_xlsx, sheet_name="Consolidado")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])
    df = df.rename(columns={"curva_amarela": "cref_amarela"})
    df = df[["mes_dt", "cref_amarela"]].dropna().drop_duplicates().sort_values("mes_dt")
    return df.reset_index(drop=True)


def load_cref_historico(cfg: Config) -> pd.DataFrame:
    df = load_cref(cfg).copy()
    return df[df["mes_dt"].dt.year.between(2021, 2025)].reset_index(drop=True)


def load_cref_proj(cfg: Config) -> pd.DataFrame:
    df = load_cref(cfg).copy()
    return df[df["mes_dt"].dt.year == 2026].reset_index(drop=True)


# =============================================================================
# EAR HISTÓRICA
# =============================================================================
def load_ear_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_historico_csv, sep=";")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values(["subsistema", "data"]).copy()
    df["mes_dt"] = to_month_start(df["data"])

    # primeiro instante do mês por subsistema
    df = (
        df.groupby(["subsistema", "mes_dt"], as_index=False)
          .first()
    )

    df = df.rename(columns={"ear_fim_per": "ear_pct_sub"})
    df = df[["mes_dt", "subsistema", "ear_pct_sub"]].copy()
    df = df[df["mes_dt"].dt.year.between(2021, 2025)].reset_index(drop=True)
    return df


# =============================================================================
# GT E GFOM HISTÓRICAS
# =============================================================================
def load_gfom_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";")
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["mes_dt"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()

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

def load_gt_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.geracao_historico_csv, sep=";")
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df["mes_dt"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()


    df["fonte"] = df["fonte"].astype(str).str.upper().str.strip()

    # filtra apenas térmica
    df = df[df["fonte"].str.contains("GT", na=False)].copy()

    # AJUSTAR o nome da coluna de geração, se necessário
    df["ger_mwmed"] = pd.to_numeric(df["ger_mwmed"], errors="coerce")

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
          .agg(gt_hist_mwmed=("ger_mwmed", "mean"))
          .sort_values(["mes_dt", "subsistema"])
          .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year.between(2021, 2025)].copy()
    return out

# =============================================================================
# BASE DE TREINO
# =============================================================================
def build_train(cfg: Config) -> pd.DataFrame:
    gfom = load_gfom_historico(cfg)
    gt = load_gt_historico(cfg)
    ear = load_ear_historico(cfg)
    cref = load_cref_historico(cfg)

    train = (
        gfom.merge(gt, on=["mes_dt", "subsistema"], how="inner")
            .merge(ear, on=["mes_dt", "subsistema"], how="inner")
            .merge(cref, on="mes_dt", how="left")
            .dropna(subset=["gfom_hist_mwmed", "gt_hist_mwmed", "ear_pct_sub", "cref_amarela"])
            .sort_values(["mes_dt", "subsistema"])
            .reset_index(drop=True)
    )

    train = build_trigger(
        train,
        ear_col="ear_pct_sub",
        cref_col="cref_amarela",
        inclusive=cfg.trigger_inclusive
    )

    train["share_hist"] = (
        train["gfom_hist_mwmed"] / train["gt_hist_mwmed"].replace(0, np.nan)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    train = add_calendar_features(train, "mes_dt")
    return train


# =============================================================================
# EAR PROJETADA 2026
# =============================================================================
def load_ear_proj_2026(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_2026_csv, sep=";")
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")

    # ordenar para pegar o primeiro dia e primeira hora do mês
    df = df.sort_values(["id_rodada", "cenario", "subsistema", "data_hora"]).copy()
    df["mes_dt"] = to_month_start(df["data_hora"])

    out = (
        df.groupby(["id_rodada", "cenario", "subsistema", "mes_dt"], as_index=False)
          .first()
    )

    out = out.rename(columns={"ear_per": "ear_pct_sub"})
    out = out[["id_rodada", "cenario", "mes_dt", "subsistema", "ear_pct_sub"]].copy()
    out = out[out["mes_dt"].dt.year == 2026].reset_index(drop=True)
    return out


# =============================================================================
# GT PROJETADA 2026
# =============================================================================
def load_gt_proj_2026(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_2026_csv, sep=";")

    # monta uma data mensal a partir de ano e mes
    df["mes_dt"] = pd.to_datetime(
        dict(year=df["ano"], month=df["mes"], day=1),
        errors="coerce"
    )

    out = (
        df.groupby(["id_rodada", "cenario", "mes_dt", "subsistema"], as_index=False)
          .agg(gt_proj_mwmed=("geracao_mwmed", "sum"))
          .sort_values(["id_rodada", "cenario", "mes_dt", "subsistema"])
          .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year == 2026].copy()
    return out


# =============================================================================
# BASE DE PROJEÇÃO
# =============================================================================
def build_predict_base(cfg: Config) -> pd.DataFrame:
    ear = load_ear_proj_2026(cfg)
    gt = load_gt_proj_2026(cfg)
    cref = load_cref_proj(cfg)

    df = (
        ear.merge(gt, on=["id_rodada", "cenario", "mes_dt", "subsistema"], how="inner")
           .merge(cref, on="mes_dt", how="left")
           .dropna(subset=["ear_pct_sub", "gt_proj_mwmed", "cref_amarela"])
           .sort_values(["id_rodada", "cenario", "mes_dt", "subsistema"])
           .reset_index(drop=True)
    )

    df = build_trigger(
        df,
        ear_col="ear_pct_sub",
        cref_col="cref_amarela",
        inclusive=cfg.trigger_inclusive
    )

    df = add_calendar_features(df, "mes_dt")
    return df


# =============================================================================
# MODELO RANDOM FOREST
# =============================================================================
def get_feature_cols() -> list[str]:
    return [
        "ear_pct_sub",
        "cref_amarela",
        "gap_below",
        "intensidade",
        "gt_hist_mwmed",
        "mes_num",
        "mes_sin",
        "mes_cos",
    ]


def filter_ml_train(train: pd.DataFrame) -> pd.DataFrame:
    # treinamos apenas nos meses com gatilho ativo
    base = train[train["gatilho"] == 1].copy()

    # mantém zeros dentro do gatilho para o modelo aprender gatilho fraco
    if base.empty:
        raise ValueError("A base de treino com gatilho ativo ficou vazia.")

    return base.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True)


def train_random_forest(train: pd.DataFrame, cfg: Config):
    base = filter_ml_train(train)
    features = get_feature_cols()

    X = base[features].copy()
    y = base["share_hist"].copy()

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model, features, base


def run_holdout_backtest(train: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = train.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True).copy()

    meses = sorted(df["mes_dt"].drop_duplicates())
    if len(meses) <= cfg.test_months + 6:
        log.warning("Histórico curto para holdout robusto. Backtest simplificado será ignorado.")
        return pd.DataFrame(), pd.DataFrame()

    meses_teste = meses[-cfg.test_months:]
    treino = df[~df["mes_dt"].isin(meses_teste)].copy()
    teste = df[df["mes_dt"].isin(meses_teste)].copy()

    model, features, _ = train_random_forest(treino, cfg)

    pred = teste.copy()
    pred["share_pred"] = 0.0

    mask = pred["gatilho"] == 1
    if mask.any():
        pred.loc[mask, "share_pred"] = model.predict(pred.loc[mask, features])

    pred["share_pred"] = pred["share_pred"].clip(lower=0.0, upper=1.0)
    pred["gfom_pred_mwmed"] = pred["share_pred"] * pred["gt_hist_mwmed"]
    pred.loc[pred["gatilho"] == 0, "gfom_pred_mwmed"] = 0.0

    y_true = pred["gfom_hist_mwmed"].values
    y_pred = pred["gfom_pred_mwmed"].values

    metrics = pd.DataFrame([{
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "smape": smape(y_true, y_pred),
        "n_obs": len(pred),
    }])

    return pred, metrics


def compute_feature_importance(model, base_ml: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    perm = permutation_importance(
        model,
        base_ml[features],
        base_ml["share_hist"],
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    out = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return out


# =============================================================================
# PROJEÇÃO FINAL
# =============================================================================
def apply_projection(df_proj: pd.DataFrame, model, features: list[str]) -> pd.DataFrame:
    out = df_proj.copy()

    # compatibiliza a feature de escala do treino com a projeção
    out["gt_hist_mwmed"] = out["gt_proj_mwmed"]

    out["share_pred"] = 0.0
    mask = out["gatilho"] == 1
    if mask.any():
        out.loc[mask, "share_pred"] = model.predict(out.loc[mask, features])

    out["share_pred"] = out["share_pred"].clip(lower=0.0, upper=1.0)
    out["gfom_proj_mwmed"] = out["share_pred"] * out["gt_proj_mwmed"]
    out.loc[out["gatilho"] == 0, "gfom_proj_mwmed"] = 0.0

    return out


def finalize_output(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    out["mes_dt"] = pd.to_datetime(out["mes_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["gfom_mwmed"] = pd.to_numeric(out["gfom_proj_mwmed"], errors="coerce").fillna(0.0).clip(lower=0.0)

    out = out[list(cfg.output_cols)].copy()
    return out


# =============================================================================
# SALVAR
# =============================================================================
def save_outputs(
    out_final: pd.DataFrame,
    backtest_pred: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    importance_df: pd.DataFrame,
    cfg: Config,
) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    out_final["gfom_mwmed"] = pd.to_numeric(
        out_final["gfom_mwmed"], errors="coerce"
    ).fillna(0.0).round(2)

    out_final.to_csv(
        out_dir / "ute_gfom.csv",
        index=False,
        sep=";",
        decimal=".",
        encoding="utf-8-sig"
    )

    if not backtest_pred.empty:
        backtest_pred.to_csv(
            out_dir / "backtest_predicoes_rf.csv",
            index=False,
            sep=",",
            decimal="."
        )

    if not backtest_metrics.empty:
        backtest_metrics.to_csv(
            out_dir / "backtest_metricas_rf.csv",
            index=False,
            sep=",",
            decimal="."
        )

    if not importance_df.empty:
        importance_df.to_csv(
            out_dir / "feature_importance_rf.csv",
            index=False,
            sep=",",
            decimal="."
        )

    log.info("Arquivos salvos em: %s", out_dir)


# =============================================================================
# MAIN
# =============================================================================
def main(cfg: Config = CFG) -> None:
    log.info("Iniciando projeção híbrida de GFOM por subsistema")

    # treino histórico
    train = build_train(cfg)

    # backtest
    backtest_pred, backtest_metrics = run_holdout_backtest(train, cfg)

    # treino final
    model, features, base_ml = train_random_forest(train, cfg)

    # importância
    importance_df = compute_feature_importance(model, base_ml, features)

    # projeção
    df_proj = build_predict_base(cfg)
    proj = apply_projection(df_proj, model, features)

    # saída final
    out_final = finalize_output(proj, cfg)

    # salvar
    save_outputs(out_final, backtest_pred, backtest_metrics, importance_df, cfg)

    log.info("Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()