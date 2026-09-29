# -*- coding: utf-8 -*-
"""
BACKTEST COMPARATIVO - RANDOM FOREST vs GRADIENT BOOSTING
Métricas calculadas no nível agregado do SIN

Lógica:
1. Treina por subsistema
2. Prediz GFOM por subsistema
3. Agrega real e previsto por mês no SIN
4. Calcula MAE, RMSE, R², Precision, Recall e F1 no SIN
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(
    r"./dados_brutos"  # Aponte para a pasta local com os dados brutos do ONS (ver secao "Dados" do README) — caminho original removido por conter identificador pessoal
)


@dataclass
class Config:
    base_dir: Path = BASE_DIR

    gt_historico_csv: Path = BASE_DIR / "gt_historico.csv"
    geracao_historico_csv: Path = BASE_DIR / "geracao_historico.csv"
    ear_historico_csv: Path = BASE_DIR / "ear_historico.csv"
    cref_xlsx: Path = BASE_DIR / "cref_2021_2026.xlsx"

    out_dir: Path = BASE_DIR / "ute_gfom"

    trigger_inclusive: bool = True
    occurrence_threshold_mwmed: float = 1.0

    min_train_months: int = 12

    rf_random_state: int = 42
    rf_n_estimators: int = 500
    rf_max_depth: int | None = 8
    rf_min_samples_leaf: int = 2
    rf_min_samples_split: int = 4

    gb_random_state: int = 42
    gb_n_estimators: int = 300
    gb_learning_rate: float = 0.05
    gb_max_depth: int = 3
    gb_min_samples_leaf: int = 2
    gb_min_samples_split: int = 4
    gb_subsample: float = 1.0


CFG = Config()


# =============================================================================
# LOG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("backtest_sin_rf_gb")


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_month_start(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s.dt.to_period("M").dt.to_timestamp()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["mes_num"] = pd.to_datetime(out[date_col]).dt.month
    out["mes_sin"] = np.sin(2 * np.pi * out["mes_num"] / 12.0)
    out["mes_cos"] = np.cos(2 * np.pi * out["mes_num"] / 12.0)
    return out


def build_trigger(
    df: pd.DataFrame,
    ear_col: str,
    cref_col: str,
    inclusive: bool = True,
) -> pd.DataFrame:
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


# =============================================================================
# LEITURA DAS BASES
# =============================================================================

def load_cref(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.cref_xlsx, sheet_name="Consolidado")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])

    df = df.rename(columns={"curva_amarela": "cref_amarela"})

    df = (
        df[["mes_dt", "cref_amarela"]]
        .dropna()
        .drop_duplicates()
        .sort_values("mes_dt")
        .reset_index(drop=True)
    )

    return df[df["mes_dt"].dt.year.between(2021, 2025)].reset_index(drop=True)


def load_ear_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_historico_csv, sep=";", low_memory=False)

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes_dt"] = to_month_start(df["data"])

    df = df.sort_values(["subsistema", "data"]).copy()

    df = (
        df.groupby(["subsistema", "mes_dt"], as_index=False)
        .first()
    )

    df = df.rename(columns={"ear_fim_per": "ear_pct_sub"})

    df = df[["mes_dt", "subsistema", "ear_pct_sub"]].copy()
    df = df[df["mes_dt"].dt.year.between(2021, 2025)]

    return df.reset_index(drop=True)


def load_gfom_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";", low_memory=False)

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce", utc=True)
    df["data_hora"] = df["data_hora"].dt.tz_localize(None)
    df["mes_dt"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()

    df["val_verifgarantiaenergetica"] = pd.to_numeric(
        df["val_verifgarantiaenergetica"],
        errors="coerce"
    )

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gfom_hist_mwmed=("val_verifgarantiaenergetica", "mean"))
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year.between(2021, 2025)]

    return out.reset_index(drop=True)


def load_gt_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.geracao_historico_csv, sep=";", low_memory=False)

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce", utc=True)
    df["data_hora"] = df["data_hora"].dt.tz_localize(None)
    df["mes_dt"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()

    df["fonte"] = df["fonte"].astype(str).str.upper().str.strip()
    df = df[df["fonte"].str.contains("GT", na=False)].copy()

    df["ger_mwmed"] = pd.to_numeric(df["ger_mwmed"], errors="coerce")

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(gt_hist_mwmed=("ger_mwmed", "mean"))
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year.between(2021, 2025)]

    return out.reset_index(drop=True)


# =============================================================================
# BASE DE TREINO
# =============================================================================

def build_train(cfg: Config) -> pd.DataFrame:
    gfom = load_gfom_historico(cfg)
    gt = load_gt_historico(cfg)
    ear = load_ear_historico(cfg)
    cref = load_cref(cfg)

    train = (
        gfom.merge(gt, on=["mes_dt", "subsistema"], how="inner")
        .merge(ear, on=["mes_dt", "subsistema"], how="inner")
        .merge(cref, on="mes_dt", how="left")
        .dropna(
            subset=[
                "gfom_hist_mwmed",
                "gt_hist_mwmed",
                "ear_pct_sub",
                "cref_amarela",
            ]
        )
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    train = build_trigger(
        train,
        ear_col="ear_pct_sub",
        cref_col="cref_amarela",
        inclusive=cfg.trigger_inclusive,
    )

    train["share_hist"] = (
        train["gfom_hist_mwmed"] / train["gt_hist_mwmed"].replace(0, np.nan)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    train = add_calendar_features(train, "mes_dt")

    return train


def filter_ml_train(train: pd.DataFrame) -> pd.DataFrame:
    base = train.dropna(
        subset=[
            "ear_pct_sub",
            "cref_amarela",
            "gap_below",
            "intensidade",
            "gt_hist_mwmed",
            "mes_num",
            "mes_sin",
            "mes_cos",
            "share_hist",
        ]
    ).copy()

    if base.empty:
        raise ValueError("A base de treino ficou vazia.")

    return base.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True)


# =============================================================================
# MODELOS
# =============================================================================

def train_random_forest(train: pd.DataFrame, cfg: Config):
    base = filter_ml_train(train)
    features = get_feature_cols()

    X = base[features].copy()
    y = base["share_hist"].copy()

    model = RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        min_samples_split=cfg.rf_min_samples_split,
        random_state=cfg.rf_random_state,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model, features


def train_gradient_boosting(train: pd.DataFrame, cfg: Config):
    base = filter_ml_train(train)
    features = get_feature_cols()

    X = base[features].copy()
    y = base["share_hist"].copy()

    model = GradientBoostingRegressor(
        n_estimators=cfg.gb_n_estimators,
        learning_rate=cfg.gb_learning_rate,
        max_depth=cfg.gb_max_depth,
        min_samples_leaf=cfg.gb_min_samples_leaf,
        min_samples_split=cfg.gb_min_samples_split,
        subsample=cfg.gb_subsample,
        random_state=cfg.gb_random_state,
    )

    model.fit(X, y)

    return model, features


# =============================================================================
# PREDIÇÃO
# =============================================================================

def apply_model_to_test(
    test_df: pd.DataFrame,
    model,
    features: list[str],
    model_name: str,
) -> pd.DataFrame:
    pred = test_df.copy()

    pred["modelo"] = model_name
    pred["share_pred"] = 0.0

    mask = pred["gatilho"] == 1

    if mask.any():
        pred.loc[mask, "share_pred"] = model.predict(pred.loc[mask, features])

    pred["share_pred"] = pred["share_pred"].clip(lower=0.0, upper=1.0)

    pred["gfom_pred_mwmed"] = pred["share_pred"] * pred["gt_hist_mwmed"]
    pred.loc[pred["gatilho"] == 0, "gfom_pred_mwmed"] = 0.0

    return pred


# =============================================================================
# AGREGAÇÃO SIN E MÉTRICAS
# =============================================================================

def aggregate_to_sin(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sin = (
        pred.groupby(["modelo", "mes_dt"], as_index=False)
        .agg(
            gfom_real_sin=("gfom_hist_mwmed", "sum"),
            gfom_pred_sin=("gfom_pred_mwmed", "sum"),
        )
        .sort_values(["modelo", "mes_dt"])
        .reset_index(drop=True)
    )

    threshold = cfg.occurrence_threshold_mwmed

    sin["ocorreu_real"] = (sin["gfom_real_sin"] > threshold).astype(int)
    sin["ocorreu_pred"] = (sin["gfom_pred_sin"] > threshold).astype(int)

    return sin


def compute_metrics_sin(pred_sin_modelo: pd.DataFrame) -> dict:
    modelo = pred_sin_modelo["modelo"].iloc[0]

    y_true = pred_sin_modelo["gfom_real_sin"].values
    y_pred = pred_sin_modelo["gfom_pred_sin"].values

    y_true_cls = pred_sin_modelo["ocorreu_real"].values
    y_pred_cls = pred_sin_modelo["ocorreu_pred"].values

    return {
        "modelo": modelo,
        "nivel_avaliacao": "SIN_agregado_mensal",
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
        "recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
        "f1_score": f1_score(y_true_cls, y_pred_cls, zero_division=0),
        "n_obs": len(pred_sin_modelo),
        "n_eventos_reais": int(pred_sin_modelo["ocorreu_real"].sum()),
        "n_eventos_preditos": int(pred_sin_modelo["ocorreu_pred"].sum()),
    }


# =============================================================================
# BACKTEST ROLLING-ORIGIN
# =============================================================================

def run_rolling_backtest_compare(cfg: Config):
    df = build_train(cfg).sort_values(["mes_dt", "subsistema"]).reset_index(drop=True)

    meses = sorted(df["mes_dt"].drop_duplicates())

    resultados = []

    for i in range(cfg.min_train_months, len(meses)):
        mes_teste = meses[i]
        meses_treino = meses[:i]

        treino = df[df["mes_dt"].isin(meses_treino)].copy()
        teste = df[df["mes_dt"] == mes_teste].copy()

        if treino.empty or teste.empty:
            continue

        log.info("Rodando backtest para mês: %s", pd.Timestamp(mes_teste).strftime("%Y-%m"))

        rf_model, rf_features = train_random_forest(treino, cfg)
        pred_rf = apply_model_to_test(
            test_df=teste,
            model=rf_model,
            features=rf_features,
            model_name="Random Forest",
        )

        gb_model, gb_features = train_gradient_boosting(treino, cfg)
        pred_gb = apply_model_to_test(
            test_df=teste,
            model=gb_model,
            features=gb_features,
            model_name="Gradient Boosting",
        )

        resultados.append(pred_rf)
        resultados.append(pred_gb)

    if not resultados:
        raise RuntimeError("Nenhuma previsão foi gerada no backtest.")

    pred_all = pd.concat(resultados, ignore_index=True)

    pred_sin = aggregate_to_sin(pred_all, cfg)

    metrics = pd.DataFrame(
        [
            compute_metrics_sin(pred_sin[pred_sin["modelo"] == "Random Forest"].copy()),
            compute_metrics_sin(pred_sin[pred_sin["modelo"] == "Gradient Boosting"].copy()),
        ]
    )

    return pred_all, pred_sin, metrics


# =============================================================================
# SALVAR
# =============================================================================

def save_outputs(
    pred_all: pd.DataFrame,
    pred_sin: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: Config,
) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    pred_all_out = pred_all.copy()
    pred_all_out["mes_dt"] = pd.to_datetime(pred_all_out["mes_dt"]).dt.strftime("%Y-%m-%d")

    pred_sin_out = pred_sin.copy()
    pred_sin_out["mes_dt"] = pd.to_datetime(pred_sin_out["mes_dt"]).dt.strftime("%Y-%m-%d")

    metrics_out = metrics.copy()

    num_cols = [
        "mae",
        "rmse",
        "r2",
        "precision",
        "recall",
        "f1_score",
    ]

    for col in num_cols:
        metrics_out[col] = pd.to_numeric(metrics_out[col], errors="coerce").round(4)

    pred_all_out.to_csv(
        out_dir / "backtest_predicoes_subsistema_rf_gb.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    pred_sin_out.to_csv(
        out_dir / "backtest_predicoes_sin_rf_gb.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    metrics_out.to_csv(
        out_dir / "backtest_metricas_sin_rf_gb.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    log.info("Arquivos salvos em: %s", out_dir)


# =============================================================================
# MAIN
# =============================================================================

def main(cfg: Config = CFG) -> None:
    log.info("Iniciando backtest comparativo RF vs GB com métricas agregadas no SIN")

    pred_all, pred_sin, metrics = run_rolling_backtest_compare(cfg)

    save_outputs(
        pred_all=pred_all,
        pred_sin=pred_sin,
        metrics=metrics,
        cfg=cfg,
    )

    log.info("Métricas finais:")
    log.info("\n%s", metrics)

    log.info("Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()