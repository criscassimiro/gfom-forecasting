# -*- coding: utf-8 -*-
"""
PROJEÇÃO HÍBRIDA DE GFOM POR SUBSISTEMA - GRADIENT BOOSTING

Objetivo:
- Rodar o modelo Gradient Boosting com a mesma lógica usada no RF final.
- Gatilho operacional: EAR_subsistema <= CRef amarela.
- Magnitude: GradientBoostingRegressor.
- Backtest: walk-forward / rolling-origin.
- Avaliação: 2022-2025, mês x subsistema.
- Métricas: MAE, RMSE, R², sMAPE, Precision, Recall e F1.

Saídas:
- ute_gfom_gb.csv
- backtest_predicoes_gb.csv
- backtest_metricas_gb.csv
- feature_importance_gb.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
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

    # Entradas históricas
    gt_historico_csv: Path = BASE_DIR / "gt_historico.csv"
    geracao_historico_csv: Path = BASE_DIR / "geracao_historico.csv"
    ear_historico_csv: Path = BASE_DIR / "ear_historico.csv"
    cref_xlsx: Path = BASE_DIR / "cref_2021_2026.xlsx"

    # Entradas de projeção 2026
    ear_2026_csv: Path = BASE_DIR / "ear.csv"
    gt_2026_csv: Path = BASE_DIR / "ute.csv"

    # Saídas
    out_dir: Path = BASE_DIR / "ute_gfom"

    # Modelo Gradient Boosting
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 3
    subsample: float = 0.9
    min_samples_leaf: int = 2
    min_samples_split: int = 4

    # Backtest walk-forward
    min_train_months: int = 1

    # Regra do gatilho
    trigger_inclusive: bool = True

    # Critério de ocorrência da GFOM
    occurrence_threshold_mwmed: float = 0.0

    # Colunas finais da projeção
    output_cols: tuple = (
        "id_rodada",
        "cenario",
        "mes_dt",
        "subsistema",
        "gfom_mwmed",
    )


CFG = Config()


# =============================================================================
# LOG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("gfom_hibrido_gb")


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


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(pd.Series(y_true).fillna(0), dtype=float)
    y_pred = np.asarray(pd.Series(y_pred).fillna(0), dtype=float)

    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0

    if mask.sum() == 0:
        return np.nan

    return float(
        100 * np.mean(
            2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
        )
    )


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(pd.Series(y_true).fillna(0), dtype=float)
    y_pred = np.asarray(pd.Series(y_pred).fillna(0), dtype=float)

    denom = np.sum(np.abs(y_true))

    if denom == 0:
        return np.nan

    return float(100 * np.sum(np.abs(y_true - y_pred)) / denom)


def bias(y_true, y_pred) -> float:
    y_true = np.asarray(pd.Series(y_true).fillna(0), dtype=float)
    y_pred = np.asarray(pd.Series(y_pred).fillna(0), dtype=float)

    return float(np.mean(y_pred - y_true))


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


def filtrar_periodo_validacao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["mes_dt"] = pd.to_datetime(out["mes_dt"], errors="coerce")

    out = out[
        (out["mes_dt"] >= pd.Timestamp("2022-01-01")) &
        (out["mes_dt"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    return out.reset_index(drop=True)


# =============================================================================
# CREF
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

    return df


def load_cref_historico(cfg: Config) -> pd.DataFrame:
    df = load_cref(cfg).copy()
    df = df[df["mes_dt"].dt.year.between(2021, 2025)]
    return df.reset_index(drop=True)


def load_cref_proj(cfg: Config) -> pd.DataFrame:
    df = load_cref(cfg).copy()
    df = df[df["mes_dt"].dt.year == 2026]
    return df.reset_index(drop=True)


# =============================================================================
# EAR HISTÓRICA
# =============================================================================

def load_ear_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_historico_csv, sep=";", low_memory=False)

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values(["subsistema", "data"]).copy()

    df["mes_dt"] = to_month_start(df["data"])

    df = (
        df.groupby(["subsistema", "mes_dt"], as_index=False)
        .first()
    )

    df = df.rename(columns={"ear_fim_per": "ear_pct_sub"})

    df = df[["mes_dt", "subsistema", "ear_pct_sub"]].copy()
    df = df[df["mes_dt"].dt.year.between(2021, 2025)]

    return df.reset_index(drop=True)


# =============================================================================
# GFOM E GT HISTÓRICAS
# =============================================================================

def load_gfom_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";", low_memory=False)

    df["data_hora"] = pd.to_datetime(
        df["data_hora"],
        errors="coerce",
        utc=True,
    )

    df["data_hora"] = df["data_hora"].dt.tz_localize(None)
    df["mes_dt"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()

    df["val_verifgarantiaenergetica"] = pd.to_numeric(
        df["val_verifgarantiaenergetica"],
        errors="coerce",
    )

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(
            gfom_hist_mwmed=("val_verifgarantiaenergetica", "mean")
        )
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year.between(2021, 2025)]

    return out.reset_index(drop=True)


def load_gt_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.geracao_historico_csv, sep=";", low_memory=False)

    df["data_hora"] = pd.to_datetime(
        df["data_hora"],
        errors="coerce",
        utc=True,
    )

    df["data_hora"] = df["data_hora"].dt.tz_localize(None)
    df["mes_dt"] = df["data_hora"].dt.to_period("M").dt.to_timestamp()

    df["fonte"] = df["fonte"].astype(str).str.upper().str.strip()
    df = df[df["fonte"].str.contains("GT", na=False)].copy()

    df["ger_mwmed"] = pd.to_numeric(
        df["ger_mwmed"],
        errors="coerce",
    )

    out = (
        df.groupby(["mes_dt", "subsistema"], as_index=False)
        .agg(
            gt_hist_mwmed=("ger_mwmed", "mean")
        )
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
    cref = load_cref_historico(cfg)

    train = (
        gfom
        .merge(gt, on=["mes_dt", "subsistema"], how="inner")
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
        train["gfom_hist_mwmed"] /
        train["gt_hist_mwmed"].replace(0, np.nan)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    train = add_calendar_features(train, "mes_dt")

    return train


# =============================================================================
# BASE DE PROJEÇÃO 2026
# =============================================================================

def load_ear_proj_2026(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.ear_2026_csv, sep=";", low_memory=False)

    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")

    df = df.sort_values(
        ["id_rodada", "cenario", "subsistema", "data_hora"]
    ).copy()

    df["mes_dt"] = to_month_start(df["data_hora"])

    out = (
        df.groupby(
            ["id_rodada", "cenario", "subsistema", "mes_dt"],
            as_index=False,
        )
        .first()
    )

    out = out.rename(columns={"ear_per": "ear_pct_sub"})

    out = out[
        ["id_rodada", "cenario", "mes_dt", "subsistema", "ear_pct_sub"]
    ].copy()

    out = out[out["mes_dt"].dt.year == 2026]

    return out.reset_index(drop=True)


def load_gt_proj_2026(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_2026_csv, sep=";", low_memory=False)

    df["mes_dt"] = pd.to_datetime(
        dict(
            year=df["ano"],
            month=df["mes"],
            day=1,
        ),
        errors="coerce",
    )

    out = (
        df.groupby(
            ["id_rodada", "cenario", "mes_dt", "subsistema"],
            as_index=False,
        )
        .agg(
            gt_proj_mwmed=("geracao_mwmed", "sum")
        )
        .sort_values(
            ["id_rodada", "cenario", "mes_dt", "subsistema"]
        )
        .reset_index(drop=True)
    )

    out = out[out["mes_dt"].dt.year == 2026]

    return out.reset_index(drop=True)


def build_predict_base(cfg: Config) -> pd.DataFrame:
    ear = load_ear_proj_2026(cfg)
    gt = load_gt_proj_2026(cfg)
    cref = load_cref_proj(cfg)

    df = (
        ear
        .merge(
            gt,
            on=["id_rodada", "cenario", "mes_dt", "subsistema"],
            how="inner",
        )
        .merge(cref, on="mes_dt", how="left")
        .dropna(
            subset=[
                "ear_pct_sub",
                "gt_proj_mwmed",
                "cref_amarela",
            ]
        )
        .sort_values(
            ["id_rodada", "cenario", "mes_dt", "subsistema"]
        )
        .reset_index(drop=True)
    )

    df = build_trigger(
        df,
        ear_col="ear_pct_sub",
        cref_col="cref_amarela",
        inclusive=cfg.trigger_inclusive,
    )

    df = add_calendar_features(df, "mes_dt")

    return df


# =============================================================================
# MODELO GRADIENT BOOSTING
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
    base = train.copy()

    base = base.dropna(
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
        raise ValueError("A base de treino ficou vazia após o tratamento.")

    return base.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True)


def train_gradient_boosting(train: pd.DataFrame, cfg: Config):
    base = filter_ml_train(train)
    features = get_feature_cols()

    X = base[features].copy()
    y = base["share_hist"].copy()

    model = GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
        random_state=cfg.random_state,
    )

    model.fit(X, y)

    return model, features, base


# =============================================================================
# MÉTRICAS
# =============================================================================

def calcular_metricas_classificacao(y_true_cont, y_pred_cont, threshold: float = 0.0) -> dict:
    y_true_bin = (pd.Series(y_true_cont).fillna(0) > threshold).astype(int)
    y_pred_bin = (pd.Series(y_pred_cont).fillna(0) > threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "tn": int(((y_true_bin == 0) & (y_pred_bin == 0)).sum()),
        "fp": int(((y_true_bin == 0) & (y_pred_bin == 1)).sum()),
        "fn": int(((y_true_bin == 1) & (y_pred_bin == 0)).sum()),
        "tp": int(((y_true_bin == 1) & (y_pred_bin == 1)).sum()),
    }


def calcular_metricas_regressao(y_true, y_pred) -> dict:
    y_true = pd.Series(y_true).fillna(0).astype(float)
    y_pred = pd.Series(y_pred).fillna(0).astype(float)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "bias": bias(y_true, y_pred),
        "wape_percent": wape(y_true, y_pred),
        "smape_percent": smape(y_true, y_pred),
        "real_mean": float(y_true.mean()),
        "pred_mean": float(y_pred.mean()),
        "real_sum": float(y_true.sum()),
        "pred_sum": float(y_pred.sum()),
    }


def calcular_metricas_completas(y_true, y_pred, threshold: float = 0.0) -> dict:
    resultados = {}

    resultados.update(
        calcular_metricas_regressao(y_true, y_pred)
    )

    resultados.update(
        calcular_metricas_classificacao(
            y_true,
            y_pred,
            threshold=threshold,
        )
    )

    return resultados


# =============================================================================
# BACKTEST WALK-FORWARD / ROLLING-ORIGIN
# =============================================================================

def run_walk_forward_backtest_gb(
    train: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = (
        train
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
        .copy()
    )

    meses = sorted(df["mes_dt"].drop_duplicates())

    if len(meses) < (cfg.min_train_months + 1):
        log.warning("Histórico insuficiente para backtest walk-forward.")
        return pd.DataFrame(), pd.DataFrame()

    preds = []

    for i in range(cfg.min_train_months, len(meses)):
        mes_teste = meses[i]
        meses_treino = meses[:i]

        treino = df[df["mes_dt"].isin(meses_treino)].copy()
        teste = df[df["mes_dt"] == mes_teste].copy()

        if treino.empty or teste.empty:
            continue

        try:
            model, features, _ = train_gradient_boosting(treino, cfg)
        except Exception as e:
            log.warning(
                "Falha ao treinar GB para o mês %s: %s",
                mes_teste,
                e,
            )
            continue

        pred = teste.copy()
        pred["modelo"] = "Gradient Boosting"
        pred["share_pred"] = 0.0

        mask = pred["gatilho"] == 1

        if mask.any():
            pred.loc[mask, "share_pred"] = model.predict(
                pred.loc[mask, features]
            )

        pred["share_pred"] = pred["share_pred"].clip(
            lower=0.0,
            upper=1.0,
        )

        pred["gfom_pred_mwmed"] = (
            pred["share_pred"] * pred["gt_hist_mwmed"]
        )

        pred.loc[pred["gatilho"] == 0, "gfom_pred_mwmed"] = 0.0

        preds.append(pred)

    if not preds:
        log.warning("Nenhuma previsão foi gerada no backtest.")
        return pd.DataFrame(), pd.DataFrame()

    backtest_pred = (
        pd.concat(preds, ignore_index=True)
        .sort_values(["mes_dt", "subsistema"])
        .reset_index(drop=True)
    )

    base_eval = filtrar_periodo_validacao(backtest_pred)

    if base_eval.empty:
        return backtest_pred, pd.DataFrame()

    y_true = base_eval["gfom_hist_mwmed"]
    y_pred = base_eval["gfom_pred_mwmed"]

    metricas = calcular_metricas_completas(
        y_true,
        y_pred,
        threshold=cfg.occurrence_threshold_mwmed,
    )

    metrics = pd.DataFrame(
        [{
            "modelo": "Gradient Boosting",
            "periodo": "2022_2025",
            "n_obs": len(base_eval),
            "dt_inicio_backtest": base_eval["mes_dt"].min(),
            "dt_fim_backtest": base_eval["mes_dt"].max(),
            **metricas,
        }]
    )

    return backtest_pred, metrics


def calcular_metricas_por_subsistema(backtest_pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred)

    if df.empty:
        return pd.DataFrame()

    resultados = []

    for subsistema, base in df.groupby("subsistema", dropna=False):
        metricas = calcular_metricas_completas(
            base["gfom_hist_mwmed"],
            base["gfom_pred_mwmed"],
            threshold=cfg.occurrence_threshold_mwmed,
        )

        resultados.append({
            "subsistema": subsistema,
            "n_obs": len(base),
            **metricas,
        })

    return (
        pd.DataFrame(resultados)
        .sort_values("subsistema")
        .reset_index(drop=True)
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance(
    model,
    base_ml: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

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
    })

    out = out.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    return out


# =============================================================================
# PROJEÇÃO FINAL 2026
# =============================================================================

def apply_projection(
    df_proj: pd.DataFrame,
    model,
    features: list[str],
) -> pd.DataFrame:

    out = df_proj.copy()

    out["gt_hist_mwmed"] = out["gt_proj_mwmed"]

    out["share_pred"] = 0.0

    mask = out["gatilho"] == 1

    if mask.any():
        out.loc[mask, "share_pred"] = model.predict(
            out.loc[mask, features]
        )

    out["share_pred"] = out["share_pred"].clip(
        lower=0.0,
        upper=1.0,
    )

    out["gfom_proj_mwmed"] = (
        out["share_pred"] * out["gt_proj_mwmed"]
    )

    out.loc[out["gatilho"] == 0, "gfom_proj_mwmed"] = 0.0

    return out


def finalize_output(
    df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:

    out = df.copy()

    out["mes_dt"] = pd.to_datetime(
        out["mes_dt"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    out["gfom_mwmed"] = pd.to_numeric(
        out["gfom_proj_mwmed"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    out = out[list(cfg.output_cols)].copy()

    return out


# =============================================================================
# SALVAR
# =============================================================================

def save_outputs(
    out_final: pd.DataFrame,
    backtest_pred: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    metricas_subsistema: pd.DataFrame,
    importance_df: pd.DataFrame,
    cfg: Config,
) -> None:

    out_dir = ensure_dir(cfg.out_dir)

    out_final = out_final.copy()
    backtest_pred = backtest_pred.copy()
    backtest_metrics = backtest_metrics.copy()
    metricas_subsistema = metricas_subsistema.copy()
    importance_df = importance_df.copy()

    if not out_final.empty and "gfom_mwmed" in out_final.columns:
        out_final["gfom_mwmed"] = pd.to_numeric(
            out_final["gfom_mwmed"],
            errors="coerce",
        ).fillna(0.0).round(2)

    if not backtest_pred.empty and "mes_dt" in backtest_pred.columns:
        backtest_pred["mes_dt"] = pd.to_datetime(
            backtest_pred["mes_dt"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    if not backtest_metrics.empty:
        for col in backtest_metrics.columns:
            if backtest_metrics[col].dtype.kind in "fc":
                backtest_metrics[col] = backtest_metrics[col].round(4)

    if not metricas_subsistema.empty:
        for col in metricas_subsistema.columns:
            if metricas_subsistema[col].dtype.kind in "fc":
                metricas_subsistema[col] = metricas_subsistema[col].round(4)

    out_final.to_csv(
        out_dir / "ute_gfom_gb.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    if not backtest_pred.empty:
        backtest_pred.to_csv(
            out_dir / "backtest_predicoes_gb.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

    if not backtest_metrics.empty:
        backtest_metrics.to_csv(
            out_dir / "backtest_metricas_gb.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

    if not metricas_subsistema.empty:
        metricas_subsistema.to_csv(
            out_dir / "backtest_metricas_gb_por_subsistema.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )

    if not importance_df.empty:
        importance_df.to_csv(
            out_dir / "feature_importance_gb.csv",
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
    log.info("Iniciando modelo híbrido GFOM - Gradient Boosting")

    train = build_train(cfg)

    log.info("Base de treino criada com %s linhas.", len(train))

    backtest_pred, backtest_metrics = run_walk_forward_backtest_gb(
        train,
        cfg,
    )

    metricas_subsistema = calcular_metricas_por_subsistema(
        backtest_pred,
        cfg,
    )

    model, features, base_ml = train_gradient_boosting(
        train,
        cfg,
    )

    importance_df = compute_feature_importance(
        model,
        base_ml,
        features,
    )

    df_proj = build_predict_base(cfg)

    proj = apply_projection(
        df_proj,
        model,
        features,
    )

    out_final = finalize_output(
        proj,
        cfg,
    )

    save_outputs(
        out_final=out_final,
        backtest_pred=backtest_pred,
        backtest_metrics=backtest_metrics,
        metricas_subsistema=metricas_subsistema,
        importance_df=importance_df,
        cfg=cfg,
    )

    log.info("Métricas GB:")
    log.info("\n%s", backtest_metrics)

    log.info("Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()