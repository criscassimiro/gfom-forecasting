# -*- coding: utf-8 -*-
"""
PROJEÇÃO HÍBRIDA DE GFOM POR SUBSISTEMA
- Gatilho operacional: EAR_subsistema <= CRef amarela
- Magnitude: RandomForestRegressor
- Backtest walk-forward
- Validação principal: 2022-2025
- Saída final em um único arquivo Excel
- Geração automática de plots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
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

    # Backtest
    min_train_months: int = 1

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


def get_plot_dir(cfg: Config) -> Path:
    return ensure_dir(cfg.out_dir / "plots")


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

    return float(100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


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
    df = df[df["mes_dt"].dt.year.between(2021, 2025)].reset_index(drop=True)
    return df


# =============================================================================
# GFOM E GT HISTÓRICAS
# =============================================================================
def load_gfom_historico(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.gt_historico_csv, sep=";", low_memory=False)
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce", utc=True)
    df["data_hora"] = df["data_hora"].dt.tz_localize(None)
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
    df = pd.read_csv(cfg.ear_2026_csv, sep=";", low_memory=False)
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
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
    df = pd.read_csv(cfg.gt_2026_csv, sep=";", low_memory=False)

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
    base = train.copy()

    base = base.dropna(subset=[
        "ear_pct_sub",
        "cref_amarela",
        "gap_below",
        "intensidade",
        "gt_hist_mwmed",
        "mes_num",
        "mes_sin",
        "mes_cos",
        "share_hist",
    ]).copy()

    if base.empty:
        raise ValueError("A base de treino ficou vazia após o tratamento.")

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


# =============================================================================
# MÉTRICAS
# =============================================================================
def calcular_metricas_classificacao(y_true_cont, y_pred_cont) -> dict:
    y_true_bin = (pd.Series(y_true_cont).fillna(0) > 0).astype(int)
    y_pred_bin = (pd.Series(y_pred_cont).fillna(0) > 0).astype(int)

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


def calcular_metricas_evento(y_true, y_pred, threshold_evento: float = 0.0) -> dict:
    y_true = pd.Series(y_true).fillna(0).astype(float)
    y_pred = pd.Series(y_pred).fillna(0).astype(float)

    mask = y_true > threshold_evento

    if mask.sum() == 0:
        return {
            "n_evento": 0,
            "mae_evento": np.nan,
            "rmse_evento": np.nan,
            "wape_evento_percent": np.nan,
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    return {
        "n_evento": int(mask.sum()),
        "mae_evento": mean_absolute_error(yt, yp),
        "rmse_evento": rmse(yt, yp),
        "wape_evento_percent": wape(yt, yp),
    }


def calcular_metricas_zero(y_true, y_pred, threshold_evento: float = 0.0) -> dict:
    y_true = pd.Series(y_true).fillna(0).astype(float)
    y_pred = pd.Series(y_pred).fillna(0).astype(float)

    mask = y_true <= threshold_evento

    if mask.sum() == 0:
        return {
            "n_zero": 0,
            "mae_zero": np.nan,
            "false_positive_count": np.nan,
            "false_positive_rate": np.nan,
            "pred_mean_zero": np.nan,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    false_positive_count = int((yp > threshold_evento).sum())

    return {
        "n_zero": int(mask.sum()),
        "mae_zero": mean_absolute_error(yt, yp),
        "false_positive_count": false_positive_count,
        "false_positive_rate": float(false_positive_count / mask.sum()),
        "pred_mean_zero": float(yp.mean()),
    }


def calcular_metricas_completas(y_true, y_pred, threshold_evento: float = 0.0) -> dict:
    resultados = {}
    resultados.update(calcular_metricas_regressao(y_true, y_pred))
    resultados.update(calcular_metricas_classificacao(y_true, y_pred))
    resultados.update(calcular_metricas_evento(y_true, y_pred, threshold_evento))
    resultados.update(calcular_metricas_zero(y_true, y_pred, threshold_evento))
    return resultados


# =============================================================================
# BACKTEST
# =============================================================================
def run_holdout_backtest(train: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = train.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True).copy()

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
            model, features, _ = train_random_forest(treino, cfg)
        except Exception as e:
            log.warning("Falha ao treinar para o mês %s: %s", mes_teste, e)
            continue

        pred = teste.copy()
        pred["share_pred"] = 0.0

        mask = pred["gatilho"] == 1
        if mask.any():
            pred.loc[mask, "share_pred"] = model.predict(pred.loc[mask, features])

        pred["share_pred"] = pred["share_pred"].clip(lower=0.0, upper=1.0)
        pred["gfom_pred_mwmed"] = pred["share_pred"] * pred["gt_hist_mwmed"]
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

    metricas = calcular_metricas_completas(y_true, y_pred)

    metrics = pd.DataFrame([{
        "periodo": "2022_2025",
        "n_obs": len(base_eval),
        "dt_inicio_backtest": base_eval["mes_dt"].min(),
        "dt_fim_backtest": base_eval["mes_dt"].max(),
        **metricas,
    }])

    return backtest_pred, metrics


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


def calcular_metricas_periodos(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    df = backtest_pred.copy()
    df["mes_dt"] = pd.to_datetime(df["mes_dt"], errors="coerce")

    periodos = {
        "2022_2025": ("2022-01-01", "2025-12-31"),
    }

    resultados = []

    for nome_periodo, (dt_ini, dt_fim) in periodos.items():
        base = df[
            (df["mes_dt"] >= pd.Timestamp(dt_ini)) &
            (df["mes_dt"] <= pd.Timestamp(dt_fim))
        ].copy()

        if base.empty:
            continue

        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]

        metricas = calcular_metricas_completas(y_true, y_pred)

        resultados.append({
            "periodo": nome_periodo,
            "inicio": dt_ini,
            "fim": dt_fim,
            "n_obs": len(base),
            **metricas,
        })

    return pd.DataFrame(resultados)


def calcular_metricas_por_subsistema(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred)

    if df.empty:
        return pd.DataFrame()

    resultados = []

    for subsistema, base in df.groupby("subsistema", dropna=False):
        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]

        metricas = calcular_metricas_completas(y_true, y_pred)

        resultados.append({
            "subsistema": subsistema,
            "inicio": "2022-01-01",
            "fim": "2025-12-31",
            "n_obs": len(base),
            **metricas,
        })

    return pd.DataFrame(resultados).sort_values("subsistema").reset_index(drop=True)


def calcular_metricas_por_subsistema_ano(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred)

    if df.empty:
        return pd.DataFrame()

    df["ano"] = df["mes_dt"].dt.year
    resultados = []

    for (subsistema, ano), base in df.groupby(["subsistema", "ano"], dropna=False):
        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]

        metricas = calcular_metricas_completas(y_true, y_pred)

        resultados.append({
            "subsistema": subsistema,
            "ano": ano,
            "n_obs": len(base),
            **metricas,
        })

    return pd.DataFrame(resultados).sort_values(["subsistema", "ano"]).reset_index(drop=True)


def calcular_metricas_por_rodada(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty or "id_rodada" not in backtest_pred.columns:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred)

    if df.empty:
        return pd.DataFrame()

    resultados = []

    for id_rodada, base in df.groupby("id_rodada", dropna=False):
        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]

        metricas = calcular_metricas_completas(y_true, y_pred)

        resultados.append({
            "id_rodada": id_rodada,
            "n_obs": len(base),
            **metricas,
        })

    return pd.DataFrame(resultados).sort_values("id_rodada").reset_index(drop=True)


def calcular_metricas_por_rodada_subsistema(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    if "id_rodada" not in backtest_pred.columns or "subsistema" not in backtest_pred.columns:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred)

    if df.empty:
        return pd.DataFrame()

    resultados = []

    for (id_rodada, subsistema), base in df.groupby(["id_rodada", "subsistema"], dropna=False):
        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]

        metricas = calcular_metricas_completas(y_true, y_pred)

        resultados.append({
            "id_rodada": id_rodada,
            "subsistema": subsistema,
            "n_obs": len(base),
            **metricas,
        })

    return (
        pd.DataFrame(resultados)
        .sort_values(["id_rodada", "subsistema"])
        .reset_index(drop=True)
    )


def calcular_metricas_por_faixa(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred)

    if df.empty:
        return pd.DataFrame()

    bins = [-np.inf, 0, 100, 500, np.inf]
    labels = ["zero", "0_100", "100_500", "gt_500"]

    df["faixa_gfom"] = pd.cut(
        df["gfom_hist_mwmed"],
        bins=bins,
        labels=labels,
        right=True
    )

    resultados = []

    for faixa, base in df.groupby("faixa_gfom", dropna=False, observed=True):
        if base.empty:
            continue

        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]

        metricas = calcular_metricas_completas(y_true, y_pred)

        resultados.append({
            "faixa_gfom": faixa,
            "n_obs": len(base),
            **metricas,
        })

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).reset_index(drop=True)


# =============================================================================
# PLOTS
# =============================================================================
def build_backtest_plot_base(backtest_pred: pd.DataFrame) -> pd.DataFrame:
    if backtest_pred.empty:
        return pd.DataFrame()

    df = filtrar_periodo_validacao(backtest_pred).copy()
    if df.empty:
        return df

    df["mes_dt"] = pd.to_datetime(df["mes_dt"], errors="coerce")
    df["erro"] = df["gfom_pred_mwmed"] - df["gfom_hist_mwmed"]
    df["erro_abs"] = df["erro"].abs()
    df["real_bin"] = (df["gfom_hist_mwmed"] > 0).astype(int)
    df["pred_bin"] = (df["gfom_pred_mwmed"] > 0).astype(int)

    return df


def plot_backtest_sin(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty:
        return

    base = (
        df.groupby("mes_dt", as_index=False)
          .agg(
              gfom_real=("gfom_hist_mwmed", "sum"),
              gfom_prev=("gfom_pred_mwmed", "sum")
          )
          .sort_values("mes_dt")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(base["mes_dt"], base["gfom_real"], label="Observed GFOM")
    ax.plot(base["mes_dt"], base["gfom_prev"], label="Predicted GFOM")

    ax.set_title("GFOM Backtest - SIN")
    ax.set_xlabel("Month")
    ax.set_ylabel("MWmed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "backtest_sin_real_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_subsistemas(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty or "subsistema" not in df.columns:
        return

    for subsistema, base in df.groupby("subsistema", dropna=False):
        base_plot = (
            base.groupby("mes_dt", as_index=False)
                .agg(
                    gfom_real=("gfom_hist_mwmed", "sum"),
                    gfom_prev=("gfom_pred_mwmed", "sum")
                )
                .sort_values("mes_dt")
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(base_plot["mes_dt"], base_plot["gfom_real"], label="Observed GFOM")
        ax.plot(base_plot["mes_dt"], base_plot["gfom_prev"], label="Predicted GFOM")

        ax.set_title(f"GFOM Backtest - {subsistema}")
        ax.set_xlabel("Month")
        ax.set_ylabel("MWmed")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        nome = f"backtest_{str(subsistema).lower()}_real_vs_pred.png"
        fig.savefig(get_plot_dir(cfg) / nome, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_real_vs_prev(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty:
        return

    x = df["gfom_hist_mwmed"].astype(float)
    y = df["gfom_pred_mwmed"].astype(float)

    lim_max = max(float(x.max()), float(y.max())) if len(df) else 1.0
    lim_max = max(lim_max, 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.7)

    ax.plot([0, lim_max], [0, lim_max], linestyle="--")
    ax.set_title("Observed vs Predicted GFOM")
    ax.set_xlabel("Observed GFOM (MWmed)")
    ax.set_ylabel("Predicted GFOM (MWmed)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "scatter_real_vs_pred.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metricas_subsistema(metricas_subsistema: pd.DataFrame, cfg: Config) -> None:
    if metricas_subsistema.empty or "subsistema" not in metricas_subsistema.columns:
        return

    metricas = [
        ("mae", "MAE by Subsystem", "MAE"),
        ("rmse", "RMSE by Subsystem", "RMSE"),
        ("wape_percent", "WAPE by Subsystem", "WAPE (%)"),
        ("f1", "F1-score by Subsystem", "F1-score"),
    ]

    df = metricas_subsistema.copy().sort_values("subsistema")

    for col, titulo, ylabel in metricas:
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["subsistema"].astype(str), df[col].astype(float))

        ax.set_title(titulo)
        ax.set_xlabel("Subsystem")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(get_plot_dir(cfg) / f"metricas_subsistema_{col}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_confusion_matrix(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty:
        return

    tn = int(((df["real_bin"] == 0) & (df["pred_bin"] == 0)).sum())
    fp = int(((df["real_bin"] == 0) & (df["pred_bin"] == 1)).sum())
    fn = int(((df["real_bin"] == 1) & (df["pred_bin"] == 0)).sum())
    tp = int(((df["real_bin"] == 1) & (df["pred_bin"] == 1)).sum())

    matriz = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matriz)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Event", "Event"])
    ax.set_yticklabels(["No Event", "Event"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, cfg: Config) -> None:
    if importance_df.empty:
        return

    df = importance_df.copy().sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature"].astype(str), df["importance_mean"].astype(float))

    ax.set_title("Feature Importance")
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_erro_absoluto_mensal(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty:
        return

    base = (
        df.groupby("mes_dt", as_index=False)
          .agg(erro_abs=("erro_abs", "mean"))
          .sort_values("mes_dt")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(base["mes_dt"], base["erro_abs"])
    ax.set_title("Absolute Error by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Absolute Error (MWmed)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "erro_absoluto_mensal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histograma_erro(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["erro"].astype(float), bins=30)
    ax.set_title("Error Distribution")
    ax.set_xlabel("Prediction Error (MWmed)")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "histograma_erro.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_erro_subsistema(backtest_pred: pd.DataFrame, cfg: Config) -> None:
    df = build_backtest_plot_base(backtest_pred)
    if df.empty or "subsistema" not in df.columns:
        return

    subs = sorted(df["subsistema"].dropna().astype(str).unique())
    data = [df.loc[df["subsistema"].astype(str) == s, "erro"].astype(float).values for s in subs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=subs)
    ax.set_title("Error by Subsystem")
    ax.set_xlabel("Subsystem")
    ax.set_ylabel("Prediction Error (MWmed)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(get_plot_dir(cfg) / "boxplot_erro_subsistema.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    backtest_pred: pd.DataFrame,
    metricas_subsistema: pd.DataFrame,
    importance_df: pd.DataFrame,
    cfg: Config,
) -> None:
    plot_backtest_sin(backtest_pred, cfg)
    plot_backtest_subsistemas(backtest_pred, cfg)
    plot_scatter_real_vs_prev(backtest_pred, cfg)
    plot_metricas_subsistema(metricas_subsistema, cfg)
    plot_confusion_matrix(backtest_pred, cfg)
    plot_feature_importance(importance_df, cfg)
    plot_erro_absoluto_mensal(backtest_pred, cfg)
    plot_histograma_erro(backtest_pred, cfg)
    plot_boxplot_erro_subsistema(backtest_pred, cfg)

    log.info("Gráficos salvos em: %s", get_plot_dir(cfg))


# =============================================================================
# PROJEÇÃO FINAL
# =============================================================================
def apply_projection(df_proj: pd.DataFrame, model, features: list[str]) -> pd.DataFrame:
    out = df_proj.copy()

    # compatibiliza a feature GT do treino com a projeção
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
    metricas_periodos: pd.DataFrame,
    metricas_subsistema: pd.DataFrame,
    metricas_subsistema_ano: pd.DataFrame,
    metricas_rodada: pd.DataFrame,
    metricas_rodada_subsistema: pd.DataFrame,
    metricas_faixa: pd.DataFrame,
    cfg: Config,
) -> None:
    out_dir = ensure_dir(cfg.out_dir)
    arq_excel = out_dir / "resultados_gfom_rf.xlsx"

    out_final = out_final.copy()
    backtest_pred = backtest_pred.copy()
    backtest_metrics = backtest_metrics.copy()
    importance_df = importance_df.copy()
    metricas_periodos = metricas_periodos.copy()
    metricas_subsistema = metricas_subsistema.copy()
    metricas_subsistema_ano = metricas_subsistema_ano.copy()
    metricas_rodada = metricas_rodada.copy()
    metricas_rodada_subsistema = metricas_rodada_subsistema.copy()
    metricas_faixa = metricas_faixa.copy()

    if not out_final.empty and "gfom_mwmed" in out_final.columns:
        out_final["gfom_mwmed"] = pd.to_numeric(
            out_final["gfom_mwmed"], errors="coerce"
        ).fillna(0.0).round(2)

    if not backtest_pred.empty and "mes_dt" in backtest_pred.columns:
        backtest_pred["mes_dt"] = pd.to_datetime(backtest_pred["mes_dt"], errors="coerce")

    if not backtest_metrics.empty:
        if "dt_inicio_backtest" in backtest_metrics.columns:
            backtest_metrics["dt_inicio_backtest"] = pd.to_datetime(
                backtest_metrics["dt_inicio_backtest"], errors="coerce"
            )
        if "dt_fim_backtest" in backtest_metrics.columns:
            backtest_metrics["dt_fim_backtest"] = pd.to_datetime(
                backtest_metrics["dt_fim_backtest"], errors="coerce"
            )

    with pd.ExcelWriter(arq_excel, engine="openpyxl") as writer:
        out_final.to_excel(writer, sheet_name="projecao_final", index=False)

        if not backtest_pred.empty:
            backtest_pred.to_excel(writer, sheet_name="backtest_predicoes", index=False)

        if not backtest_metrics.empty:
            backtest_metrics.to_excel(writer, sheet_name="backtest_metricas", index=False)

        if not metricas_periodos.empty:
            metricas_periodos.to_excel(writer, sheet_name="metricas_periodos", index=False)

        if not metricas_subsistema.empty:
            metricas_subsistema.to_excel(writer, sheet_name="metricas_subsistema", index=False)

        if not metricas_subsistema_ano.empty:
            metricas_subsistema_ano.to_excel(writer, sheet_name="metricas_subs_ano", index=False)

        if not metricas_rodada.empty:
            metricas_rodada.to_excel(writer, sheet_name="metricas_rodada", index=False)

        if not metricas_rodada_subsistema.empty:
            metricas_rodada_subsistema.to_excel(writer, sheet_name="metricas_rod_subs", index=False)

        if not metricas_faixa.empty:
            metricas_faixa.to_excel(writer, sheet_name="metricas_faixa", index=False)

        if not importance_df.empty:
            importance_df.to_excel(writer, sheet_name="feature_importance", index=False)

    log.info("Arquivo Excel salvo em: %s", arq_excel)


# =============================================================================
# MAIN
# =============================================================================
def main(cfg: Config = CFG) -> None:
    log.info("Iniciando projeção híbrida de GFOM por subsistema")

    # base histórica
    train = build_train(cfg)

    # backtest walk-forward
    backtest_pred, backtest_metrics = run_holdout_backtest(train, cfg)

    # métricas válidas para artigo
    metricas_periodos = calcular_metricas_periodos(backtest_pred)
    metricas_subsistema = calcular_metricas_por_subsistema(backtest_pred)
    metricas_subsistema_ano = calcular_metricas_por_subsistema_ano(backtest_pred)
    metricas_rodada = calcular_metricas_por_rodada(backtest_pred)
    metricas_rodada_subsistema = calcular_metricas_por_rodada_subsistema(backtest_pred)
    metricas_faixa = calcular_metricas_por_faixa(backtest_pred)

    # treino final com toda a base histórica
    model, features, base_ml = train_random_forest(train, cfg)

    # importância das variáveis
    importance_df = compute_feature_importance(model, base_ml, features)

    # gerar gráficos
    generate_plots(
        backtest_pred=backtest_pred,
        metricas_subsistema=metricas_subsistema,
        importance_df=importance_df,
        cfg=cfg,
    )

    # projeção 2026
    df_proj = build_predict_base(cfg)
    proj = apply_projection(df_proj, model, features)

    # saída final
    out_final = finalize_output(proj, cfg)

    # salvar tudo em um único Excel
    save_outputs(
        out_final,
        backtest_pred,
        backtest_metrics,
        importance_df,
        metricas_periodos,
        metricas_subsistema,
        metricas_subsistema_ano,
        metricas_rodada,
        metricas_rodada_subsistema,
        metricas_faixa,
        cfg,
    )

    log.info("Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()