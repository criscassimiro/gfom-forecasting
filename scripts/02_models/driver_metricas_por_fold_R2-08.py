# -*- coding: utf-8 -*-
"""
Driver para checar granularidade de "fold" no cálculo de média/desvio-padrão
para a Tabela 5 (R2-08). Não altera a lógica de treino/validação de
Metricas_proj_gfom_RF.py — só reusa build_train() e run_holdout_backtest()
tal como estão, e adiciona agregação por mês e por ano sobre o mesmo
backtest_pred já produzido pelo script original.
"""
import numpy as np
import pandas as pd

import Metricas_proj_gfom_RF_run as m

cfg = m.CFG

print("Carregando bases e construindo treino...")
train = m.build_train(cfg)
print(f"train shape: {train.shape}, meses: {train['mes_dt'].nunique()}, subsistemas: {sorted(train['subsistema'].unique())}")

print("Rodando backtest walk-forward (idêntico ao script original)...")
backtest_pred, backtest_metrics = m.run_holdout_backtest(train, cfg)
print("\n=== Métricas agregadas (pooled, N=192) — deve bater com a Tabela 5 publicada ===")
print(backtest_metrics.to_string(index=False))

base_eval = m.filtrar_periodo_validacao(backtest_pred)
print(f"\nbase_eval shape: {base_eval.shape}")

# ---------------------------------------------------------------
# Fold = MÊS (todos os subsistemas juntos naquele mês, n=4 por fold)
# ---------------------------------------------------------------
def metricas_por_mes(df):
    resultados = []
    for mes, base in df.groupby("mes_dt", dropna=False):
        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]
        met = m.calcular_metricas_completas(y_true, y_pred)
        resultados.append({"mes_dt": mes, "n_obs": len(base), **met})
    return pd.DataFrame(resultados).sort_values("mes_dt").reset_index(drop=True)

met_mes = metricas_por_mes(base_eval)
met_mes.to_csv("metricas_por_mes.csv", index=False, sep=";")

# ---------------------------------------------------------------
# Fold = ANO (todos os subsistemas e todos os meses daquele ano, n=48 por fold)
# ---------------------------------------------------------------
def metricas_por_ano(df):
    d = df.copy()
    d["ano"] = pd.to_datetime(d["mes_dt"]).dt.year
    resultados = []
    for ano, base in d.groupby("ano", dropna=False):
        y_true = base["gfom_hist_mwmed"]
        y_pred = base["gfom_pred_mwmed"]
        met = m.calcular_metricas_completas(y_true, y_pred)
        resultados.append({"ano": ano, "n_obs": len(base), **met})
    return pd.DataFrame(resultados).sort_values("ano").reset_index(drop=True)

met_ano = metricas_por_ano(base_eval)
met_ano.to_csv("metricas_por_ano.csv", index=False, sep=";")

cols_relevantes = ["mae", "rmse", "r2", "precision", "recall", "f1"]

print("\n=== Fold = MÊS (n=4/fold, 48 folds) ===")
print(met_mes[["mes_dt", "n_obs"] + cols_relevantes].to_string(index=False))
n_nan_r2_mes = met_mes["r2"].isna().sum()
print(f"\nNúmero de meses com R2 = NaN: {n_nan_r2_mes} de {len(met_mes)}")
print("\nMédia ± desvio-padrão entre os 48 folds mensais:")
for c in cols_relevantes:
    vals = met_mes[c].dropna()
    print(f"  {c}: mean={vals.mean():.4f}  std={vals.std():.4f}  (n_validos={len(vals)}/{len(met_mes)})")

print("\n=== Fold = ANO (n=48/fold, 4 folds) ===")
print(met_ano[["ano", "n_obs"] + cols_relevantes].to_string(index=False))
n_nan_r2_ano = met_ano["r2"].isna().sum()
print(f"\nNúmero de anos com R2 = NaN: {n_nan_r2_ano} de {len(met_ano)}")
print("\nMédia ± desvio-padrão entre os 4 folds anuais:")
for c in cols_relevantes:
    vals = met_ano[c].dropna()
    print(f"  {c}: mean={vals.mean():.4f}  std={vals.std():.4f}  (n_validos={len(vals)}/{len(met_ano)})")

print("\nOK - CSVs salvos: metricas_por_mes.csv, metricas_por_ano.csv")
