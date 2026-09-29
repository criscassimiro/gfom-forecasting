# -*- coding: utf-8 -*-
"""
Driver para R2-09 (desempenho estratificado por regime) e para salvar as
previsões pooled (base_eval) reutilizáveis por outros itens da revisão.
Reusa build_train() e run_holdout_backtest() de Metricas_proj_gfom_RF.py sem
alterar a logica de treino/validacao.

Regime (normal/alerta/critica) replicado EXATAMENTE da funcao classificar_zona
em EAR_CREF_Carga20212026.py (script de feature engineering, ja usado para
gerar a Figura 8 do artigo):
    critica: ear_fim_per <= cref_vermelha
    alerta:  ear_fim_per <= cref_amarela (e nao critica)
    normal:  caso contrario
onde ear_fim_per e a media mensal do EAR% POR SUBSISTEMA, e cref_amarela/
cref_vermelha sao as curvas de referencia nacionais (SIN), repetidas para
todos os subsistemas no mesmo mes/ano.
"""
import numpy as np
import pandas as pd

import Metricas_proj_gfom_RF_run as m

cfg = m.CFG

print("Carregando bases e construindo treino...")
train = m.build_train(cfg)

print("Rodando backtest walk-forward (identico ao script original)...")
backtest_pred, backtest_metrics = m.run_holdout_backtest(train, cfg)
print("\n=== Metricas agregadas (pooled, N=192) - deve bater com a Tabela 5 publicada ===")
print(backtest_metrics.to_string(index=False))

base_eval = m.filtrar_periodo_validacao(backtest_pred)
base_eval["mes_dt"] = pd.to_datetime(base_eval["mes_dt"])
base_eval["ano"] = base_eval["mes_dt"].dt.year
base_eval["mes"] = base_eval["mes_dt"].dt.month
print(f"\nbase_eval shape: {base_eval.shape}")
print("colunas base_eval:", base_eval.columns.tolist())

# salva as previsoes pooled cruas para reuso em outros itens (R3-12, R3-11, R2-05, R4-06)
base_eval.to_csv("backtest_pred_pooled.csv", index=False, sep=";")
print("Salvo: backtest_pred_pooled.csv")

# ---------------------------------------------------------------
# EAR por subsistema-mes (mesma agregacao do script de feature engineering)
# ---------------------------------------------------------------
ear = pd.read_csv("resultados/ear_historico.csv", sep=";")
ear["data"] = pd.to_datetime(ear["data"], errors="coerce", utc=True).dt.tz_localize(None)
ear["ear_fim_per"] = pd.to_numeric(ear["ear_fim_per"], errors="coerce")
ear = ear.dropna(subset=["data", "subsistema", "ear_fim_per"]).copy()
ear["ano"] = ear["data"].dt.year
ear["mes"] = ear["data"].dt.month
ear["subsistema"] = ear["subsistema"].astype(str).str.strip()
ear_mensal = (
    ear.groupby(["subsistema", "ano", "mes"], as_index=False)["ear_fim_per"]
    .mean()
)

# ---------------------------------------------------------------
# CRef nacional (amarela e vermelha)
# ---------------------------------------------------------------
cref = pd.read_excel("resultados/cref_2021_2026.xlsx")
cref["data"] = pd.to_datetime(cref["data"])
cref["ano"] = cref["data"].dt.year
cref["mes"] = cref["data"].dt.month
cref_tratada = cref[["ano", "mes", "curva_amarela", "curva_vermelha"]].rename(
    columns={"curva_amarela": "cref_amarela_chk", "curva_vermelha": "cref_vermelha"}
)

# ---------------------------------------------------------------
# Merge com base_eval e classificacao de zona (regime)
# ---------------------------------------------------------------
base_regime = base_eval.merge(ear_mensal, on=["subsistema", "ano", "mes"], how="left")
base_regime = base_regime.merge(cref_tratada, on=["ano", "mes"], how="left")

n_missing = base_regime["ear_fim_per"].isna().sum()
print(f"\nLinhas sem ear_fim_per apos merge: {n_missing} de {len(base_regime)}")
n_missing_cref = base_regime["cref_vermelha"].isna().sum()
print(f"Linhas sem cref_vermelha apos merge: {n_missing_cref} de {len(base_regime)}")

# checagem de consistencia: cref_amarela ja existente no train (se houver) deve bater com cref_amarela_chk
if "cref_amarela" in base_regime.columns:
    diff = (base_regime["cref_amarela"] - base_regime["cref_amarela_chk"]).abs()
    print(f"Maior divergencia entre cref_amarela do pipeline original e do arquivo de referencia: {diff.max():.6f}")

def classificar_zona(row):
    if pd.isna(row["ear_fim_per"]) or pd.isna(row["cref_vermelha"]) or pd.isna(row["cref_amarela_chk"]):
        return np.nan
    if row["ear_fim_per"] <= row["cref_vermelha"]:
        return "critical"
    elif row["ear_fim_per"] <= row["cref_amarela_chk"]:
        return "alert"
    else:
        return "normal"

base_regime["regime"] = base_regime.apply(classificar_zona, axis=1)
print("\nContagem de observacoes por regime:")
print(base_regime["regime"].value_counts(dropna=False))

base_regime.to_csv("backtest_pred_com_regime.csv", index=False, sep=";")
print("Salvo: backtest_pred_com_regime.csv")

# ---------------------------------------------------------------
# Metricas de classificacao (Precision/Recall/F1) por regime
# ---------------------------------------------------------------
resultados = []
for regime, base in base_regime.groupby("regime", dropna=False):
    y_true = base["gfom_hist_mwmed"]
    y_pred = base["gfom_pred_mwmed"]
    met = m.calcular_metricas_completas(y_true, y_pred)
    resultados.append({"regime": regime, "n_obs": len(base), **met})

df_result = pd.DataFrame(resultados)
ordem = {"normal": 0, "alert": 1, "critical": 2}
df_result["ordem"] = df_result["regime"].map(ordem)
df_result = df_result.sort_values("ordem").drop(columns="ordem")

cols_relevantes = ["regime", "n_obs", "precision", "recall", "f1", "mae", "rmse", "r2"]
print("\n=== Desempenho por regime (normal / alert / critical) ===")
print(df_result[cols_relevantes].to_string(index=False))

df_result.to_csv("metricas_por_regime_R2-09.csv", index=False, sep=";")
print("\nOK - salvo: metricas_por_regime_R2-09.csv")
