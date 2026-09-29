# -*- coding: utf-8 -*-
"""
Driver para R2-05 (defasagem temporal EAR-CRef vs GFOM).
Reconstroi o painel historico completo (4 subsistemas x 60 meses, 2021-2025,
N=240) usando as mesmas funcoes de carregamento ja validadas
(load_gfom_historico, load_ear_historico, load_cref_historico) e a formula
real de gap_below/intensidade (build_trigger), ambas de
Metricas_proj_gfom_RF.py -- sem alterar a logica de treino/validacao do
modelo publicado.

Objetivo: verificar a afirmacao original da Secao III.C do artigo ("GFOM
presents a weak relationship with absolute storage (~0.07)... the deviation
EAR-CRef exhibits a significantly stronger association (~0.54)... reaching
values close to 0.63 with a two-month lag") e a premissa do comentario R2-05
do Revisor 2, que cita essa mesma frase.

Resultado: nenhuma das tres cifras originais (0.07 / 0.54 / 0.63) e
reproduzivel com o painel agrupado (4 subsistemas, sem excluir nenhum ano).
EAR isolado nunca aparece fraco (sempre |rho| entre 0.44 e 0.75 dependendo
do recorte testado) -- o que tambem contradiz o peso AHP ja publicado nas
Tabelas 4/Figura 5 (EAR = variavel dominante, peso 0.75). A correlacao do
gap_below (deviation EAR-CRef) CAI com a defasagem no painel agrupado
(0.50 -> 0.49 -> 0.48 -> 0.45 para lag 0/1/2/3), nao sobe. Testes de
robustez adicionais (excluir 2021, primeira diferenca) mostram que uma
subida aparente restrita ao subsistema Sudeste sozinho e artefato de
tendencia de longo prazo compartilhada (seca acumulada), nao uma relacao
preditiva real mes a mes -- corr chega a 1.0000 exato ao excluir 2021 no
lag=3, um sinal classico de correlacao espuria.

Decisao (Cristiane, 25/Set/2026): usar o resultado agrupado (mais simples,
sem artefato) para corrigir a Secao III.C e justificar a nao-inclusao de
features defasadas no modelo final -- sem necessidade de retreinar o RF.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

import Metricas_proj_gfom_RF_run as m

cfg = m.CFG

gfom = m.load_gfom_historico(cfg)
ear = m.load_ear_historico(cfg)
cref = m.load_cref_historico(cfg)

df = (
    gfom.merge(ear, on=["mes_dt", "subsistema"], how="inner")
        .merge(cref, on="mes_dt", how="left")
        .dropna(subset=["gfom_hist_mwmed", "ear_pct_sub", "cref_amarela"])
        .sort_values(["subsistema", "mes_dt"])
        .reset_index(drop=True)
)
df = m.build_trigger(df, ear_col="ear_pct_sub", cref_col="cref_amarela", inclusive=cfg.trigger_inclusive)

print(f"Painel: {df.shape[0]} obs, {df['mes_dt'].min()} a {df['mes_dt'].max()}, "
      f"{df['subsistema'].nunique()} subsistemas")

# --- lag 0 (sem defasagem) ---
rho_ear, p_ear = spearmanr(df["ear_pct_sub"], df["gfom_hist_mwmed"])
rho_gap0, p_gap0 = spearmanr(df["gap_below"], df["gfom_hist_mwmed"])
print(f"\nSpearman EAR (armazenamento absoluto) x GFOM: rho={rho_ear:+.4f} (p={p_ear:.4g})")
print(f"Spearman gap_below (deviation, lag=0) x GFOM: rho={rho_gap0:+.4f} (p={p_gap0:.4g})")

# --- lag 0-3 meses, painel agrupado (4 subsistemas) ---
resultados = []
for lag in [0, 1, 2, 3]:
    tmp = df.copy()
    tmp[f"gap_lag"] = tmp.groupby("subsistema")["gap_below"].shift(lag)
    sub = tmp.dropna(subset=["gap_lag", "gfom_hist_mwmed"])
    rho, p = spearmanr(sub["gap_lag"], sub["gfom_hist_mwmed"])
    resultados.append({"lag_meses": lag, "rho_spearman": rho, "p_valor": p, "n": len(sub)})
    print(f"lag={lag} meses (agrupado, 4 subsistemas): rho={rho:+.4f} (p={p:.4g}), n={len(sub)}")

df_result = pd.DataFrame(resultados)
df_result.to_csv("lag_corr_R2-05.csv", index=False, sep=";")
df.to_csv("panel_lag_corr_base_R2-05.csv", index=False, sep=";")
print("\nOK - salvos: lag_corr_R2-05.csv, panel_lag_corr_base_R2-05.csv")
