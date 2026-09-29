# -*- coding: utf-8 -*-
"""
Driver para R3-12 (baseline nao-data-driven baseado em regra EAR-CRef).
Reusa as previsoes pooled ja salvas (backtest_pred_pooled.csv, gerado por
driver_regime.py a partir de build_train()+run_holdout_backtest(), sem
alterar a logica de treino/validacao) para nao precisar re-treinar o RF.

Regra nao-data-driven avaliada: gatilho = 1 se ear_fim_per <= cref_amarela
(coluna 'gatilho' ja existe no pipeline original como feature de entrada do
RF -- aqui ela e avaliada como CLASSIFICADOR standalone, nos mesmos 192
folds de teste da Tabela 5/4, para comparar com o desempenho do RF.
"""
import pandas as pd
import Metricas_proj_gfom_RF_run as m

base_eval = pd.read_csv("backtest_pred_pooled.csv", sep=";")
print("shape:", base_eval.shape)
print("colunas:", base_eval.columns.tolist())

# ---- RF (ja publicado, recalculado aqui so para conferencia) ----
met_rf = m.calcular_metricas_completas(base_eval["gfom_hist_mwmed"], base_eval["gfom_pred_mwmed"])

# ---- Baseline nao-data-driven: regra EAR <= CRef_amarela ----
met_regra = m.calcular_metricas_completas(base_eval["gfom_hist_mwmed"], base_eval["gatilho"])

cols = ["precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]
comparacao = pd.DataFrame(
    [
        {"modelo": "Random Forest (publicado)", **{c: met_rf[c] for c in cols}},
        {"modelo": "Regra EAR<=CRef_amarela (baseline)", **{c: met_regra[c] for c in cols}},
    ]
)
print("\n=== Comparacao RF vs baseline nao-data-driven (N=192, mesmos folds) ===")
print(comparacao.to_string(index=False))

comparacao.to_csv("comparacao_baseline_R3-12.csv", index=False, sep=";")
print("\nOK - salvo: comparacao_baseline_R3-12.csv")

# taxa de eventos real (para contextualizar precision/recall de um classificador trivial)
taxa_evento = (base_eval["gfom_hist_mwmed"] > 0).mean()
print(f"\nTaxa real de ocorrencia de GFOM na amostra de teste: {taxa_evento:.4f} ({int((base_eval['gfom_hist_mwmed']>0).sum())} de {len(base_eval)})")
