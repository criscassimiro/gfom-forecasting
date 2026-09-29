# -*- coding: utf-8 -*-
"""
Driver para R2-10 (baseline simples: regressao logistica para ocorrencia de GFOM).

Compara, no mesmo protocolo walk-forward e mesma janela de validacao
(2022-2025, N=192) ja usados para a Tabela 5/6 publicada, um classificador
de regressao logistica treinado diretamente sobre as 8 features do pipeline
(get_feature_cols(), Metricas_proj_gfom_RF_local.py) prevendo ocorrencia
binaria (share_hist > 0), SEM o gating pelo gatilho EAR<=CRef que o
RandomForestRegressor publicado usa (o RF so preve valor >0 para meses com
gatilho==1; fora disso forca 0 -- por isso a Tabela 6 mostra RF identico a
regra fisica). A regressao logistica aqui decide por si mesma, a partir do
conjunto completo de features (que inclui gap_below e intensidade, ja
derivadas do gatilho), se cada mes e ocorrencia ou nao.

Reusa build_train, filter_ml_train, get_feature_cols, filtrar_periodo_validacao
de Metricas_proj_gfom_RF_local.py -- sem alterar nenhuma logica ja publicada.
"""
import sys
sys.path.insert(0, "/sessions/rcw-015oxuvxnzz53lzq1dsf1hgu/mnt/artigo GFOM")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import Metricas_proj_gfom_RF_local as m

cfg = m.CFG
FEATURES = m.get_feature_cols()


def run_holdout_backtest_logistic(train: pd.DataFrame, cfg, class_weight=None):
    df = train.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True).copy()
    meses = sorted(df["mes_dt"].drop_duplicates())

    preds = []
    n_fallback_single_class = 0

    for i in range(cfg.min_train_months, len(meses)):
        mes_teste = meses[i]
        meses_treino = meses[:i]

        treino = df[df["mes_dt"].isin(meses_treino)].copy()
        teste = df[df["mes_dt"] == mes_teste].copy()

        if treino.empty or teste.empty:
            continue

        treino_f = m.filter_ml_train(treino)
        teste_f = teste.dropna(subset=FEATURES).copy()

        if treino_f.empty or teste_f.empty:
            continue

        X_train = treino_f[FEATURES].copy()
        y_train = (treino_f["share_hist"] > 0).astype(int)

        pred = teste_f.copy()

        if y_train.nunique() < 2:
            n_fallback_single_class += 1
            majority = int(y_train.mode().iloc[0]) if len(y_train) else 0
            pred["occ_pred"] = majority
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            model = LogisticRegression(
                max_iter=2000,
                class_weight=class_weight,
                random_state=cfg.random_state,
            )
            model.fit(X_train_s, y_train)
            X_test_s = scaler.transform(teste_f[FEATURES])
            pred["occ_pred"] = model.predict(X_test_s)

        preds.append(pred)

    backtest_pred = (
        pd.concat(preds, ignore_index=True)
          .sort_values(["mes_dt", "subsistema"])
          .reset_index(drop=True)
    )

    base_eval = m.filtrar_periodo_validacao(backtest_pred)
    y_true = (base_eval["share_hist"] > 0).astype(int)
    y_pred = base_eval["occ_pred"].astype(int)

    metrics = {
        "n_obs": len(base_eval),
        "n_fallback_single_class_folds": n_fallback_single_class,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }
    return backtest_pred, metrics


print("=" * 70)
print("Construindo painel de treino (build_train)...")
train = m.build_train(cfg)
print(f"Painel: {train.shape[0]} linhas, {train['mes_dt'].min()} a {train['mes_dt'].max()}")

n_pos_full = int((train["share_hist"] > 0).sum())
n_total_full = int(len(train))
print(f"GFOM>0 no painel completo: {n_pos_full}/{n_total_full} ({100*n_pos_full/n_total_full:.2f}%)")

print("\n" + "=" * 70)
print("VARIANTE 1: LogisticRegression SEM class_weight (default)")
_, metrics_unw = run_holdout_backtest_logistic(train, cfg, class_weight=None)
for k, v in metrics_unw.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("VARIANTE 2: LogisticRegression COM class_weight='balanced'")
backtest_pred_bal, metrics_bal = run_holdout_backtest_logistic(train, cfg, class_weight="balanced")
for k, v in metrics_bal.items():
    print(f"  {k}: {v}")

comp = pd.DataFrame([
    {"variante": "logistic_unweighted", **metrics_unw},
    {"variante": "logistic_balanced", **metrics_bal},
])

out_path = "/sessions/rcw-015oxuvxnzz53lzq1dsf1hgu/mnt/artigo GFOM/artigo-IEEE/resultados_referencia/logistic_baseline_comparison_R2-10.csv"
comp.to_csv(out_path, index=False)
print(f"\nSalvo em: {out_path}")
print("\n" + "=" * 70)
print("COMPARACAO LADO A LADO")
print(comp.to_string(index=False))
