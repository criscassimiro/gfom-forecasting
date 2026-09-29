# -*- coding: utf-8 -*-
"""
Driver para R2-04 (desbalanceamento de classes).

O pipeline publicado treina um UNICO RandomForestRegressor sobre share_hist
(fracao continua de geracao termica atribuivel ao GFOM), sem classificador
dedicado -- a classificacao de ocorrencia e obtida por threshold em zero
sobre a saida continua da regressao (ver Secao II.H.2, resposta ao R3-11).
Por isso, tecnicas classicas de balanceamento como SMOTE ou class_weight nao
se aplicam diretamente: elas corrigem a fronteira de decisao de um
classificador treinado sobre um alvo categorico desbalanceado, premissa que
nao existe aqui.

Este driver testa o analogo que de fato se aplica a essa arquitetura de
regressao: ponderar as observacoes (sample_weight) durante o treino do
RandomForestRegressor, dando peso maior aos meses com GFOM>0 (share_hist>0).
O peso usado e o classico de frequencia inversa (mesma logica do
class_weight='balanced' do scikit-learn: n_neg/n_pos), recomputado a cada
fold do walk-forward (o conjunto de treino cresce a cada mes).

Reusa build_train, run_holdout_backtest (baseline, para conferencia contra a
Tabela 5 ja publicada), train_random_forest, filtrar_periodo_validacao e
calcular_metricas_completas de Metricas_proj_gfom_RF_local.py -- sem alterar
nenhuma logica ja publicada. Apenas adiciona uma variante com sample_weight
para comparacao.
"""
import sys
sys.path.insert(0, "/sessions/rcw-015oxuvxnzz53lzq1dsf1hgu/mnt/artigo GFOM")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import Metricas_proj_gfom_RF_local as m

cfg = m.CFG


def train_random_forest_weighted(train: pd.DataFrame, cfg, weight_mode: str = "balanced"):
    base = m.filter_ml_train(train)
    features = m.get_feature_cols()

    X = base[features].copy()
    y = base["share_hist"].copy()

    pos = (y > 0)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())

    if weight_mode == "balanced" and n_pos > 0:
        w_pos = n_neg / n_pos
    else:
        w_pos = 1.0

    sample_weight = np.where(pos, w_pos, 1.0)

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=sample_weight)

    return model, features, base, w_pos


def run_holdout_backtest_weighted(train: pd.DataFrame, cfg):
    df = train.sort_values(["mes_dt", "subsistema"]).reset_index(drop=True).copy()
    meses = sorted(df["mes_dt"].drop_duplicates())

    preds = []
    w_pos_log = []

    for i in range(cfg.min_train_months, len(meses)):
        mes_teste = meses[i]
        meses_treino = meses[:i]

        treino = df[df["mes_dt"].isin(meses_treino)].copy()
        teste = df[df["mes_dt"] == mes_teste].copy()

        if treino.empty or teste.empty:
            continue

        try:
            model, features, _, w_pos = train_random_forest_weighted(treino, cfg)
            w_pos_log.append({"mes_teste": mes_teste, "w_pos": w_pos})
        except Exception as e:
            print(f"Falha ao treinar (weighted) para o mes {mes_teste}: {e}")
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

    backtest_pred = (
        pd.concat(preds, ignore_index=True)
          .sort_values(["mes_dt", "subsistema"])
          .reset_index(drop=True)
    )

    base_eval = m.filtrar_periodo_validacao(backtest_pred)
    y_true = base_eval["gfom_hist_mwmed"]
    y_pred = base_eval["gfom_pred_mwmed"]
    metricas = m.calcular_metricas_completas(y_true, y_pred)

    metrics = pd.DataFrame([{
        "periodo": "2022_2025",
        "n_obs": len(base_eval),
        **metricas,
    }])

    w_pos_df = pd.DataFrame(w_pos_log)
    return backtest_pred, metrics, w_pos_df


print("=" * 70)
print("Construindo painel de treino (build_train)...")
train = m.build_train(cfg)
print(f"Painel: {train.shape[0]} linhas, {train['mes_dt'].min()} a {train['mes_dt'].max()}")

n_pos_full = int((train["share_hist"] > 0).sum())
n_total_full = int(len(train))
print(f"GFOM>0 no painel completo: {n_pos_full}/{n_total_full} ({100*n_pos_full/n_total_full:.2f}%)")

print("\n" + "=" * 70)
print("BASELINE (sem sample_weight) -- deve reproduzir a Tabela 5 ja publicada")
backtest_pred_base, metrics_base = m.run_holdout_backtest(train, cfg)
print(metrics_base[["periodo", "n_obs", "mae", "rmse", "r2", "smape_percent",
                     "precision", "recall", "f1", "tp", "fp", "fn", "tn"]].to_string(index=False))

print("\n" + "=" * 70)
print("VARIANTE COM sample_weight BALANCEADO (w_pos = n_neg/n_pos por fold)")
backtest_pred_w, metrics_w, w_pos_df = run_holdout_backtest_weighted(train, cfg)
print(metrics_w[["periodo", "n_obs", "mae", "rmse", "r2", "smape_percent",
                  "precision", "recall", "f1", "tp", "fp", "fn", "tn"]].to_string(index=False))

print("\nFaixa de w_pos usada ao longo dos folds do walk-forward:")
print(f"  min={w_pos_df['w_pos'].min():.2f}  max={w_pos_df['w_pos'].max():.2f}  "
      f"mean={w_pos_df['w_pos'].mean():.2f}")

# comparação lado a lado
comp = pd.concat([
    metrics_base.assign(variante="baseline"),
    metrics_w.assign(variante="sample_weight_balanced"),
], ignore_index=True)
cols = ["variante", "periodo", "n_obs", "mae", "rmse", "r2", "smape_percent",
        "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
comp = comp[cols]

out_path = "/sessions/rcw-015oxuvxnzz53lzq1dsf1hgu/mnt/artigo GFOM/artigo-IEEE/resultados_referencia/sample_weight_comparison_R2-04.csv"
comp.to_csv(out_path, index=False)
print(f"\nSalvo em: {out_path}")
print("\n" + "=" * 70)
print("COMPARACAO LADO A LADO")
print(comp.to_string(index=False))
