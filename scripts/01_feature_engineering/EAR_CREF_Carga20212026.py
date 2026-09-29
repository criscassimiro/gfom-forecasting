import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# CONFIGURAÇÃO DOS ARQUIVOS
# =========================
PASTA_BASE = Path(".")

ARQ_EAR = "ear_historico.csv"
ARQ_CREF = "curvas_de_referencia.csv"
ARQ_CARGA = "carga_historico.csv"
ARQ_GFOM = "gt_historico.csv"

# Se quiser analisar só um subsistema, defina aqui.
# Exemplo: FILTRO_SUBSISTEMA = "Sudeste"
FILTRO_SUBSISTEMA = None


# =========================
# FUNÇÕES AUXILIARES
# =========================
def carregar_flexivel(caminho):
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    # Excel
    if caminho.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(caminho)
        return padronizar_colunas(df)

    # CSV
    tentativas = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
    ]

    for tentativa in tentativas:
        try:
            df = pd.read_csv(caminho, **tentativa)
            if df.shape[1] > 1:
                return padronizar_colunas(df)
        except Exception:
            continue

    raise ValueError(f"Não foi possível ler corretamente o arquivo: {caminho}")


def padronizar_colunas(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def encontrar_coluna(df, candidatos):
    cols = df.columns.tolist()
    for c in candidatos:
        if c in cols:
            return c
    print("Colunas disponíveis:", cols)
    raise KeyError(
        f"Nenhuma das colunas esperadas foi encontrada: {candidatos}")


# =========================
# 1. CARREGAR BASES
# =========================
ear = carregar_flexivel(PASTA_BASE / ARQ_EAR)
cref = carregar_flexivel(PASTA_BASE / ARQ_CREF)

print("CREF COLUNAS REAL:")
print(cref.columns.tolist())
print(cref.head())

carga = carregar_flexivel(PASTA_BASE / ARQ_CARGA)
gfom = carregar_flexivel(PASTA_BASE / ARQ_GFOM)

print("Colunas EAR:", ear.columns.tolist())
print("Colunas CREF:", cref.columns.tolist())
print("Colunas CARGA:", carga.columns.tolist())
print("Colunas GFOM:", gfom.columns.tolist())


# =========================
# 2. IDENTIFICAR COLUNAS
# =========================
# EAR
col_data_ear = encontrar_coluna(ear, ["data", "data_hora", "datetime"])
col_subs_ear = encontrar_coluna(ear, ["subsistema", "subsis", "submercado"])
col_ear = encontrar_coluna(ear, ["ear_fim_per", "ear_per", "ear"])

# CREF
col_ano_cref = encontrar_coluna(cref, ["ano"])
col_mes_cref = encontrar_coluna(cref, ["mes"])
col_cref_sin = encontrar_coluna(
    cref,
    ["curva_amarela", "curva_am", "cref_sin", "curvaamarela", "cref_amarela"]
)

# CARGA
col_data_carga = encontrar_coluna(carga, ["data_hora", "data", "datetime"])
col_subs_carga = encontrar_coluna(
    carga, ["subsistema", "subsis", "submercado"])
col_carga = encontrar_coluna(carga, ["carga_mwmed", "carga", "carga_mwmedio"])

# GFOM
col_data_gfom = encontrar_coluna(gfom, ["data_hora", "data", "datetime"])
col_subs_gfom = encontrar_coluna(gfom, ["subsistema", "subsis", "submercado"])
col_gfom = encontrar_coluna(gfom, ["val_verifgarantiaenergetica"])


# =========================
# 3. TRATAR EAR
# =========================
ear[col_data_ear] = pd.to_datetime(ear[col_data_ear], errors="coerce")
ear[col_ear] = pd.to_numeric(ear[col_ear], errors="coerce")

ear = ear.dropna(subset=[col_data_ear, col_subs_ear, col_ear]).copy()

ear["ano"] = ear[col_data_ear].dt.year
ear["mes"] = ear[col_data_ear].dt.month
ear[col_subs_ear] = ear[col_subs_ear].astype(str).str.strip()

if FILTRO_SUBSISTEMA:
    ear = ear[ear[col_subs_ear].str.lower(
    ) == FILTRO_SUBSISTEMA.lower()].copy()

ear_mensal = (
    ear.groupby([col_subs_ear, "ano", "mes"], as_index=False)[col_ear]
    .mean()
    .rename(columns={
        col_subs_ear: "subsistema",
        col_ear: "ear_fim_per"
    })
)


# =========================
# 4. TRATAR CREF DO SIN
# =========================
cref[col_ano_cref] = pd.to_numeric(cref[col_ano_cref], errors="coerce")
cref[col_mes_cref] = pd.to_numeric(cref[col_mes_cref], errors="coerce")

col_cref_amarela = encontrar_coluna(
    cref,
    ["curva_amarela", "curva_am", "cref_amarela"]
)

col_cref_vermelha = encontrar_coluna(
    cref,
    ["curva_vermelha", "cref_vermelha"]
)

cref[col_cref_amarela] = pd.to_numeric(cref[col_cref_amarela], errors="coerce")
cref[col_cref_vermelha] = pd.to_numeric(
    cref[col_cref_vermelha], errors="coerce")

cref = cref.dropna(subset=[col_ano_cref, col_mes_cref]).copy()

cref_tratada = (
    cref[[col_ano_cref, col_mes_cref, col_cref_amarela, col_cref_vermelha]]
    .rename(columns={
        col_ano_cref: "ano",
        col_mes_cref: "mes",
        col_cref_amarela: "cref_amarela",
        col_cref_vermelha: "cref_vermelha"
    })
)

cref_tratada["ano"] = cref_tratada["ano"].astype(int)
cref_tratada["mes"] = cref_tratada["mes"].astype(int)


# =========================
# 5. TRATAR CARGA
# =========================
carga[col_data_carga] = pd.to_datetime(carga[col_data_carga], errors="coerce")
carga[col_carga] = pd.to_numeric(carga[col_carga], errors="coerce")

carga = carga.dropna(subset=[col_data_carga, col_subs_carga, col_carga]).copy()

carga["ano"] = carga[col_data_carga].dt.year
carga["mes"] = carga[col_data_carga].dt.month
carga[col_subs_carga] = carga[col_subs_carga].astype(str).str.strip()

if FILTRO_SUBSISTEMA:
    carga = carga[carga[col_subs_carga].str.lower(
    ) == FILTRO_SUBSISTEMA.lower()].copy()

carga_mensal = (
    carga.groupby([col_subs_carga, "ano", "mes"], as_index=False)[col_carga]
    .mean()
    .rename(columns={
        col_subs_carga: "subsistema",
        col_carga: "carga_mwmed"
    })
)


# =========================
# 6. TRATAR GFOM HISTÓRICO
# =========================
gfom[col_data_gfom] = pd.to_datetime(gfom[col_data_gfom], errors="coerce")
gfom[col_gfom] = pd.to_numeric(gfom[col_gfom], errors="coerce").fillna(0)

gfom = gfom.dropna(subset=[col_data_gfom, col_subs_gfom]).copy()

gfom["ano"] = gfom[col_data_gfom].dt.year
gfom["mes"] = gfom[col_data_gfom].dt.month
gfom[col_subs_gfom] = gfom[col_subs_gfom].astype(str).str.strip()

if FILTRO_SUBSISTEMA:
    gfom = gfom[gfom[col_subs_gfom].str.lower(
    ) == FILTRO_SUBSISTEMA.lower()].copy()

# soma mensal do GFOM por subsistema
gfom_mensal = (
    gfom.groupby([col_subs_gfom, "ano", "mes"], as_index=False)[col_gfom]
    .sum()
    .rename(columns={
        col_subs_gfom: "subsistema",
        col_gfom: "gfom"
    })
)


# =========================
# 7. MERGE FINAL E7
# =========================

df_e7 = (
    ear_mensal
    .merge(cref_tratada, on=["ano", "mes"], how="left")
    .merge(carga_mensal, on=["subsistema", "ano", "mes"], how="left")
    .merge(gfom_mensal, on=["subsistema", "ano", "mes"], how="left")
)

df_e7["gfom"] = df_e7["gfom"].fillna(0)

# distâncias
df_e7["dist_cref_amarela"] = df_e7["ear_fim_per"] - df_e7["cref_amarela"]
df_e7["dist_cref_vermelha"] = df_e7["ear_fim_per"] - df_e7["cref_vermelha"]

# gatilhos binários
df_e7["gatilho_cref_amarela"] = (
    df_e7["ear_fim_per"] <= df_e7["cref_amarela"]).astype(int)
df_e7["gatilho_cref_vermelha"] = (
    df_e7["ear_fim_per"] <= df_e7["cref_vermelha"]).astype(int)

# regime / zona


def classificar_zona(row):
    if row["ear_fim_per"] <= row["cref_vermelha"]:
        return "critica"
    elif row["ear_fim_per"] <= row["cref_amarela"]:
        return "alerta"
    else:
        return "normal"


df_e7["zona_cref"] = df_e7.apply(classificar_zona, axis=1)
print("\nAmostra da base final:")
print(df_e7.head(20))

# =========================
# 8. CORRELAÇÃO
# =========================
variaveis = [
    "cref_amarela",
    "ear_fim_per",
    "carga_mwmed",
    "gfom"
]

df_corr = df_e7[variaveis].dropna().copy()

corr_spearman = df_corr.corr(method="spearman")
corr_pearson = df_corr.corr(method="pearson")

print("\n=== Correlação de Spearman ===")
print(corr_spearman)

print("\n=== Correlação de Pearson ===")
print(corr_pearson)


# =========================
# 9. PESOS INICIAIS PARA O AHP
# =========================
corr_gfom = corr_spearman["gfom"].drop("gfom").abs()
pesos = corr_gfom / corr_gfom.sum()

print("\n=== Pesos iniciais baseados na correlação com GFOM ===")
print(pesos)


# =========================
# 10. SCORE E7 OPCIONAL
# =========================
# Se quiser usar depois como score sintético
# Atenção: aqui as variáveis estão em escala original.
# Se quiser um score comparável, ideal normalizar antes.
# Mantive comentado para não misturar etapas.

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# vars_score = ["cref_sin", "ear_fim_per", "carga_mwmed"]
# df_e7[vars_score] = scaler.fit_transform(df_e7[vars_score])
# df_e7["score_e7"] = (
#     pesos["cref_sin"] * df_e7["cref_sin"] +
#     pesos["ear_fim_per"] * df_e7["ear_fim_per"] +
#     pesos["carga_mwmed"] * df_e7["carga_mwmed"]
# )


# =========================
# 11. ORDENAR E SALVAR
# =========================
df_e7 = df_e7.sort_values(["subsistema", "ano", "mes"]).reset_index(drop=True)

df_e7.to_csv(PASTA_BASE / "dataset_E7_correlacao.csv",
             index=False, sep=";", encoding="utf-8-sig")
corr_spearman.to_csv(PASTA_BASE / "correlacao_spearman_E7.csv",
                     sep=";", encoding="utf-8-sig")
corr_pearson.to_csv(PASTA_BASE / "correlacao_pearson_E7.csv",
                    sep=";", encoding="utf-8-sig")
pesos.to_csv(PASTA_BASE / "pesos_iniciais_AHP_E7.csv",
             sep=";", encoding="utf-8-sig", header=["peso"])


# =========================
# 12. PREPARAR BASE PARA GRÁFICOS
# =========================
df_plot = df_e7.copy()
df_plot["data_ref"] = pd.to_datetime(
    df_plot["ano"].astype(str) + "-" + df_plot["mes"].astype(str) + "-01"
)

# Se quiser focar em um subsistema específico para os gráficos:
SUBSISTEMA_GRAFICO = "Nordeste"   # pode trocar para "Sudeste", "Sul", etc.
df_sub = df_plot[df_plot["subsistema"] ==
                 SUBSISTEMA_GRAFICO].sort_values("data_ref").copy()


# =========================
# 13. FIGURA 1 - HEATMAP DE CORRELAÇÃO (SPEARMAN)
# =========================
corr_fig = df_e7[["ear_fim_per", "cref_amarela",
                  "carga_mwmed", "gfom"]].corr(method="spearman")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_fig, aspect="auto")

ax.set_xticks(range(len(corr_fig.columns)))
ax.set_xticklabels(corr_fig.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr_fig.index)))
ax.set_yticklabels(corr_fig.index)

for i in range(len(corr_fig.index)):
    for j in range(len(corr_fig.columns)):
        ax.text(j, i, f"{corr_fig.iloc[i, j]:.2f}", ha="center", va="center")

ax.set_title("Spearman correlation matrix")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("fig1_heatmap_correlacao_spearman.png",
            dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 14. FIGURA 2 - PESOS FINAIS
# =========================
pesos_plot = pd.Series({
    "EAR": pesos["ear_fim_per"],
    "CRef amarela": pesos["cref_amarela"],
    "Carga": pesos["carga_mwmed"]
})

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(pesos_plot.index, pesos_plot.values)

ax.set_ylabel("Weight")
ax.set_title("Data-driven weights for variables")
ax.set_ylim(0, max(pesos_plot.values) * 1.2)

for i, v in enumerate(pesos_plot.values):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig("fig2_pesos_variaveis_e7.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 15. FIGURA 3 - SÉRIE TEMPORAL (GFOM + EAR + CREF AMARELA)
# =========================
fig, ax1 = plt.subplots(figsize=(11, 5))

# --- GFOM (barra) ---
ax1.bar(
    df_sub["data_ref"],
    df_sub["gfom"],
    width=25,
    alpha=0.6,
    label="Monthly GFOM"
)
ax1.set_ylabel("GFOM (MWmed)")
ax1.set_title(f"GFOM, Storage and Reference Curve (Yellow) - {SUBSISTEMA_GRAFICO}")

# --- Eixo secundário ---
ax2 = ax1.twinx()

ax2.plot(
    df_sub["data_ref"],
    df_sub["ear_fim_per"],
    linewidth=2,
    label="Storage"
)

ax2.plot(
    df_sub["data_ref"],
    df_sub["cref_amarela"],
    linestyle="--",
    linewidth=2,
    label="Reference Curve (Yellow)"
)

ax2.set_ylabel("Storage / Reference Curve (%)")

# --- Formatação eixo X ---
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# --- Grid leve (paper style) ---
ax1.grid(True, linestyle="--", alpha=0.3)

# --- Legenda combinada ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    frameon=False
)

# --- Layout ---
plt.tight_layout()

# --- Save ---
plt.savefig(
    f"fig3_serie_temporal_gfom_ear_cref_{SUBSISTEMA_GRAFICO.lower()}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================
# 16. FIGURA 4 - GFOM POR REGIME OPERATIVO
# =========================
# Requer que exista a coluna zona_cref
ordem = ["normal", "alert", "critical"]
dados_box = [df_e7.loc[df_e7["zona_cref"] == z, "gfom"].dropna()
             for z in ordem]

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(dados_box, tick_labels=ordem)
ax.set_ylabel("GFOM")
ax.set_title("GFOM distribution by operational regime")

plt.tight_layout()
plt.savefig("fig4_boxplot_gfom_por_regime.png", dpi=300, bbox_inches="tight")
plt.show()
