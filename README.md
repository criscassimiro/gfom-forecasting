# gfom-forecasting

Código de apoio ao artigo "Forecasting Discretionary Out-of-Merit Thermal Dispatch in the Brazilian Power System Using Ensemble Learning" (IEEE Access, Manuscript ID Access-2026-19690).

## Sobre o GFOM

GFOM (Geração Fora da Ordem de Mérito) é o despacho termelétrico fora da ordem de mérito econômico no Sistema Interligado Nacional (SIN) brasileiro, acionado por critérios de segurança operativa. O objetivo deste trabalho é prever a ocorrência e a magnitude do GFOM a partir de variáveis operacionais (EAR, curva de referência, carga, geração térmica histórica).

## Estrutura do repositório

```
scripts/
  01_feature_engineering/   Seleção e ponderação de variáveis (correlação de Spearman)
  02_models/                 Modelos Random Forest e Gradient Boosting (scripts canônicos)
  03_figuras/                 Geração das figuras do artigo
  04_legado/                  Versões anteriores/exploratórias, mantidas para rastreabilidade

resultados_referencia/       Saídas numéricas (CSV) já calculadas, usadas como referência
                              para conferência — não são geradas automaticamente ao clonar
                              o repositório (dependem dos dados brutos, não versionados)
```

## Scripts canônicos e sua relação com o artigo

| Script | Papel no artigo |
|---|---|
| `01_feature_engineering/EAR_CREF_Carga20212026.py` | Calcula os pesos das variáveis por normalização da magnitude da correlação de Spearman com o GFOM (Figura 5, Tabela de pesos) |
| `02_models/proj_gfom_hibrido_RF.py` | Backtest de holdout (últimos 12 meses, por subsistema) do Random Forest — fonte da linha RF da Tabela 4 |
| `02_models/Metricas_proj_gfom_RF.py` | Backtest final do Random Forest (janela 2022–2025, subsistema × mês agrupado) — fonte da Tabela 5 |
| `02_models/E7_compracao_RF_GB.py` | Comparação RF × Gradient Boosting — fonte da linha GB da Tabela 4 |
| `03_figuras/*.py` | Scripts de geração das figuras do artigo |
| `04_legado/*.py` | Iterações anteriores (não são a fonte de nenhum resultado publicado); mantidos apenas para histórico |

## Hiperparâmetros usados (extraídos diretamente do código — ver Tabela 2 do artigo)

**Random Forest**: `n_estimators=500, max_depth=8, min_samples_leaf=2, min_samples_split=4, random_state=42`

**Gradient Boosting**: `n_estimators=300, learning_rate=0.05, max_depth=3, min_samples_leaf=2, min_samples_split=4, subsample=1.0, random_state=42`

Não foi localizada, em nenhum script, uma rotina de busca de hiperparâmetros (grid search); os valores acima foram fixados diretamente no código.

## Dados

Os dados brutos (séries históricas de EAR, ENA, carga e geração — arquivos CSV de até ~400 MB, com origem no ONS) **não estão neste repositório**, por tamanho e por serem dados operacionais de terceiros. Os scripts esperam esses arquivos em um diretório local configurado na variável `BASE_DIR` de cada script.

## Requisitos

Ver `requirements.txt`. Testado com Python 3.10+.

## Status

Repositório criado em setembro/2026 durante a revisão do artigo para nova submissão ao IEEE Access, como parte do esforço de reprodutibilidade solicitado pelos revisores.
