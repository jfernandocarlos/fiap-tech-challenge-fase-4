# Credit Scoring Observability — Tech Challenge Fase 4

Camada de **sustentação e confiabilidade** para um modelo de Credit Scoring
em produção: contratos de qualidade de dados (Pandera), simulação e detecção
de Data Drift (Evidently + PSI/KS), observabilidade (logs + MLflow + dashboard)
e documentação de governança (LGPD).

> Pós-graduação em Engenharia de Machine Learning — FIAP
> Autor: **José Fernando Carlos — RM373126**

---

## Sumário

- [Problema](#problema)
- [Stack](#stack)
- [Estrutura](#estrutura)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Governança e LGPD](#governança-e-lgpd)

---

## Problema

Uma fintech suspeita que o modelo de crédito está degradando em silêncio após
mudanças econômicas (inflação, novos perfis). O foco **não** é criar o melhor
modelo do mundo — é **interceptar dados ruins**, **detectar drift** e
**documentar compliance**.

| Dataset | Papel |
|---------|-------|
| Referência | Treino / baseline estatístico (`data/reference/`) |
| Produção | Distribuição deslocada para análise de drift (`data/production/`) |
| Inválido | Lote com erros para provar que o contrato bloqueia (`data/invalid/`) |

Dataset sintético (~8.000 amostras), inspirado em German Credit / Lending Club,
gerado localmente (sem Kaggle).

## Stack

- **scikit-learn** — GradientBoostingClassifier (modelo baseline)
- **Pandera** — data contracts (idade, renda, duplicatas, etc.)
- **Evidently AI** — relatório HTML de Data Drift
- **SciPy** — PSI auxiliar + Kolmogorov-Smirnov por feature
- **MLflow** — tracking de treino e drift
- **structlog** — logs JSON do pipeline
- **Poetry + Ruff + pytest + GitHub Actions**

## Estrutura

```
.
├── src/credit/
│   ├── config.py, logging_config.py
│   ├── data/          # generate, loader, contracts (Pandera)
│   ├── models/        # training + predict
│   ├── drift/         # simulate + detect (Evidently)
│   ├── monitoring/    # pipeline logs + MLflow
│   └── reporting/     # dashboard HTML
├── scripts/           # CLI de cada etapa
├── tests/
├── docs/              # architecture, monitoring, governance
├── reports/           # drift_report.html, monitoring_dashboard.html
├── data/{reference,production,invalid}/
├── models/
└── logs/
```

## Setup

Pré-requisitos: Python 3.10–3.12, [Poetry](https://python-poetry.org/), Git.

```bash
make install
cp .env.example .env
make validate
make test
```

## Pipeline

```bash
make pipeline
# ou passo a passo:
make generate-data
make validate-data                                          # deve PASSAR
poetry run python -m scripts.validate_data data/invalid/credit_invalid_batch.csv  # deve FALHAR
make train
make simulate-drift
make detect-drift
make dashboard
```

Artefatos esperados:

| Arquivo | Conteúdo |
|---------|----------|
| `models/credit_model.joblib` | Modelo treinado |
| `models/metrics.json` | Accuracy, F1, ROC-AUC |
| `reports/drift_report.html` | Relatório Evidently |
| `reports/drift_summary.json` | PSI/KS por feature |
| `reports/monitoring_dashboard.html` | Visão consolidada |
| `logs/pipeline.jsonl` | Trilha de execução |
| `mlruns/` | Experimentos MLflow (`poetry run mlflow ui`) |

### Regras do contrato (Pandera)

1. `age` entre 18 e 75, sem nulos
2. `income` acima do mínimo, sem nulos
3. `customer_id` único (sem duplicatas)
4. `loan_amount` mínimo e `debt_ratio` em [0, 1]

### Drift simulado

- `income` × 1.45 (inflação nominal)
- `late_payments` + Poisson (stress financeiro)
- leve deslocamento em `debt_ratio` (concept drift suave)

## Governança e LGPD

Documentação completa em [`docs/governance.md`](docs/governance.md). Resumo:

- **PII:** dataset sintético com IDs pseudonimizados; sem CPF/nome/e-mail.
- **Base legal:** execução contratual / procedimentos preliminares de crédito;
  direito à revisão humana (Art. 20 LGPD).
- **Retenção:** referência + modelo por até 5 anos; produção/logs 12 meses;
  relatórios de drift 24 meses.
- **Causalidade:** inflação desloca `income`; stress eleva atrasos; modelo
  treinado no regime antigo perde calibração → retreino + monitoramento contínuo.

Ver também: [`docs/architecture.md`](docs/architecture.md), [`docs/monitoring_plan.md`](docs/monitoring_plan.md).

## Critérios do enunciado (checklist)

- [x] Pipeline de validação / data contracts (Pandera)
- [x] Scripts de simulação e detecção de drift
- [x] Relatórios HTML (Evidently + dashboard)
- [x] Documentação de governança no README + `docs/`
- [x] Histórico de commits semântico
