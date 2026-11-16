# Plano de Monitoramento

## Objetivo

Dar visibilidade operacional ao pipeline de crédito: saber se a ingestão
passou no contrato, se o treino concluiu, e se o modelo está sob drift.

## Métricas escolhidas

| Métrica | Fonte | Alerta |
|---------|-------|--------|
| Status da etapa | `logs/pipeline.jsonl` | `status=failure` |
| Linhas processadas | detalhes do evento | queda abrupta de volume |
| Features com drift | `reports/drift_summary.json` | `n_drifted > 0` |
| PSI por feature | mesmo arquivo | PSI ≥ 0.20 |
| p-value KS | mesmo arquivo | p < 0.05 |
| ROC-AUC do modelo | `models/metrics.json` + MLflow | queda vs baseline |

## Ferramentas

1. **Evidently AI** — relatório HTML comparando referência × produção.
2. **MLflow** — runs de treino e de drift (params, métricas, artefatos).
3. **Dashboard HTML** — visão consolidada em `reports/monitoring_dashboard.html`.
4. **Logs JSONL** — trilha auditável de cada etapa.

## Como operar no dia a dia

```bash
make pipeline          # ponta a ponta
open reports/drift_report.html
open reports/monitoring_dashboard.html
poetry run mlflow ui   # http://127.0.0.1:5000
```

## Resposta a alerta de drift

1. Abrir o relatório Evidently e confirmar quais features mudaram.
2. Checar se a mudança é esperada (ex.: campanha, sazonalidade) ou anomalia.
3. Se confirmado: retreinar com janela recente, validar contrato, promover via MLflow.
4. Registrar a decisão (base legal / impacto) na documentação de governança.
