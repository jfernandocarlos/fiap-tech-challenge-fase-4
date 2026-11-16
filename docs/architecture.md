# Arquitetura — Credit Scoring Observability

## Contexto

Fintech de crédito com modelo de scoring em produção. Mudanças econômicas
(inflação, novos perfis) podem degradar o modelo em silêncio. Este projeto
adiciona a camada de **sustentação e confiabilidade**: contratos de dados,
detecção de drift e rastreamento operacional.

## Fluxo

```
[Dataset Referência] ──► Pandera (contrato) ──► Treino (sklearn GBM)
                                                    │
                                                    ▼
                                              MLflow (métricas/modelo)
                                                    │
[Dataset Produção] ◄── simulação de drift (income, late_payments)
        │
        ▼
  Predições + Evidently (Data Drift) + PSI/KS
        │
        ▼
  logs JSONL + dashboard HTML + alerta se PSI alto
```

## Separação referência × produção

| Dataset | Uso | Onde |
|---------|-----|------|
| Referência | Treino e baseline estatístico | `data/reference/` |
| Produção | Análise de drift (distribuição deslocada) | `data/production/` |
| Inválido | Demonstração do contrato bloqueando lixo | `data/invalid/` |

## Componentes

| Peça | Papel |
|------|-------|
| Pandera | Data contract — bloqueia ingestão inválida |
| GradientBoosting | Modelo leve de classificação binária |
| Evidently | Relatório HTML de Data Drift |
| PSI / KS | Testes estatísticos por feature |
| MLflow | Tracking de treino e drift |
| JSONL + dashboard | Observabilidade operacional local |

## Métricas de saúde monitoradas

- Status de execução das etapas (`success` / `failure` / `alert`)
- Volume de linhas processadas
- Quantidade de features com drift
- ROC-AUC / Accuracy / F1 do modelo baseline
- Limiar PSI configurável (`params.yaml` → `drift.psi_threshold`)
