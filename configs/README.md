# Configurações

Duas fontes de verdade, com responsabilidades separadas:

| Arquivo | Conteúdo | Exemplos |
|---------|----------|----------|
| `.env` | Infraestrutura / runtime | `MLFLOW_TRACKING_URI`, `DRIFT_PSI_THRESHOLD` |
| `params.yaml` | Hiperparâmetros ML e limiares de validação/drift | `n_estimators`, `age_min`, `income_multiplier` |

Não misture as duas. Segredos e ambiente vão no `.env` (não versionado).
Parâmetros de experimento ficam no `params.yaml` (versionado).
