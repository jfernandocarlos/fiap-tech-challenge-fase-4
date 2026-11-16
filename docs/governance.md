# Governança, Privacidade (LGPD) e Causalidade

## 1. Dados sensíveis (PII) neste projeto

O dataset é **sintético**. Mesmo assim tratamos as colunas como se fossem
dados de clientes reais, porque o enunciado exige plano de privacidade.

| Campo | Classificação LGPD | Tratamento neste projeto |
|-------|--------------------|--------------------------|
| `customer_id` | identificador | pseudônimo (`REF-xxxxx` / `PRD-xxxxx`), sem CPF/e-mail |
| `age`, `income`, `employment_years` | dado pessoal / financeiro | uso exclusivo para scoring; sem reidentificação |
| `default` | dado sensível de crédito | acesso restrito aos artefatos de treino |

**Não coletamos** nome, CPF, endereço, telefone ou biometria.

## 2. Base legal (Art. 7º LGPD)

Para decisão de crédito automatizada em cenário real, a base típica é:

- **Execução de contrato / procedimentos preliminares** (análise de pedido de crédito); e/ou
- **Legítimo interesse**, com teste de balanceamento e direito de oposição.

O titular deve poder solicitar **revisão humana** de decisões automatizadas
(Art. 20 LGPD).

## 3. Privacy by Design / by Default

- Minimização: só features necessárias ao score.
- Pseudonimização de IDs.
- Separação referência × produção (não misturar bases).
- Contrato Pandera impede ingestão de dados fora de faixa / nulos críticos.
- Segredos e ambiente no `.env` (não versionado).

## 4. Retenção

| Artefato | Retenção sugerida |
|----------|-------------------|
| Dataset referência | enquanto o modelo estiver em produção + 5 anos (auditoria) |
| Dataset produção / drift | 12 meses rolantes |
| Logs de pipeline | 12 meses |
| Relatórios Evidently | 24 meses (evidência de monitoramento) |
| Modelo MLflow | versões produtivas até substituição + 5 anos |

Ao expirar: exclusão segura (wipe) e registro em ata de retenção.

## 5. Análise causal da degradação

**Situação observada:** PSI alto em `income` e `late_payments` no dataset
de produção.

**Causa raiz simulada (econômica):**

1. Inflação eleva a renda nominal → desloca a distribuição de `income`
   sem necessariamente melhorar capacidade real de pagamento.
2. Stress financeiro aumenta atrasos → `late_payments` sobe.
3. O modelo, treinado em regime anterior, interpreta mal o novo perfil
   (data drift → risco de concept drift na relação features → `default`).

**Prescrição:**

- Congelar promoção de novas versões até retreino.
- Retreinar com janela que cubra o novo regime.
- Recalibrar limiar de corte e reavaliar fairness por faixa de renda/idade.
- Manter monitoramento PSI/KS contínuo.

## 6. Mitigação de vieses

- Monitorar taxa de aprovação / inadimplência predita por faixas de `age`
  e `home_ownership`.
- Evitar proxies sensíveis não justificados.
- Documentar features e importâncias em `models/metrics.json`.
