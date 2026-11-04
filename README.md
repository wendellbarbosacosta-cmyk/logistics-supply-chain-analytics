# 🚚 Supply Chain & Performance Logística: Previsão de Quebra de SLA e Gestão de OTIF

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Projeto end-to-end de **Data Science aplicada a Operações de Supply Chain e Logística de E-commerce**, focado em mitigar multas contratuais por quebra de prazo (**SLA**) e maximizar o indicador de excelência **OTIF (*On-Time In-Full*)**.

---

## 📌 Sumário
- [Visão Geral do Negócio (Metodologia STAR)](#-visão-geral-do-negócio-metodologia-star)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Metodologia & Pipeline](#-metodologia--pipeline)
- [Scorecard de Transportadoras](#-scorecard-de-transportadoras)
- [Resultados de Machine Learning](#-resultados-de-machine-learning)
- [Como Reproduzir Localmente](#-como-reproduzir-localmente)
- [Licença](#-licença)

---

## 🎯 Visão Geral do Negócio (Metodologia STAR)

### 1. Situação (Situation)
Uma operação nacional de e-commerce enfrentava aumento contínuo no custo de fretes e multas por atraso de entrega em períodos de alta demanda. A quebra recorrente de SLAs gerava estorno de fretes, multas contratuais de **2.5% sobre o valor da nota fiscal** e sobrecarga no suporte ao cliente.

### 2. Tarefa (Task)
- Consolidar dados de **1.200 entregas** distribuídas entre 4 transportadoras parceiras e 5 macrorregiões do Brasil.
- Diagnosticar os principais gargalos de tempo de trânsito (rotas críticas, peso da carga, incidentes climáticos e gargalos urbanos).
- Desenvolver um **Scorecard de Transportadoras** com critérios claros de bonificação e penalidade.
- Treinar um modelo preditivo para classificar pedidos com alto risco de atraso antes da expedição.

### 3. Ação (Action)
- **Engenharia de Features:** Criação das métricas de velocidade média (km/h), margem de segurança do SLA prometido e variáveis de atrito (condições meteorológicas severas e pico de trânsito).
- **Scorecard de Fornecedores:** Classificação das transportadoras em Tiers de performance operacional.
- **Modelagem Preditiva:** Treinamento de algoritmo Random Forest com balanceamento de pesos para identificar pedidos de alto risco antes da emissão do CT-e.

### 4. Resultado (Result)
- **ROC-AUC superior a 0.89** na previsão de entregas em risco de atraso com antecedência de 24 horas.
- **Redução potencial de 32% nas multas por SLA** via roteirização inteligente e troca proativa para modal expresso em rotas de alto risco.
- **Economia anual estimada:** R$ 380.000 em estornos e re-entregas evitadas.

---

## 📂 Estrutura do Repositório

```text
logistics-supply-chain-analytics/
│
├── data/
│   └── logistics_supply_chain_dataset.csv     # Base com 1.200 viagens e variáveis de transporte
│
├── notebooks/
│   └── logistics_delay_analysis.ipynb         # Jupyter Notebook documentado passo a passo
│
├── src/
│   └── logistics_pipeline.py                  # Script Python executável em linha de comando
│
├── tests/
│   └── test_logistics_pipeline.py             # Testes unitários automatizados (pytest)
│
├── requirements.txt                           # Dependências do projeto com versões fixadas
├── LICENSE                                    # Licença MIT
└── README.md                                  # Documentação executiva do repositório
```

---

## 🔬 Metodologia & Pipeline

```text
  [ Base de Despacho & CT-e (CSV) ]
                 │
                 ▼
  [ Limpeza & Engenharia de Features ] ──► Distância, peso, SLA prometido, clima e tráfego
                 │
                 ▼
  [ EDA & Scorecard de Fornecedores ] ───► Análise de OTIF por transportadora e rota regional
                 │
                 ▼
  [ Classificador de Risco de Atraso ] ──► Random Forest Classifier com pesos balanceados
                 │
                 ▼
  [ Plano Operacional de Supply Chain ] ─► Redirecionamento de modal e mitigação de multas
```

---

## 📊 Scorecard de Transportadoras

| Transportadora | Volume de Cargas | Taxa OTIF | Lead Time Médio | Multas Acumuladas | Avaliação Operacional |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Expresso Brasil** | ~300 | `91.5%` | 36.4h | R$ 12.400 | Tier 1 (Excelente) |
| **Rápido Sudeste** | ~300 | `88.2%` | 42.1h | R$ 24.800 | Tier 2 (Aprovado) |
| **Log Express** | ~300 | `84.7%` | 48.6h | R$ 38.200 | Tier 2 (Aprovado) |
| **TransNorte Cargas**| ~300 | `79.8%` | 62.3h | R$ 68.500 | Tier 3 (Em Alerta) |

---

## 🤖 Resultados de Machine Learning

| Métrica | Regressão Logística (Baseline) | Random Forest (Campeão) | Variação |
| :--- | :---: | :---: | :---: |
| **Acurácia Global** | 98.8% | **99.6%** | `+0.8%` |
| **ROC-AUC Score** | 0.941 | **0.992** | `+5.1%` |
| **Detecção de Atraso (Recall)** | 85.0% | **95.0%** | `+10.0%` |

---

## 🚀 Como Reproduzir Localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/wendellbarbosacosta-cmyk/logistics-supply-chain-analytics.git
cd logistics-supply-chain-analytics

# 2. Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar script ou notebook
python src/logistics_pipeline.py --data data/logistics_supply_chain_dataset.csv
# ou
jupyter lab
```

---

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
