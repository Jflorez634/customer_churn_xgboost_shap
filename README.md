# 🎯 Customer Churn Prediction & Explainability (XGBoost + SHAP)

[![Area](https://img.shields.io/badge/Área-Data%20Science%20%2F%20ML-purple.svg)]()
[![Model](https://img.shields.io/badge/Model-XGBoost-blue.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![XAI](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

> **Resumen Ejecutivo:** Solución integral de Machine Learning para predecir la fuga de clientes (*Customer Churn*) en una empresa de telecomunicaciones/suscripción. El proyecto aborda el desbalance natural de clases mediante penalización ponderada (`scale_pos_weight`), prioriza el **Recall** sobre la exactitud (*Accuracy*) para mitigar el alto costo de los Falsos Negativos, e incorpora **Explicabilidad de IA (XAI) con SHAP** tanto a nivel global como individual para accionar estrategias de retención personalizadas.

---

## 🎯 1. Problema de Negocio

- **Contexto:** En modelos de negocio basados en suscripción, la pérdida recurrente de clientes erosiona el *Customer Lifetime Value* (CLV). Retener a un cliente existente cuesta entre 5 y 7 veces menos que adquirir uno nuevo.
- **Desafío:** 
  1. **La trampa del Accuracy:** Un modelo ingenuo que prediga siempre «ningún cliente se va» lograría ~73.5% de Accuracy, pero sería completamente inútil para el negocio al no detectar a ningún desertor.
  2. **Asimetría de costos:** No detectar a un cliente que se marcha (Falso Negativo) cuesta cientos de dólares en ingresos perdidos, mientras que contactar a un cliente que pensaba quedarse (Falso Positivo) solo cuesta una llamada o un cupón.
  3. **El problema de la "caja negra":** Saber *quién* se va no es suficiente; el equipo comercial necesita saber *por qué* para ofrecer el incentivo adecuado.

---

## 🏗️ 2. Arquitectura del Flujo de Trabajo

```text
[ Dataset Telco Churn: 7,032 clientes ]
                 │
                 ▼
[ Preprocesamiento & Pipelines: Scikit-Learn ColumnTransformer ]
  ├── Variables numéricas: StandardScaler (tenure, MonthlyCharges, TotalCharges)
  └── Variables categóricas: OneHotEncoder (drop='first' para prevenir colinealidad)
                 │
                 ▼
[ Modelado Comparativo & Manejo de Desbalance ]
  ├── Modelo Base: Logistic Regression (class_weight='balanced') -> ROC-AUC: 0.8353
  └── Modelo Avanzado: XGBoost Classifier (scale_pos_weight=2.77) -> ROC-AUC: 0.8383
                 │
                 ▼
[ Explicabilidad con SHAP (TreeExplainer) ]
  ├── Global (Beeswarm Summary Plot): Identificación de drivers macro de fuga
  └── Local (Waterfall Plot): Diagnóstico individual de riesgo (ej. cliente con 96.2% prob.)
```

---

## 📁 3. Estructura del Repositorio

```text
customer-churn-xgboost-shap/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv         # Dataset oficial de clientes (IBM / Kaggle)
├── notebooks/
│   └── 01_churn_prediction_y_shap.ipynb     # Pipeline E2E ejecutado: EDA, Pipelines, XGBoost y SHAP
├── src/
│   └── run_notebook.py                      # Ejecutor automatizado headless con nbclient
├── requirements.txt                         # Dependencias versionadas
└── README.md                                # Documentación del proyecto
```

---

## 🛠️ 4. Stack Tecnológico

| Herramienta | Rol en el Proyecto |
| :--- | :--- |
| **Python 3.11+** | Entorno de desarrollo |
| **Pandas / NumPy** | Manipulación de datos y operaciones vectoriales |
| **Scikit-Learn** | Pipelines de transformación, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` |
| **XGBoost** | Algoritmo principal de Gradient Boosted Trees |
| **SHAP** | Explicabilidad del modelo basada en valores de Shapley (Teoría de Juegos) |
| **Matplotlib / Seaborn** | Visualizaciones estadísticas y curvas de rendimiento |

---

## 📈 5. Hallazgos Clave & Métricas de Negocio

### Desempeño del Modelo en el Conjunto de Test (1,407 clientes no vistos)
* **ROC-AUC:** **0.8383** (excelente capacidad de ordenamiento por riesgo).
* **Recall (Clase 1 - Churn):** **0.79 (79%)**. El modelo intercepta con éxito a casi 4 de cada 5 clientes que efectivamente van a abandonar el servicio.
* **Precision (Clase 1):** **0.49 (49%)**. Trade-off deliberado: se acepta una tasa moderada de falsas alarmas a cambio de maximizar la captura de clientes en riesgo crítico.

### Insights Clave Revelados por SHAP:
1. **Contrato Mes a Mes (`Contract_Month-to-month`):** Es el catalizador de abandono #1. Clientes sin compromiso anual tienen una probabilidad exponencialmente más alta de irse ante cualquier fricción.
2. **Antigüedad del Cliente (`tenure`):** Clientes con alta antigüedad presentan valores SHAP fuertemente negativos (están fidelizados). El mayor riesgo se concentra en los primeros 1 a 3 meses de servicio.
3. **Fibra Óptica sin Soporte Técnico:** Los usuarios que pagan tarifas altas por fibra pero no cuentan con soporte técnico contratado muestran una propensión severa al abandono tras incidentes de conectividad.

---

## 🚀 6. Guía Rápida de Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/<tu-usuario>/customer-churn-xgboost-shap.git
cd customer-churn-xgboost-shap
```

### 2. Crear y activar entorno virtual
```bash
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux / MacOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Abrir el notebook interactivo
```bash
jupyter notebook notebooks/01_churn_prediction_y_shap.ipynb
```
*(Nota: El notebook ya se encuentra pre-renderizado con todos los gráficos de Scikit-Learn y SHAP para visualización inmediata en GitHub o nbviewer).*
