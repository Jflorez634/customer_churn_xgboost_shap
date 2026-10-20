# 🧠 Apuntes para Notion: Semana 2 — Churn Prediction con XGBoost y SHAP
> **Área:** Data Science & Machine Learning  
> **Tema:** Clasificación con Desbalance, Explicabilidad (XAI) con SHAP y Optimización de Costes  
> **Dificultad:** ⭐⭐⭐⭐☆  
> **Tecnologías:** `Python`, `Scikit-Learn`, `XGBoost`, `SHAP`, `Seaborn`  
> **Estado:** ⚪ *Planeado*

---

## 🎯 1. Elevator Pitch (Para Entrevistas)
> *"En este proyecto desarrollé un modelo predictivo de Customer Churn utilizando XGBoost para una compañía de telecomunicaciones. En lugar de optimizar por métricas estándar como Accuracy, diseñé una matriz de decisión costo-beneficio que equilibra el costo de retener a un cliente frente al costo de perderlo. Además, integré explicabilidad con SHAP para desglosar a nivel individual qué factores impulsan la decisión de cada usuario, permitiendo al equipo de marketing ejecutar acciones hiper-personalizadas."*

---

## 📋 2. Checklist de Desarrollo Paso a Paso

- [ ] **Fase 1: EDA & Análisis de Fuga**
  - [ ] Calcular la tasa base de churn (ej. 26.5% abandono).
  - [ ] Explorar la correlación entre antigüedad del cliente (*tenure*) y probabilidad de fuga.
  - [ ] Analizar el impacto de servicios adicionales (soporte técnico, seguridad online, facturación electrónica).
- [ ] **Fase 2: Preprocesamiento & Pipelines**
  - [ ] Codificación One-Hot para variables nominales multiclase.
  - [ ] Escalado estándar / robusto para variables numéricas continuas.
  - [ ] `train_test_split` estratificado.
- [ ] **Fase 3: Modelado & Comparativa**
  - [ ] Modelo base (Regresión Logística / Árbol de Decisión simple).
  - [ ] Modelo avanzado (XGBoost Classifier con ajuste de hiperparámetros).
  - [ ] Evaluación mediante ROC-AUC y Precision-Recall AUC (PR-AUC).
- [ ] **Fase 4: Explicabilidad con SHAP**
  - [ ] Generar el `TreeExplainer` de SHAP.
  - [ ] Graficar el `summary_plot` (visión global de qué variables empujan hacia churn y cuáles hacia retención).
  - [ ] Crear un `waterfall_plot` para 2 clientes reales (uno con riesgo inminente y otro fidelizado).
- [ ] **Fase 5: Documentación y Resultados de Negocio**
  - [ ] Cuantificar el ahorro en dólares o moneda local según la simulación de retención.

---

## 💡 3. Conceptos Técnicos Clave

### A. Valores Shapley y SHAP
* Provienen de la **teoría de juegos cooperativos** (Premio Nobel Lloyd Shapley). Miden la contribución marginal de cada jugador (en este caso, cada variable predictora) al resultado del juego (la predicción del modelo).
* A diferencia de `feature_importances_` tradicional de los árboles (que solo indica qué variable se usó más para dividir nodos), SHAP indica **la dirección y magnitud** del impacto para cada fila individual.

### B. Curva Precision-Recall vs Curva ROC en Clases Desbalanceadas
* La curva ROC puede ser engañosamente optimista cuando los negativos verdaderos son muy numerosos.
* La curva **Precision-Recall** se enfoca exclusivamente en la clase positiva (clientes que se van), siendo mucho más informativa para ajustar el umbral de corte comercial.

---

## 🎤 4. Preguntas Típicas de Entrevista Técnica

### P1: *"¿Por qué no usaste directamente el umbral por defecto de 0.5 para predecir si un cliente se marcha?"*
> **Respuesta recomendada:** *"Porque el umbral de 0.5 asume implícitamente que el costo de un Falso Negativo (perder un cliente de $1,000) es idéntico al de un Falso Positivo (ofrecer un descuento de $20 a un cliente que se iba a quedar). Al modelar los costos económicos reales, encontramos que un umbral de 0.38 maximiza la rentabilidad neta de la campaña de retención."*
