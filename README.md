# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability

The Basel II Accord emphasizes accurate risk measurement, transparency, and proper documentation in credit risk management. Financial institutions must be able to justify lending decisions and demonstrate how risk estimates are produced. As a result, credit scoring models should be interpretable, well-documented, and easy to validate. Model assumptions, feature engineering steps, and evaluation results must be clearly recorded to support regulatory compliance and ongoing monitoring.

## 2. Need for a Proxy Target Variable

The provided dataset does not contain a direct indicator of loan default. Since supervised machine learning models require a target variable, a proxy target must be created. In this project, customer behavior will be analyzed using Recency, Frequency, and Monetary (RFM) metrics to identify potentially high-risk customers.

However, proxy-based targets introduce business risks because they do not represent actual defaults. Some customers classified as high risk may repay loans successfully, while some classified as low risk may default. Therefore, model predictions should be interpreted as estimates based on behavioral patterns rather than confirmed default outcomes.

## 3. Trade-Offs Between Interpretable and High-Performance Models

Logistic Regression combined with Weight of Evidence (WoE) provides a highly interpretable framework that allows analysts and regulators to understand how each feature influences risk predictions. This transparency supports regulatory compliance and easier model validation.

In contrast, Gradient Boosting models such as XGBoost or LightGBM often achieve higher predictive performance by capturing complex non-linear relationships in the data. However, these models are more difficult to interpret and explain to regulators and business stakeholders.

In a regulated financial environment, the choice of model requires balancing predictive performance with transparency, interpretability, and governance requirements.
# credit-risk-model
