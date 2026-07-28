# Salary Decision Support System

A machine learning-based **Decision Support System (DSS)** for salary valuation, peer benchmarking, offer assessment, career improvement, and salary negotiation.

The system is designed from the candidate's perspective and aims to transform data and machine learning predictions into **interpretable and actionable decision support**.

Rather than answering only:

> "What salary should this candidate earn?"

the system also addresses:

- What salary range is reasonable for this profile?
- How does the candidate compare with similar profiles?
- Is the current offer low, reasonable, or competitive?
- What salary should the candidate propose during negotiation?
- What profile improvements may increase expected salary?
- Why does the system make these recommendations?

---

## Dataset

The project uses the **250K Job Salary Prediction Dataset** from Kaggle.

The dataset contains **250,000 observations**, with candidate and job-related information such as:

- Job title
- Years of experience
- Education level
- Number of skills
- Industry
- Company size
- Location
- Remote work
- Number of certifications
- Salary

The data is split into:

- **200,000 training observations**
- **50,000 testing observations**

After preprocessing and one-hot encoding, the modeling dataset contains **48 input features**.

---

# System Pipeline

```text
Raw Salary Data
       │
       ▼
EDA & Preprocessing
       │
       ▼
1. Salary Valuation
   ├── LightGBM
   ├── Random Forest
   └── XGBoost
       │
       ▼
2. KNN Peer Benchmarking
       │
       ▼
3. Offer Assessment
       │
       ▼
4. Negotiation Optimization
       │
       ▼
5. What-if Career Recourse
       │
       ▼
6. Causal Action Ranking
       │
       ▼
7. SHAP Explainability
       │
       ▼
8. Offer Anomaly Detection
```

The overall workflow can be summarized as:

> **Understand → Predict → Compare → Evaluate → Recommend → Explain**

---

# 0. Data Understanding & Preparation

Before building the decision-support modules, the project first explores, cleans, transforms, and prepares the salary dataset.

## 0.1 Exploratory Data Analysis

Notebook:

[`0_EDA_Salary_DSS.ipynb`](notebooks/0_EDA_Salary_DSS.ipynb)

Exploratory Data Analysis is used to understand the structure of the salary dataset and identify variables that may be useful for salary valuation.

The analysis includes:

- Salary distribution
- Salary by job title
- Salary by years of experience
- Salary by education level
- Salary by number of skills
- Salary by industry
- Salary by company size
- Salary by location
- Salary by remote-work arrangement
- Salary by certifications
- Outlier analysis

EDA is not used only for visualization. The insights are also used to support later decisions such as peer benchmarking, offer classification, and career recommendations.

---

## 0.2 Data Preprocessing

Notebook:

[`0_Preprocessing_Salary_DSS.ipynb`](notebooks/0_Preprocessing_Salary_DSS.ipynb)

The preprocessing pipeline prepares the raw dataset for machine learning and downstream DSS modules.

Main steps include:

- Missing-value inspection
- Duplicate detection
- Salary outlier analysis
- Feature engineering
- Train/test splitting
- Numerical feature imputation
- Standardization
- Categorical feature encoding

Numerical variables are processed using:

```text
Median Imputation
        ↓
Standard Scaling
```

Categorical variables are processed using:

```text
Most Frequent Imputation
        ↓
One-Hot Encoding
```

The preprocessing pipeline is fitted only on the training set to reduce data leakage and ensure consistent transformation of future candidate profiles.

---

# 1. Salary Valuation

The first core module estimates the market salary of a candidate based on their profile.

Instead of relying on a single algorithm, three tree-based regression models are trained and compared:

- LightGBM
- Random Forest
- XGBoost

The prediction layer forms the analytical foundation for the later decision-support modules.

---

## 1.1 LightGBM Regressor

Notebook:

[`1. LGBM_Regressor.ipynb`](notebooks/1.%20LGBM_Regressor.ipynb)

LightGBM is used to model nonlinear relationships between salary and candidate/job characteristics.

The model is suitable for large tabular datasets and provides the predictive foundation for the salary valuation component of the DSS.

LightGBM is also used in the broader system architecture for estimating salary ranges and supporting downstream recommendations.

---

## 1.2 Random Forest Regressor

Notebook:

[`1. RF_Regressor.ipynb`](notebooks/1.%20RF_Regressor.ipynb)

Random Forest is implemented as an ensemble regression model for salary estimation.

The model provides a useful comparison with boosting-based approaches and captures nonlinear relationships and interactions between candidate characteristics.

---

## 1.3 XGBoost Regressor

Notebook:

[`1. XGB_Regressor.ipynb`](notebooks/1.%20XGB_Regressor.ipynb)

XGBoost is used as another gradient-boosting regression model for salary estimation.

The model is evaluated alongside LightGBM and Random Forest to compare predictive performance and determine an appropriate model for integration into the DSS.

---

## Model Comparison

The salary models are evaluated using regression metrics such as:

- MAE
- RMSE
- MAPE
- R²

The purpose of this stage is not only to identify the most accurate model, but also to create a reliable predictive foundation for later modules such as benchmarking, offer assessment, and negotiation optimization.

---

# 2. KNN Peer Benchmarking

Notebook:

[`2__KNN_Peer_Benchmark.ipynb`](notebooks/2__KNN_Peer_Benchmark.ipynb)

Salary prediction alone does not tell a candidate how they compare with people who have similar profiles.

This module therefore uses **k-Nearest Neighbors (kNN)** to identify comparable candidates.

The system searches for profiles that are similar in terms of candidate and job characteristics and summarizes their salary distribution.

The benchmark helps answer:

> **"What do people with profiles similar to mine typically earn?"**

The module provides information such as:

- Nearest comparable profiles
- Peer salary distribution
- Median peer salary
- Salary quantiles
- Relative position of the current candidate

The project uses **k = 7** nearest neighbors for the peer benchmark.

---

# 3. Offer Assessment

Notebook:

[`3__Offer_Assessment_System.ipynb`](notebooks/3__Offer_Assessment_System.ipynb)

Once salary valuation and peer benchmarking are available, the next step is to evaluate the candidate's current salary offer.

The offer is compared with:

- Predicted salary
- Estimated salary range
- Similar candidate profiles
- Relevant benchmark statistics

The system then converts these quantitative results into an interpretable assessment.

Possible outcomes include:

```text
Low Offer
Reasonable Offer
Good Offer
Potentially Unusual Offer
```

This module transforms model predictions into information that can directly support the decision to:

> **Accept → Negotiate → Reconsider**

---

# 4. Negotiation Optimization

Notebook:

[`4__Negotiation_Optimization_System.ipynb`](notebooks/4__Negotiation_Optimization_System.ipynb)

After evaluating the current offer, the system supports the candidate in selecting an appropriate salary target for negotiation.

The objective is **not simply to recommend the highest possible salary**.

Instead, the negotiation problem considers the trade-off between:

- Expected salary benefit
- Probability of reaching an agreement
- Distance from a reasonable salary range
- Negotiation risk
- User risk preference

Conceptually:

```text
Negotiation Utility
        =
Expected Benefit
        -
Risk Penalty
```

Different candidate salary requests are evaluated and the system recommends a level that balances expected benefit with negotiation risk.

This module extends predictive analytics into **prescriptive analytics**.

---

# 5. What-if Career Recourse

Notebook:

[`5__WhatIf_Career_Recourse.ipynb`](notebooks/5__WhatIf_Career_Recourse.ipynb)

This module asks a different question:

> **"What could the candidate change to improve their expected salary?"**

The system generates hypothetical candidate profiles and evaluates how salary predictions change under different scenarios.

Possible scenarios include changes in:

- Skills
- Certifications
- Experience
- Education
- Job characteristics
- Company characteristics

The module can therefore compare:

```text
Current Profile
      ↓
Predicted Salary

versus

Modified Profile
      ↓
New Predicted Salary
```

The goal is to transform salary prediction into actionable career-development scenarios.

---

# 6. Causal Action Ranking

Notebook:

[`6__Causal_Action_Ranking.ipynb`](notebooks/6__Causal_Action_Ranking.ipynb)

After generating possible career actions, the system attempts to prioritize them according to their estimated impact.

The module explores **causal modeling**, including Causal Forest-based approaches, to estimate heterogeneous effects of possible actions.

Example questions include:

- Would adding a certification potentially improve expected salary?
- Which profile improvement should be prioritized?
- Do different candidate profiles benefit differently from the same action?

The output is used to rank potential actions and support more personalized recommendations.

> **Note:** Because the project uses observational data, causal results should be interpreted cautiously. Estimated effects depend on the assumptions required for causal identification.

---

# 7. SHAP Explainability

Notebook:

[`7_shap_explanation.ipynb`](notebooks/7_shap_explanation.ipynb)

The system should not provide salary estimates or recommendations as unexplained black-box outputs.

SHAP is therefore used to analyze how individual features contribute to model predictions.

The module supports:

### Global explanations

Identify which variables are generally important across the dataset.

### Local explanations

Explain why a particular candidate receives a specific salary estimate.

For each feature, SHAP indicates whether it contributes to:

```text
Higher predicted salary ↑
or
Lower predicted salary ↓
```

This helps connect machine learning predictions with the recommendation layer of the DSS.

---

# 8. Offer Anomaly Detection

Notebook:

[`8_offer_anomaly_detection.ipynb`](notebooks/8_offer_anomaly_detection.ipynb)

The final analytical module focuses on detecting salary offers that behave unusually relative to candidate characteristics and market patterns.

Anomaly detection is used to identify offers that may be substantially different from what the model and comparable profiles would normally suggest.

This helps distinguish between:

```text
A slightly low offer

and

An unusually low offer
```

The module provides an additional diagnostic signal to support offer assessment and negotiation decisions.

---

# Decision Support Flow

The complete system transforms a candidate profile into progressively richer information:

```text
Candidate Profile
       ↓
Salary Estimation
       ↓
Peer Comparison
       ↓
Offer Evaluation
       ↓
Negotiation Strategy
       ↓
Career Improvement Scenarios
       ↓
Action Prioritization
       ↓
Prediction Explanation
       ↓
Anomaly Diagnostics
```

The project therefore goes beyond:

> **"Predict the salary."**

and focuses on:

> **"Use salary modeling to support a better decision."**

---

# Project Structure

```text
salary-decision-support-system/
│
├── data/
│
├── notebooks/
│   │
│   ├── 0_EDA_Salary_DSS.ipynb
│   ├── 0_Preprocessing_Salary_DSS.ipynb
│   │
│   ├── 1. LGBM_Regressor.ipynb
│   ├── 1. RF_Regressor.ipynb
│   ├── 1. XGB_Regressor.ipynb
│   │
│   ├── 2__KNN_Peer_Benchmark.ipynb
│   ├── 3__Offer_Assessment_System.ipynb
│   ├── 4__Negotiation_Optimization_System.ipynb
│   ├── 5__WhatIf_Career_Recourse.ipynb
│   ├── 6__Causal_Action_Ranking.ipynb
│   ├── 7_shap_explanation.ipynb
│   └── 8_offer_anomaly_detection.ipynb
│
├── report/
│   └── Baocao_DSS.pdf
│
└── README.md
```

---

# Tech Stack

### Data Processing

Python · Pandas · NumPy · Scikit-learn

### Machine Learning

LightGBM · XGBoost · Random Forest · k-Nearest Neighbors

### Decision Support

Peer Benchmarking · Offer Assessment · Prescriptive Optimization · What-if Analysis

### Explainability & Advanced Analysis

SHAP · Causal Modeling · Anomaly Detection

### Development

Jupyter Notebook · Git · GitHub

---

# Limitations

The current project is an academic prototype.

Important limitations include:

- The dataset is synthetic and does not fully represent real labor-market conditions.
- Salary relationships may change over time and across different labor markets.
- Recommendations are intended as decision-support information rather than definitive salary advice.
- Relationships observed in the data should not automatically be interpreted as causal.
- Real-world deployment would require updated salary data, model monitoring, fairness evaluation, and user testing.

---

# Future Work

Potential extensions include:

- Integrating real-world job-market and salary data
- Improving uncertainty estimation and prediction intervals
- Evaluating model fairness
- Developing an interactive web application
- Monitoring model performance over time
- Improving personalized career recommendations
- Evaluating negotiation strategies with real users

---

# Team

Developed as part of the **Decision Support Systems** course at the  
**Faculty of Mathematics and Informatics — Hanoi University of Science and Technology**.

### Team Members

- Phạm Khánh Hưởng
- Nguyễn Khắc Chí
- Trần Trung Kiên

---

# Report

Detailed methodology, experiments, system design, results, and discussion are available in the full project report:

[**View Full Project Report**](report/Baocao_DSS.pdf)
