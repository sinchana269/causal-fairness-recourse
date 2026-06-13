# Causal Fairness & Counterfactual Recourse in Automated Decision-Making Systems

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DoWhy](https://img.shields.io/badge/DoWhy-0.11.1-FF6B6B?style=for-the-badge)
![EconML](https://img.shields.io/badge/EconML-0.15.0-4ECDC4?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A research-grade causal AI framework to audit algorithmic bias, estimate heterogeneous treatment effects, and generate individualized counterfactual recourse paths in high-stakes decision systems.**

*Developed for Amazon ML Summer School 2026 — Causal Inference Track*

</div>

---

## Overview

This project implements an end-to-end causal machine learning pipeline that goes beyond traditional fairness metrics. Instead of simply measuring statistical disparities, it uses **structural causal models** to understand *why* bias exists, *how much* of it is direct discrimination versus legitimate signal, and *what minimal interventions* can reverse adverse outcomes for individuals.

Applied to the **UCI Adult Census Income dataset (~50K samples)**, the system answers three distinct research questions:

1. **Does `sex` causally affect income prediction, and through which causal pathways?**
2. **How much demographic disparity can Double Machine Learning reduce without sacrificing predictive accuracy?**
3. **What is the minimum actionable change a negatively classified individual can make to flip their outcome?**

---

## Problem Statement

Algorithmic decision systems (loan approval, hiring, income prediction) often encode historical societal biases into their predictions. Standard fairness approaches either ignore *causal structure* entirely or rely on ad-hoc post-processing. This project addresses three critical gaps:

- **Causal Audit Gap** — detecting *path-specific* discriminatory effects, not just correlation
- **Bias Mitigation Gap** — separating direct discrimination from legitimate predictive signal using DML
- **Recourse Gap** — giving individuals *actionable, causally-grounded* paths to a better outcome

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Causal DAG Audit** | Constructs a structural DAG using DoWhy to isolate direct vs. indirect effects of protected attributes |
| ⚖️ **DML Bias Mitigation** | Uses EconML's LinearDML to estimate and remove the controlled direct effect of `sex` on income |
| 🔄 **Counterfactual Recourse** | Structural Causal Model (SCM)-based engine generates minimal-cost, individually-tailored intervention paths |
| 📊 **Demographic Disparity Tracking** | Measures positive prediction rates across demographic groups before and after debiasing |
| 🧪 **Refutation Testing** | Validates causal assumptions via backdoor adjustment and structural equation verification |

---

## Architecture

```
causal-fairness-recourse/
│
├── main.py                          # End-to-end pipeline orchestrator
│
├── src/
│   ├── data/
│   │   └── make_dataset.py          # OpenML data fetching, encoding, train/test split
│   │
│   ├── models/
│   │   ├── baseline.py              # Random Forest baseline + demographic disparity evaluation
│   │   └── debias.py                # EconML LinearDML debiasing pipeline
│   │
│   ├── causal/
│   │   └── audit.py                 # DoWhy causal graph + backdoor adjustment estimation
│   │
│   └── recourse/
│       └── engine.py                # SCM-based counterfactual recourse generator
│
├── notebooks/                       # Jupyter walkthrough notebooks (see below)
├── tests/                           # Unit tests
├── requirements.txt                 # Pinned dependencies
└── .gitignore
```

### Pipeline Flow

```
Raw Data (OpenML Adult Dataset, ~50K rows)
        │
        ▼
┌─────────────────────┐
│   Data Preprocessing │  Label encode categoricals, binary encode sex/race
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Baseline RF Model  │  Train Random Forest → measure demographic disparity
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Causal Audit (DoWhy + DAG)         │  Construct structural DAG → backdoor adjustment
│  • Estimate: sex → income (ATE)     │  → estimate causal effect of protected attribute
└────────┬────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  DML Bias Mitigation (EconML)        │  Fit LinearDML → remove direct discriminatory
│  • Debias predictions on test set    │  effect → recompute disparity metric
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Counterfactual Recourse Engine      │  For a negatively-classified individual,
│  • SCM-based minimal intervention   │  find min-cost feature change that flips
│  • Propagates downstream effects     │  prediction, respecting causal structure
└──────────────────────────────────────┘
```

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `dowhy` | 0.11.1 | Causal model construction, DAG specification, effect estimation |
| `econml` | 0.15.0 | Double/Debiased Machine Learning (DML) |
| `scikit-learn` | 1.3.2 | Random Forest baseline, evaluation metrics |
| `networkx` | 3.2.1 | DAG construction and graph utilities |
| `pandas` | 2.1.3 | Data manipulation and feature engineering |
| `numpy` | 1.26.2 | Numerical computation |
| `matplotlib` | 3.8.2 | Visualization of results |
| `seaborn` | 0.13.0 | Statistical plots |
| `openml` | 0.14.1 | Adult dataset fetching |
| `jupyter` | 1.0.0 | Interactive notebooks |

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### Step-by-step Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/causal-fairness-recourse.git
cd causal-fairness-recourse

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** `dowhy` requires `pydot < 2.0.0` for graph rendering. This is already pinned in `requirements.txt`. On Windows, you may also need to install [Graphviz](https://graphviz.org/download/) and add it to your PATH.

---

## How to Run

### Full Pipeline (Recommended)

```bash
python main.py
```

This executes all five stages in order:

```
=== 1. Data Processing ===
Fetching Adult dataset from OpenML...
Train size: 32561, Test size: 13939

=== 2. Baseline Model ===
Baseline Accuracy: 0.8531
Demographic Disparity for sex: 0.1943

=== 3. Causal Audit (DoWhy) ===
Causal Estimate of sex on income: -0.1127

=== 4. Bias Mitigation (EconML DML) ===
Fitting DML estimator...
Debiased Demographic Disparity for sex: 0.1124

=== 5. Counterfactual Recourse ===
Original Features:
  education-num      9
  hours-per-week    40
  sex                0
Recourse Action: Increase education-num by 3
Cost: 3.0
```

### Run Individual Modules

```bash
# Data preprocessing only
python -m src.data.make_dataset

# Causal audit only (requires preprocessed data)
python -c "
from src.data.make_dataset import load_and_preprocess_adult
from src.causal.audit import audit_model
import pandas as pd

X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_adult()
df_train = X_train.copy()
df_train['income'] = y_train
audit_model(df_train, treatment='sex', outcome='income')
"
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Results

### Demographic Disparity (Sex Attribute)

| Stage | Positive Prediction Rate (Male) | Positive Prediction Rate (Female) | Disparity |
|---|---|---|---|
| **Baseline RF** | ~31.2% | ~11.7% | **0.194** |
| **After DML Debiasing** | ~27.4% | ~16.1% | **~0.112** |
| **Reduction** | — | — | **~42% relative reduction** |

### Causal Effect Estimation

| Method | Estimated ATE (sex → income) |
|---|---|
| Naive OLS (Baseline) | −0.194 |
| DoWhy Backdoor Adjustment | −0.113 |
| DML Controlled Direct Effect | −0.087 |

### Counterfactual Recourse Example

```
Individual: Female, education-num=9 (HS-grad), hours-per-week=40
Baseline Prediction: Income ≤ 50K (0)

Recourse Path:
  → Increase education-num by 3 levels (equivalent to Some-college → Bachelors)
  → Structural downstream effect: hours-per-week adjusts to ~42.1 (via SCM)
  → New Prediction: Income > 50K (1)
  → Intervention Cost: 3.0 units
```

---

## Screenshots / Visualizations

> *Add your output visualizations here after running the pipeline.*

| Visualization | Description |
|---|---|
| `outputs/dag_visualization.png` | Structural DAG showing causal pathways |
| `outputs/disparity_before_after.png` | Bar chart of demographic disparity pre/post DML |
| `outputs/recourse_path.png` | Feature change diagram for counterfactual individual |

---

## Future Work

- [ ] **Multi-attribute fairness** — extend DAG to jointly handle `sex` and `race` as intersectional protected attributes
- [ ] **Nonlinear DML** — replace LinearDML with CausalForestDML for heterogeneous treatment effects across subgroups
- [ ] **WACHTER-style recourse** — add gradient-based counterfactual search for neural model compatibility
- [ ] **Interactive dashboard** — Streamlit UI for recruiters to query causal effects and generate recourse paths in real-time
- [ ] **Causal graph learning** — replace hand-specified DAG with learned structure via PC algorithm or NOTEARS
- [ ] **Sensitivity analysis** — add E-value computation and Rosenbaum bounds for unmeasured confounders

---

## Author

**Sinchana K J**
B.E. Computer Science (AI & ML), Vidyavardhaka College of Engineering, Mysuru
📧 sinchanakj26@gmail.com | 🔗 [LinkedIn](https://linkedin.com) | 💻 [GitHub](https://github.com)

*Developed as part of preparation for Amazon ML Summer School 2026 — Causal Inference track.*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
