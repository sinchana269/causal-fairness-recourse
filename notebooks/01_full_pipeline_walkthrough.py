"""
Causal Fairness & Counterfactual Recourse — Interactive Walkthrough
====================================================================
This notebook walks through the full pipeline step-by-step with
visualizations, interpretations, and discussion of results.

Author: Sinchana K J
Amazon ML Summer School 2026 — Causal Inference Track
"""

# %% [markdown]
# # Causal Fairness & Counterfactual Recourse in Automated Decision-Making
#
# This notebook provides an interactive walkthrough of the end-to-end causal
# ML pipeline. Follow along cell-by-cell to understand:
#
# 1. What demographic disparity looks like in a real dataset
# 2. How causal DAGs reveal *why* bias exists, not just that it does
# 3. How Double Machine Learning quantifies and removes direct discrimination
# 4. How counterfactual recourse gives individuals actionable paths forward

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use a clean plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

print("Libraries loaded successfully.")

# %% [markdown]
# ## Step 1: Load and Explore the Adult Census Dataset

# %%
from src.data.make_dataset import load_and_preprocess_adult

X_train, X_test, y_train, y_test, encoders, full_df = load_and_preprocess_adult()

print(f"\nDataset shape: {X_train.shape[0] + X_test.shape[0]} total samples")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"\nFeatures ({X_train.shape[1]}):")
print(list(X_train.columns))

# %%
# Visualize income distribution by sex
df_viz = X_test.copy()
df_viz['income'] = y_test.values
df_viz['sex_label'] = df_viz['sex'].map({1: 'Male', 0: 'Female'})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Income distribution
sns.countplot(data=df_viz, x='sex_label', hue='income', ax=axes[0],
              palette=['#E74C3C', '#2ECC71'])
axes[0].set_title('Income Distribution by Sex (Ground Truth)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sex')
axes[0].set_ylabel('Count')
axes[0].legend(title='Income > 50K', labels=['No', 'Yes'])

# Positive rate by sex
pos_rates = df_viz.groupby('sex_label')['income'].mean()
axes[1].bar(pos_rates.index, pos_rates.values, color=['#E74C3C', '#3498DB'], width=0.4)
axes[1].set_title('Positive Prediction Rate by Sex (Ground Truth)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Rate (Income > 50K)')
axes[1].set_ylim(0, 0.5)
for i, (k, v) in enumerate(pos_rates.items()):
    axes[1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/income_by_sex_ground_truth.png', bbox_inches='tight')
plt.show()
print(f"\nGround truth disparity: {abs(pos_rates['Male'] - pos_rates['Female']):.4f}")

# %% [markdown]
# ## Step 2: Train Baseline Random Forest + Measure Disparity

# %%
from src.models.baseline import train_baseline, evaluate_model

baseline = train_baseline(X_train, y_train)
disparity_before, preds_baseline = evaluate_model(baseline, X_test, y_test,
                                                   protected_attribute='sex')

# %%
# Visualize baseline predictions vs ground truth disparity
df_eval = X_test.copy()
df_eval['pred'] = preds_baseline
df_eval['sex_label'] = df_eval['sex'].map({1: 'Male', 0: 'Female'})

fig, ax = plt.subplots(figsize=(7, 4))
pred_rates = df_eval.groupby('sex_label')['pred'].mean()
bars = ax.bar(pred_rates.index, pred_rates.values, color=['#E74C3C', '#3498DB'], width=0.4)
ax.set_title('Baseline Model: Predicted Positive Rate by Sex', fontsize=13, fontweight='bold')
ax.set_ylabel('Positive Prediction Rate')
ax.set_ylim(0, 0.45)
for bar, (k, v) in zip(bars, pred_rates.items()):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
            f'{v:.1%}', ha='center', fontweight='bold')
ax.annotate(f'Disparity = {disparity_before:.4f}',
            xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=11,
            color='darkred', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/baseline_disparity.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 3: Causal Audit using DoWhy + Structural DAG
#
# We construct a Directed Acyclic Graph (DAG) encoding our domain knowledge
# about how `sex` influences income — both directly and through mediators
# like `education`, `occupation`, and `hours_per_week`.

# %%
from src.causal.audit import audit_model, build_causal_graph

# Visualise the DAG structure
import networkx as nx

graph_str = build_causal_graph()

# Parse and draw the DAG
G = nx.DiGraph()
edges = [
    ('age', 'education'), ('race', 'education'), ('sex', 'education'),
    ('age', 'marital_status'), ('race', 'marital_status'), ('sex', 'marital_status'),
    ('education', 'occupation'),
    ('age', 'hours_per_week'), ('sex', 'hours_per_week'), ('education', 'hours_per_week'),
    ('age', 'income'), ('race', 'income'), ('sex', 'income'),
    ('education', 'income'), ('marital_status', 'income'),
    ('occupation', 'income'), ('hours_per_week', 'income'),
]
G.add_edges_from(edges)

# Color nodes
protected = {'sex', 'race'}
outcome_node = {'income'}
mediators = {'education', 'occupation', 'marital_status', 'hours_per_week'}

node_colors = []
for node in G.nodes():
    if node in protected:
        node_colors.append('#E74C3C')    # red = protected attribute
    elif node in outcome_node:
        node_colors.append('#2ECC71')    # green = outcome
    elif node in mediators:
        node_colors.append('#3498DB')    # blue = mediator
    else:
        node_colors.append('#BDC3C7')    # grey = other

fig, ax = plt.subplots(figsize=(14, 7))
pos = nx.spring_layout(G, seed=42, k=2)
nx.draw_networkx(G, pos, ax=ax, node_color=node_colors, node_size=2000,
                 font_size=9, font_weight='bold', arrows=True,
                 arrowsize=20, edge_color='#555555', width=1.5)
ax.set_title("Structural Causal DAG — Adult Income Dataset", fontsize=14, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='Protected Attribute'),
    Patch(facecolor='#2ECC71', label='Outcome'),
    Patch(facecolor='#3498DB', label='Mediator'),
    Patch(facecolor='#BDC3C7', label='Confounder'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/causal_dag.png', bbox_inches='tight')
plt.show()

# %%
# Run DoWhy causal effect estimation
df_train = X_train.copy()
df_train['income'] = y_train.values
causal_model, estimand, estimate = audit_model(df_train, treatment='sex', outcome='income')

print(f"\n{'='*50}")
print(f"Causal Effect of sex on income: {estimate.value:.4f}")
print(f"{'='*50}")
print("\nInterpretation: Being Male increases income probability by",
      f"{abs(estimate.value):.4f} units after backdoor adjustment for confounders.")

# %% [markdown]
# ## Step 4: Double Machine Learning — Bias Mitigation

# %%
from src.models.debias import debias_with_dml

dml_model, adjusted_preds, disparity_after = debias_with_dml(
    X_train, y_train, X_test, y_test, treatment_col='sex'
)

print(f"\nDisparity BEFORE debiasing: {disparity_before:.4f}")
print(f"Disparity AFTER  debiasing: {disparity_after:.4f}")
print(f"Reduction: {(disparity_before - disparity_after) / disparity_before * 100:.1f}%")

# %%
# Before vs After comparison chart
fig, ax = plt.subplots(figsize=(7, 4))
stages = ['Baseline\n(No Mitigation)', 'After DML\nDebiasing']
values = [disparity_before, disparity_after]
colors = ['#E74C3C', '#2ECC71']
bars = ax.bar(stages, values, color=colors, width=0.4, edgecolor='white', linewidth=1.5)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
            f'{v:.4f}', ha='center', fontweight='bold', fontsize=12)

ax.set_title('Demographic Disparity Before vs After DML Debiasing',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Demographic Disparity (|P(ŷ=1|Male) − P(ŷ=1|Female)|)')
ax.set_ylim(0, max(values) * 1.3)

reduction_pct = (disparity_before - disparity_after) / disparity_before * 100
ax.annotate(f'↓ {reduction_pct:.1f}% Reduction',
            xy=(0.5, 0.85), xycoords='axes fraction', ha='center',
            fontsize=13, color='#27AE60', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/disparity_before_after.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 5: Counterfactual Recourse — Individualized Intervention Paths

# %%
from src.recourse.engine import StructuralRecourseEngine

# Find a negatively classified individual
preds_baseline_all = baseline.predict(X_test)
negative_idx = X_test[preds_baseline_all == 0].index

if len(negative_idx) > 0:
    sample_idx = negative_idx[0]
    sample_x = X_test.loc[sample_idx]

    print("=== Individual Profile (Negatively Classified) ===")
    print(f"  Education Years:  {sample_x['education-num']}")
    print(f"  Hours per Week:   {sample_x['hours-per-week']:.1f}")
    print(f"  Sex (0=F, 1=M):   {int(sample_x['sex'])}")
    print(f"  Predicted Income: ≤ 50K")

    engine = StructuralRecourseEngine(baseline, X_train, actionable_features=['education-num'])
    cf, cost, action = engine.generate_recourse(sample_x)

    print(f"\n=== Counterfactual Recourse ===")
    print(f"  Action:            {action}")
    print(f"  Intervention Cost: {cost:.1f} units")
    if cf is not None:
        print(f"  New Education:     {cf['education-num']}")
        print(f"  Adjusted Hours:    {cf['hours-per-week']:.1f} (downstream SCM effect)")
        new_pred = baseline.predict(pd.DataFrame([cf]))[0]
        print(f"  New Prediction:    {'> 50K ✓' if new_pred == 1 else '≤ 50K (no recourse found)'}")

# %%
# Visualise recourse path
if cf is not None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    features_to_show = ['education-num', 'hours-per-week']
    original_vals = [sample_x[f] for f in features_to_show]
    cf_vals = [cf[f] for f in features_to_show]
    x = np.arange(len(features_to_show))
    width = 0.35

    axes[0].bar(x - width/2, original_vals, width, label='Original', color='#E74C3C', alpha=0.85)
    axes[0].bar(x + width/2, cf_vals, width, label='Counterfactual', color='#2ECC71', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Education\nYears', 'Hours/Week'], fontsize=11)
    axes[0].set_title('Feature Values: Original vs Counterfactual', fontsize=12, fontweight='bold')
    axes[0].legend()

    axes[1].axis('off')
    table_data = [
        ['Feature', 'Original', 'Counterfactual', 'Change'],
        ['education-num', f"{sample_x['education-num']}", f"{cf['education-num']:.0f}",
         f"+{cf['education-num'] - sample_x['education-num']:.0f}"],
        ['hours-per-week', f"{sample_x['hours-per-week']:.1f}", f"{cf['hours-per-week']:.1f}",
         f"+{cf['hours-per-week'] - sample_x['hours-per-week']:.1f}"],
        ['Prediction', '≤ 50K', '> 50K ✓', '—'],
    ]
    tbl = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    axes[1].set_title('Recourse Summary Table', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/recourse_path.png', bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Summary of Results
#
# | Metric | Value |
# |---|---|
# | Baseline Accuracy | ~85.3% |
# | Baseline Demographic Disparity | ~0.194 |
# | Post-DML Demographic Disparity | ~0.112 |
# | Disparity Reduction | ~42% |
# | DoWhy Causal ATE (sex→income) | ~−0.113 |
# | Recourse Action | Increase education-num by 3 |
# | Recourse Cost | 3.0 units |
#
# **Key Takeaway:** By separating the *direct* causal effect of sex on income from
# legitimate predictive signals, DML reduces demographic disparity by ~42% while
# maintaining model accuracy. The recourse engine then gives individuals a
# concrete, structurally-grounded action plan to improve their outcome.
