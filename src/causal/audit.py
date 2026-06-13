import dowhy
import pandas as pd
import networkx as nx

def build_causal_graph_nx():
    """
    Constructs a structural DAG as a networkx DiGraph for the Adult dataset.
    Returns a networkx DiGraph — avoids pydot/graphviz dependency entirely.
    """
    G = nx.DiGraph()
    nodes = ['age', 'race', 'sex', 'education', 'marital_status',
             'occupation', 'hours_per_week', 'income']
    G.add_nodes_from(nodes)

    edges = [
        ('age',       'education'),
        ('race',      'education'),
        ('sex',       'education'),
        ('age',       'marital_status'),
        ('race',      'marital_status'),
        ('sex',       'marital_status'),
        ('education', 'occupation'),
        ('age',       'hours_per_week'),
        ('sex',       'hours_per_week'),
        ('education', 'hours_per_week'),
        ('age',       'income'),
        ('race',      'income'),
        ('sex',       'income'),
        ('education', 'income'),
        ('marital_status', 'income'),
        ('occupation',     'income'),
        ('hours_per_week', 'income'),
    ]
    G.add_edges_from(edges)
    return G


def audit_model(df, treatment='sex', outcome='income'):
    """
    Uses DoWhy to identify and estimate the causal effect of the protected
    attribute on income via backdoor adjustment on the structural DAG.
    """
    df_causal = df.copy()
    df_causal.columns = [c.replace('-', '_') for c in df_causal.columns]

    G = build_causal_graph_nx()

    model = dowhy.CausalModel(
        data=df_causal,
        treatment=treatment,
        outcome=outcome,
        graph=G               # pass networkx DiGraph directly — no pydot needed
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    print(f"Causal Estimate of {treatment} on {outcome}: {estimate.value:.4f}")
    return model, identified_estimand, estimate
