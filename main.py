import pandas as pd
from src.data.make_dataset import load_and_preprocess_adult
from src.models.baseline import train_baseline, evaluate_model
from src.causal.audit import audit_model
from src.models.debias import debias_with_dml
from src.recourse.engine import StructuralRecourseEngine
import os

def main():
    print("=== 1. Data Processing ===")
    X_train, X_test, y_train, y_test, encoders, full_df = load_and_preprocess_adult()
    
    print("\n=== 2. Baseline Model ===")
    baseline = train_baseline(X_train, y_train)
    evaluate_model(baseline, X_test, y_test, protected_attribute='sex')
    
    print("\n=== 3. Causal Audit (DoWhy) ===")
    # Merge X and y for DoWhy
    df_train = X_train.copy()
    df_train['income'] = y_train
    audit_model(df_train, treatment='sex', outcome='income')
    
    print("\n=== 4. Bias Mitigation (EconML DML) ===")
    dml_model, adjusted_preds, disparity = debias_with_dml(X_train, y_train, X_test, y_test, treatment_col='sex')
    
    print("\n=== 5. Counterfactual Recourse ===")
    # Identify a negatively classified instance
    preds = baseline.predict(X_test)
    negative_idx = X_test[preds == 0].index
    if len(negative_idx) > 0:
        sample_idx = negative_idx[0]
        sample_x = X_test.loc[sample_idx]
        print("Original Features:")
        print(sample_x[['education-num', 'hours-per-week', 'sex']])
        
        # Actionable features
        # Note: DoWhy graph changes hyphens to underscores, but baseline uses original names
        actionable_features = ['education-num']
        
        # SCM uses original names from X_train
        engine = StructuralRecourseEngine(baseline, X_train, actionable_features)
        cf, cost, action = engine.generate_recourse(sample_x)
        print(f"\nRecourse Action: {action}")
        print(f"Cost: {cost}")
        if cf is not None:
            print("Counterfactual Features:")
            print(cf[['education-num', 'hours-per-week', 'sex']])
        else:
            print("No recourse found.")

if __name__ == "__main__":
    main()
