import pandas as pd
import warnings
warnings.filterwarnings('ignore')   # suppress FutureWarning from dowhy/pandas
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
    preds = baseline.predict(X_test)
    negative_idx = X_test[preds == 0].index

    engine = StructuralRecourseEngine(baseline, X_train, ['education-num'])

    # Try multiple negative samples until recourse is found
    recourse_found = False
    for sample_idx in negative_idx[:20]:     # try up to 20 candidates
        sample_x = X_test.loc[sample_idx]
        cf, cost, action = engine.generate_recourse(sample_x)
        if cf is not None and cost < float('inf'):
            recourse_found = True
            break

    print("Original Features (negatively classified individual):")
    print(f"  education-num  : {int(sample_x['education-num'])}")
    print(f"  hours-per-week : {int(sample_x['hours-per-week'])}")
    print(f"  sex (0=F,1=M)  : {int(sample_x['sex'])}")

    if recourse_found:
        print(f"\nRecourse Action : {action}")
        print(f"Intervention Cost: {cost:.1f} unit(s)")
        print("Counterfactual Features:")
        print(f"  education-num  : {int(cf['education-num'])}")
        print(f"  hours-per-week : {cf['hours-per-week']:.1f}")
        new_pred = baseline.predict(pd.DataFrame([cf]))[0]
        print(f"  New Prediction : {'Income > $50K [FLIPPED]' if new_pred == 1 else 'No flip'}")
    else:
        print("No recourse found within search range.")

if __name__ == "__main__":
    main()
