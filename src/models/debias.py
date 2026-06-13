from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

def debias_with_dml(X_train, y_train, X_test, y_test, treatment_col='sex'):
    """
    Uses Double Machine Learning (DML) via EconML to estimate controlled direct effects
    and adjust predictions to reduce bias.
    """
    T_train = X_train[treatment_col]
    Y_train = y_train
    W_train = X_train.drop(columns=[treatment_col])
    
    T_test = X_test[treatment_col]
    W_test = X_test.drop(columns=[treatment_col])
    
    # Initialize DML estimator
    # We use regressors for model_y since we are predicting probabilities internally in DML
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        discrete_treatment=True,
        random_state=42
    )
    
    print("Fitting DML estimator...")
    est.fit(Y_train, T_train, X=None, W=W_train)
    
    # Baseline predictions
    baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    baseline.fit(X_train, y_train)
    preds_proba = baseline.predict_proba(X_test)[:, 1]
    
    # Estimate treatment effect on the test set
    te_test = est.effect(X=None, T0=0, T1=1)
    
    # Debias by adjusting probabilities
    # We remove the estimated direct effect of the protected attribute
    adjusted_preds_proba = preds_proba - (T_test * te_test)
    adjusted_preds_proba = np.clip(adjusted_preds_proba, 0, 1)
    
    # Find optimal threshold to maintain overall accuracy while reducing disparity
    # We'll stick to 0.5 for simplicity
    adjusted_preds = (adjusted_preds_proba >= 0.5).astype(int)
    
    df_eval = X_test.copy()
    df_eval['pred'] = adjusted_preds
    
    group_1_rate = df_eval[df_eval[treatment_col] == 1]['pred'].mean()
    group_0_rate = df_eval[df_eval[treatment_col] == 0]['pred'].mean()
    
    disparity = abs(group_1_rate - group_0_rate)
    print(f"Debiased Demographic Disparity for {treatment_col}: {disparity:.4f}")
    
    return est, adjusted_preds, disparity
