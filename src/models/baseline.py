from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_baseline(X_train, y_train):
    """
    Trains a baseline LightGBM classifier.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, protected_attribute='sex'):
    """
    Evaluates the model and computes demographic disparity.
    """
    preds = model.predict(X_test)
    print("Baseline Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    df_eval = X_test.copy()
    df_eval['pred'] = preds
    
    group_1_rate = df_eval[df_eval[protected_attribute] == 1]['pred'].mean()
    group_0_rate = df_eval[df_eval[protected_attribute] == 0]['pred'].mean()
    
    disparity = abs(group_1_rate - group_0_rate)
    print(f"Demographic Disparity for {protected_attribute}: {disparity:.4f}")
    
    return disparity, preds
