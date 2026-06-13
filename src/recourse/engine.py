import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class StructuralRecourseEngine:
    def __init__(self, model, X_train, actionable_features):
        """
        model: Trained predictive model
        X_train: Training data to fit structural equations
        actionable_features: list of features that can be intervened upon
        """
        self.model = model
        self.actionable_features = actionable_features
        
        # Fit a simple structural equation model for downstream effects
        # Example: education_num causally affects hours_per_week
        # H = a * E + b
        self.scm = {}
        if 'education-num' in actionable_features and 'hours-per-week' in X_train.columns:
            # We assume hours_per_week is a descendant of education_num in the DAG
            reg = LinearRegression()
            # Simple 1D regression
            E = X_train[['education-num']]
            H = X_train['hours-per-week']
            reg.fit(E, H)
            self.scm['education_num_to_hours'] = reg.coef_[0]

    def generate_recourse(self, x, target_class=1):
        """
        Finds the minimal intervention to flip the prediction to target_class
        utilizing the structural causal model.
        """
        original_pred = self.model.predict(pd.DataFrame([x]))[0]
        if original_pred == target_class:
            return x, 0.0, "No intervention needed"
            
        best_cf = None
        min_cost = float('inf')
        best_action = ""
        
        # We test interventions on actionable features
        for feature in self.actionable_features:
            cf = x.copy()
            cost = 0
            
            # Simple grid search for continuous/ordinal variables
            for step in range(1, 10):
                cf[feature] += 1
                cost += 1
                
                # Apply structural downstream effects
                if feature == 'education-num' and 'education_num_to_hours' in self.scm:
                    # Increase in education causes increase in hours-per-week
                    cf['hours-per-week'] += self.scm['education_num_to_hours']
                
                pred = self.model.predict(pd.DataFrame([cf]))[0]
                if pred == target_class:
                    if cost < min_cost:
                        min_cost = cost
                        best_cf = cf.copy()
                        best_action = f"Increase {feature} by {step}"
                    break
                    
        return best_cf, min_cost, best_action
