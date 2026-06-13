"""
Unit tests for the Causal Fairness & Counterfactual Recourse pipeline.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────────────────────
# Tests: Data Preprocessing
# ─────────────────────────────────────────────────────────────
class TestDataPreprocessing:
    """Test the data loading and preprocessing module."""

    def test_output_shapes(self):
        """Train/test splits should have consistent feature counts."""
        from src.data.make_dataset import load_and_preprocess_adult
        X_train, X_test, y_train, y_test, encoders, df = load_and_preprocess_adult()
        assert X_train.shape[1] == X_test.shape[1], "Train/test feature count mismatch"
        assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
        assert len(X_test) == len(y_test), "X_test and y_test length mismatch"

    def test_binary_target(self):
        """Target variable must be binary (0 or 1)."""
        from src.data.make_dataset import load_and_preprocess_adult
        _, _, y_train, y_test, _, _ = load_and_preprocess_adult()
        assert set(y_train.unique()).issubset({0, 1}), "y_train contains non-binary values"
        assert set(y_test.unique()).issubset({0, 1}), "y_test contains non-binary values"

    def test_protected_attributes_encoded(self):
        """sex and race columns must be binary-encoded."""
        from src.data.make_dataset import load_and_preprocess_adult
        X_train, _, _, _, _, _ = load_and_preprocess_adult()
        assert set(X_train['sex'].unique()).issubset({0, 1}), "sex not binary encoded"
        assert set(X_train['race'].unique()).issubset({0, 1}), "race not binary encoded"

    def test_no_missing_values(self):
        """Processed features must have no NaN values."""
        from src.data.make_dataset import load_and_preprocess_adult
        X_train, X_test, _, _, _, _ = load_and_preprocess_adult()
        assert X_train.isnull().sum().sum() == 0, "X_train has missing values"
        assert X_test.isnull().sum().sum() == 0, "X_test has missing values"

    def test_expected_features_present(self):
        """All 13 expected features should be in the dataset."""
        from src.data.make_dataset import load_and_preprocess_adult
        X_train, _, _, _, _, _ = load_and_preprocess_adult()
        expected = ['age', 'workclass', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country']
        for col in expected:
            assert col in X_train.columns, f"Missing expected column: {col}"


# ─────────────────────────────────────────────────────────────
# Tests: Baseline Model
# ─────────────────────────────────────────────────────────────
class TestBaselineModel:
    """Test the Random Forest baseline model."""

    @pytest.fixture(scope="class")
    def data(self):
        from src.data.make_dataset import load_and_preprocess_adult
        return load_and_preprocess_adult()

    def test_model_trains(self, data):
        """Model should train without errors."""
        from src.models.baseline import train_baseline
        X_train, _, y_train, _, _, _ = data
        model = train_baseline(X_train, y_train)
        assert model is not None

    def test_model_accuracy_above_threshold(self, data):
        """Baseline accuracy should exceed 75% on Adult dataset."""
        from src.models.baseline import train_baseline, evaluate_model
        X_train, X_test, y_train, y_test, _, _ = data
        model = train_baseline(X_train, y_train)
        disparity, preds = evaluate_model(model, X_test, y_test)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, preds)
        assert acc > 0.75, f"Accuracy {acc:.3f} below expected 0.75 threshold"

    def test_demographic_disparity_positive(self, data):
        """Demographic disparity should be measurable (> 0) in unmitigated model."""
        from src.models.baseline import train_baseline, evaluate_model
        X_train, X_test, y_train, y_test, _, _ = data
        model = train_baseline(X_train, y_train)
        disparity, _ = evaluate_model(model, X_test, y_test, protected_attribute='sex')
        assert disparity > 0.0, "Expected non-zero disparity in baseline model"


# ─────────────────────────────────────────────────────────────
# Tests: Counterfactual Recourse Engine
# ─────────────────────────────────────────────────────────────
class TestRecourseEngine:
    """Test the StructuralRecourseEngine."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Set up a tiny synthetic dataset for fast unit testing."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'education-num': np.random.randint(5, 16, n),
            'hours-per-week': np.random.randint(20, 60, n),
            'sex': np.random.randint(0, 2, n),
        })
        y = ((X['education-num'] > 10) & (X['hours-per-week'] > 40)).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_engine_initialises(self, setup):
        """Engine should construct SCM coefficients without error."""
        from src.recourse.engine import StructuralRecourseEngine
        model, X_train, _ = setup
        engine = StructuralRecourseEngine(model, X_train, ['education-num'])
        assert 'education_num_to_hours' in engine.scm

    def test_no_intervention_when_already_positive(self, setup):
        """If individual is already positively classified, cost should be 0."""
        from src.recourse.engine import StructuralRecourseEngine
        model, X_train, y = setup
        engine = StructuralRecourseEngine(model, X_train, ['education-num'])
        # Find a positively classified instance
        preds = model.predict(X_train)
        positive_idx = X_train[preds == 1].index
        if len(positive_idx) > 0:
            sample = X_train.loc[positive_idx[0]]
            _, cost, action = engine.generate_recourse(sample, target_class=1)
            assert cost == 0.0
            assert action == "No intervention needed"

    def test_recourse_flips_prediction(self, setup):
        """Counterfactual should result in a positive prediction."""
        from src.recourse.engine import StructuralRecourseEngine
        model, X_train, y = setup
        engine = StructuralRecourseEngine(model, X_train, ['education-num'])
        # Find a negatively classified instance
        preds = model.predict(X_train)
        negative_idx = X_train[preds == 0].index
        if len(negative_idx) > 0:
            sample = X_train.loc[negative_idx[0]]
            cf, cost, action = engine.generate_recourse(sample, target_class=1)
            if cf is not None:
                new_pred = model.predict(pd.DataFrame([cf]))[0]
                assert new_pred == 1, "Counterfactual did not flip prediction to 1"

    def test_recourse_cost_is_positive(self, setup):
        """When recourse is found, cost should be > 0."""
        from src.recourse.engine import StructuralRecourseEngine
        model, X_train, y = setup
        engine = StructuralRecourseEngine(model, X_train, ['education-num'])
        preds = model.predict(X_train)
        negative_idx = X_train[preds == 0].index
        if len(negative_idx) > 0:
            sample = X_train.loc[negative_idx[0]]
            _, cost, _ = engine.generate_recourse(sample, target_class=1)
            if cost != float('inf'):
                assert cost > 0.0


# ─────────────────────────────────────────────────────────────
# Tests: Causal Audit
# ─────────────────────────────────────────────────────────────
class TestCausalAudit:
    """Test the DoWhy causal audit module."""

    def test_causal_graph_builds(self):
        """DAG string should be a valid non-empty digraph specification."""
        from src.causal.audit import build_causal_graph
        graph = build_causal_graph()
        assert "digraph" in graph
        assert "sex" in graph
        assert "income" in graph
        assert "sex -> income" in graph or "sex ->income" in graph or "sex->" in graph

    def test_audit_returns_estimate(self):
        """audit_model should return a numeric causal estimate."""
        from src.causal.audit import audit_model
        from src.data.make_dataset import load_and_preprocess_adult
        X_train, _, y_train, _, _, _ = load_and_preprocess_adult()
        df_train = X_train.copy()
        df_train['income'] = y_train
        model, estimand, estimate = audit_model(df_train, treatment='sex', outcome='income')
        assert estimate.value is not None
        assert isinstance(float(estimate.value), float)
