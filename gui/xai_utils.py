# xai_utils.py
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

PERM_AVAILABLE = True  

def explain_with_permutation(model):
    """
    Computes permutation importance for the trained model.
    Returns a DataFrame with features, mean importance, and standard deviation.
    """
    try:
        if not hasattr(model, "model") or not hasattr(model.model, "feature_names_in_"):
            raise ValueError("Model is not trained properly or missing feature names.")
        
        import pandas as pd
        df = pd.read_csv("diabetes.csv")
        X = df[model.feature_names]
        y = df["Outcome"]

        result = permutation_importance(model.model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values(by='importance_mean', ascending=True)

        return perm_df

    except Exception as e:
        print(f"Permutation importance error: {e}")
        return pd.DataFrame({'feature': [], 'importance_mean': [], 'importance_std': []})

# For compatibility if PDP_AVAILABLE is used
PDP_AVAILABLE = False
def explain_with_pdp(model):
    return "Partial Dependence Plots not implemented."

