from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def predict_2026_matchup_matrix(model, season_feats_2026):
    """
    Returns a full symmetric probability matrix of 2026 teams.
    Each cell [i, j] = probability Team i beats Team j.
    """
    feats = season_feats_2026.set_index("TeamID")
    feature_cols = [c for c in feats.columns if c not in ["Season", "TeamID"]]
    teams = feats.index.values
    n = len(teams)
    
    # Initialize empty probability matrix
    prob_matrix = pd.DataFrame(0.5, index=teams, columns=teams, dtype=float)
    
    X_rows = []
    pairs = []
    
    for i in range(n):
        for j in range(n):
            if i == j:
                prob_matrix.iloc[i, j] = np.nan  # a team can't play itself
                continue
            t1, t2 = teams[i], teams[j]
            x = feats.loc[t1, feature_cols].values - feats.loc[t2, feature_cols].values
            X_rows.append(x)
            pairs.append((t1, t2))
    
    X = np.vstack(X_rows)
    p = model.predict_proba(X)[:, 1]
    
    # Fill matrix
    for idx, (t1, t2) in enumerate(pairs):
        prob_matrix.loc[t1, t2] = p[idx]
    
    return prob_matrix