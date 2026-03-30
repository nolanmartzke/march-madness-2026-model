import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LinearRegression
import math

USE_SEED_FEATURES = True

# --------------------- SEED RULE ADJUSTMENTS (2026 ONLY) ---------------------
SEED_RULE_ADJUST_STRENGTH = 5  # increase for more aggressive adjustments
SEED_RULE_DAMPING_K = 3.0  # higher => more damping on large adjustments
SEED_RULE_THRESHOLDS = {
    "R64": {
        (1, 16): 0.975,
        (2, 15): 0.965,
        (3, 14): 0.93,
        (4, 13): 0.92,
        (5, 12): 0.90,
        (6, 11): 0.75,
    },
    "R32": {
        (1, 8): 0.72,
        (4, 5): 0.55,
        (3, 6): 0.55,
        (2, 7): 0.75,
        (1, 9): 0.83,
        (2, 10): 0.80,
        (3, 11): 0.60,
    },
    "S16": {
        (1, 5): 0.74,
        (2, 3): 0.60,
        (5, 9): 0.70,
        (4, 9): 0.70,
        (7, 11): 0.75,
        (1, 4): 0.83,
        (2, 11): 0.60,
        (2, 6): 0.75,
        (3, 10): 0.80,
    },
    "E8": {
        (1, 3): 0.55,
        (1, 2): 0.50,
        (3, 5): 0.65,
        (2, 5): 0.70,
        (3, 9): 0.65,
        (1, 10): 0.75,
        (1, 11): 0.75,
        (9, 11): 0.50,
        (3, 4): 0.55,
        (4, 11): 0.75,
    },
}
SEED_RULE_R64_BOOST = {(7, 10): 0.70, (8, 9): 0.70}


def adjust_prob_for_seed_rules(prob, seed1, seed2, round_name, season):
    if season != 2026:
        return prob
    if round_name not in {"R64", "R32", "S16", "E8"}:
        return prob
    if not np.isfinite(seed1) or not np.isfinite(seed2):
        return prob
    if seed1 == seed2:
        return prob

    higher = int(min(seed1, seed2))
    lower = int(max(seed1, seed2))
    prob_higher = prob if seed1 == higher else (1.0 - prob)

    # Special boost for 7/10 and 8/9 in R64 when higher seed is already > 70%
    if round_name == "R64" and (higher, lower) in SEED_RULE_R64_BOOST:
        thresh = SEED_RULE_R64_BOOST[(higher, lower)]
        if prob_higher > thresh:
            delta = prob_higher - thresh
            adj = SEED_RULE_ADJUST_STRENGTH * delta
            adj = adj / (1.0 + SEED_RULE_DAMPING_K * abs(adj))
            prob_higher = thresh + adj
    else:
        thresh = SEED_RULE_THRESHOLDS.get(round_name, {}).get((higher, lower))
        if thresh is not None and prob_higher < thresh:
            delta = prob_higher - thresh
            adj = SEED_RULE_ADJUST_STRENGTH * delta
            adj = adj / (1.0 + SEED_RULE_DAMPING_K * abs(adj))
            prob_higher = thresh + adj

    prob_higher = float(np.clip(prob_higher, 0.01, 0.99))
    return prob_higher if seed1 == higher else (1.0 - prob_higher)


def infer_round_from_daynum(daynum):
    if daynum <= 136:
        return "R64"
    if daynum <= 138:
        return "R32"
    if daynum <= 145:
        return "S16"
    if daynum <= 146:
        return "E8"
    if daynum <= 152:
        return "F4"
    return "F2"

def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _normal_cdf_vec(x):
    v_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + v_erf(x / np.sqrt(2.0)))


def build_submission_from_sample(
    model,
    season_feats_2026,
    feature_cols,
    weights,
    sample_path,
    out_path,
    rating_scale=11.0,
    shrink=0.9,
    blend_weight=0.7,
):
    sample = pd.read_csv(sample_path)
    ids = sample["ID"].str.split("_", expand=True)
    team1 = ids[1].astype(int)
    team2 = ids[2].astype(int)

    feats = season_feats_2026.set_index("TeamID")

    pred = np.full(len(sample), 0.5, dtype=float)

    known = team1.isin(feats.index) & team2.isin(feats.index)
    if known.any():
        f = feats[feature_cols].values
        idx1 = feats.index.get_indexer(team1[known])
        idx2 = feats.index.get_indexer(team2[known])

        x_team = f[idx1]
        x_opp = f[idx2]

        seed_diff = (feats.loc[team1[known], "seed"].values - feats.loc[team2[known], "seed"].values).reshape(-1, 1)
        if not USE_SEED_FEATURES:
            seed_diff = np.zeros_like(seed_diff)
        off_vs_def = (
            feats.loc[team1[known], "season_adjOE"].values -
            feats.loc[team2[known], "season_adjDE"].values
        ).reshape(-1, 1)
        def_vs_off = (
            feats.loc[team1[known], "season_adjDE"].values -
            feats.loc[team2[known], "season_adjOE"].values
        ).reshape(-1, 1)

        X = np.hstack([x_team - x_opp, seed_diff, off_vs_def, def_vs_off])
        X_weighted = X * weights

        probs_raw = model.predict_proba(X_weighted)[:, 1]

        probs_shrunk = 0.5 + shrink * (probs_raw - 0.5)
        probs_shrunk = np.clip(probs_shrunk, 0.01, 0.99)

        rating_diff = (
            (feats.loc[team1[known], "season_adjOE"].values - feats.loc[team1[known], "season_adjDE"].values) -
            (feats.loc[team2[known], "season_adjOE"].values - feats.loc[team2[known], "season_adjDE"].values)
        )
        probs_rating = _normal_cdf_vec(rating_diff / rating_scale)

        probs_blend = blend_weight * probs_shrunk + (1.0 - blend_weight) * probs_rating
        probs_blend = np.clip(probs_blend, 0.01, 0.99)

        pred[known] = probs_blend

    out_df = pd.DataFrame({"ID": sample["ID"], "Pred": pred})
    out_df.to_csv(out_path, index=False)
    return out_path

def evaluate_model_by_season(
    all_games,
    season_feats,
    save_csv=False,
    csv_dir="predictions",
    shrink=1.0,
    rating_scale=8.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
    xgb_params=None,
    verbose=True,
):
    years = sorted(season_feats["Season"].unique())
    scores = []

    if save_csv and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for test_year in years:
        if test_year <= 2014 or test_year == 2026:
            continue

        if verbose:
            print(f"\nTesting season {test_year}")

        # ---------------- TRAIN DATA ----------------
        train_games = all_games[all_games["Season"] < test_year].copy()
        train_feats = season_feats[season_feats["Season"] < test_year].copy()

        # Only tournament games for test
        test_games = all_games[
            (all_games["Season"] == test_year) & 
            (all_games["is_tourney"] == 1)
        ].copy()
        test_feats = season_feats[season_feats["Season"] == test_year].copy()

        if test_games.empty:
            if verbose:
                print(f"No tournament games for {test_year}, skipping.")
            continue

        # ---------------- HISTORICAL SEED EXPECTATION ----------------
        # Combine both winner-loser and loser-winner to get matchup stats in both directions
        train_tourney = train_games[train_games["is_tourney"] == 1].copy()
        team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
        train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
        train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)

        # Create seed difference in both directions
        seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
        seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]

        # Combine for historical probability mapping
        df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
        df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
        df_seed = pd.concat([df_w, df_l])

        seed_matchup_mean = df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()

        # ---------------- FEATURE IMPORTANCE ----------------
        coef_importance = feature_importance_regression(train_games, train_feats)

        # ---------------- TRAIN XGBOOST ----------------
        X_train, y_train, feature_cols, weights, game_weights = build_matchup_training_weighted(
            train_games, train_feats, coef_importance
        )
        model = train_xgboost_model(X_train, y_train, game_weights, params=xgb_params)

        # ---------------- BUILD TEST MATRIX ----------------
        feats = test_feats.set_index(["Season", "TeamID"])
        g = test_games.copy()
        feats_team = feats.add_suffix("_team")
        feats_opp = feats.add_suffix("_opp")
        g = g.merge(feats_team, left_on=["Season","TeamID"], right_index=True, how="left")
        g = g.merge(feats_opp, left_on=["Season","opp_team_id"], right_index=True, how="left")

        X_team = g[[f"{c}_team" for c in feature_cols]].values
        X_opp = g[[f"{c}_opp" for c in feature_cols]].values
        seed_diff = (g["seed_team"] - g["seed_opp"]).values.reshape(-1,1)
        if not USE_SEED_FEATURES:
            seed_diff = np.zeros_like(seed_diff)
        off_vs_def = (g["season_adjOE_team"] - g["season_adjDE_opp"]).values.reshape(-1,1)
        def_vs_off = (g["season_adjDE_team"] - g["season_adjOE_opp"]).values.reshape(-1,1)

        X_test = np.hstack([X_team - X_opp, seed_diff, off_vs_def, def_vs_off])
        base_weights = np.abs(coef_importance.reindex([f"{c}_diff" for c in feature_cols])).fillna(1.0).values
        weights_array = np.concatenate([base_weights, np.ones(3)])
        X_test_weighted = X_test * weights_array

        y_test = g["Win"].values

        # ---------------- PREDICT WIN PROBABILITY ----------------
        probs_raw = model.predict_proba(X_test_weighted)[:, 1]

        # ---------------- NO SEED TUNING ----------------
        probs_tuned = probs_raw.copy()

        # ---------------- RATING-ONLY MODEL ----------------
        # Use Torvik-style net rating (season_adjOE - season_adjDE) as a pure rating baseline.
        net_team = g["season_adjOE_team"] - g["season_adjDE_team"]
        net_opp = g["season_adjOE_opp"] - g["season_adjDE_opp"]
        rating_diff = (net_team - net_opp).values
        probs_rating = np.array([_normal_cdf(d / rating_scale) for d in rating_diff])

        # ---------------- GLOBAL SHRINK (for Brier) ----------------
        probs_shrunk = 0.5 + shrink * (probs_tuned - 0.5)
        probs_shrunk = np.clip(probs_shrunk, 0.01, 0.99)

        # ---------------- BLEND MODEL + RATING ----------------
        probs_blend = blend_weight * probs_shrunk + (1.0 - blend_weight) * probs_rating
        probs_blend = np.clip(probs_blend, 0.01, 0.99)

        # ---------------- PREDICT MARGIN ----------------
        # Prepare margin training
        margin_train = train_games.copy()
        margin_train["Margin"] = margin_train["WScore"] - margin_train["LScore"]
        margin_train = margin_train[np.isfinite(margin_train["Margin"])]
        feats_train = train_feats.set_index(["Season","TeamID"])
        margin_train = margin_train.merge(feats_train.add_suffix("_team"), left_on=["Season","WTeamID"], right_index=True)
        margin_train = margin_train.merge(feats_train.add_suffix("_opp"), left_on=["Season","LTeamID"], right_index=True)

        X_margin_train = np.hstack([
            margin_train[[f"{c}_team" for c in feature_cols]].values - margin_train[[f"{c}_opp" for c in feature_cols]].values,
            np.zeros((margin_train.shape[0], 1)),
            (margin_train["season_adjOE_team"].values - margin_train["season_adjDE_opp"].values).reshape(-1,1),
            (margin_train["season_adjDE_team"].values - margin_train["season_adjOE_opp"].values).reshape(-1,1)
        ])
        y_margin_train = margin_train["Margin"].values
        margin_model = LinearRegression()
        margin_model.fit(X_margin_train, y_margin_train)

        predicted_margin = margin_model.predict(X_test)

        # ---------------- FILTER: LOWER TEAMID (KAGGLE FORMAT, DEDUP) ----------------
        # Keep one row per game and align "Actual" with lower TeamID winning.
        mask = g["TeamID"] < g["opp_team_id"]
        pred_df = pd.DataFrame({
            "Season": g["Season"].values[mask],
            "TeamID": g["TeamID"].values[mask],
            "OppID": g["opp_team_id"].values[mask],
            "Team": g["team"].values[mask],
            "Opp": g["opp_team"].values[mask],
            "Actual": y_test[mask],
            "Pred_Prob_Raw": probs_raw[mask],
            "Pred_Prob_Tuned": probs_tuned[mask],
            "Pred_Prob_Shrunk": probs_shrunk[mask],
            "Pred_Prob_Rating": probs_rating[mask],
            "Pred_Prob_Blend": probs_blend[mask],
            "Predicted_Margin": predicted_margin[mask],
            "Actual_Margin": (g["WScore"].values - g["LScore"].values)[mask]
        })

        # ---------------- SEED INFO ----------------
        team_seed = dict(zip(test_feats["TeamID"], test_feats["seed"]))
        pred_df["Seed_Team"] = pred_df["TeamID"].map(team_seed)
        pred_df["Seed_Opp"] = pred_df["OppID"].map(team_seed)
        pred_df["Seed_Diff"] = pred_df["Seed_Team"] - pred_df["Seed_Opp"]

        # ---------------- UPSets ----------------
        pred_df['Upset'] = np.where(
            ((pred_df['Seed_Diff'] < 0) & (pred_df['Actual'] == 0)) |
            ((pred_df['Seed_Diff'] > 0) & (pred_df['Actual'] == 1)), 1, 0
        )
        pred_df['Chalk'] = 1 - pred_df['Upset']

        # ---------------- LOGLOSS ----------------
        logloss = log_loss(pred_df["Actual"], pred_df["Pred_Prob_Tuned"])
        brier_shrunk = brier_score_loss(pred_df["Actual"], pred_df["Pred_Prob_Shrunk"])
        brier_rating = brier_score_loss(pred_df["Actual"], pred_df["Pred_Prob_Rating"])
        brier_blend = brier_score_loss(pred_df["Actual"], pred_df["Pred_Prob_Blend"])
        if verbose:
            print(
                f"LogLoss {test_year}: {logloss:.4f} | "
                f"Brier Shrunk {test_year}: {brier_shrunk:.4f} | "
                f"Brier Rating {test_year}: {brier_rating:.4f} | "
                f"Brier Blend {test_year}: {brier_blend:.4f}"
            )
        scores.append(brier_blend)

        # Optional: save CSV
        if save_csv:
            pred_df.to_csv(os.path.join(csv_dir, f"predictions_{test_year}.csv"), index=False)

        # Optional: print top predictions
        if verbose:
            pred_df["Brier_Contribution"] = (pred_df["Pred_Prob_Blend"] - pred_df["Actual"]) ** 2
            pred_df["Predicted_Margin_Signed"] = np.where(
                pred_df["Pred_Prob_Blend"] >= 0.5,
                np.abs(pred_df["Predicted_Margin"]),
                -np.abs(pred_df["Predicted_Margin"]),
            )
            pred_df["Actual_Margin_Signed"] = np.where(
                pred_df["Actual"] == 1,
                np.abs(pred_df["Actual_Margin"]),
                -np.abs(pred_df["Actual_Margin"]),
            )
            print(
                pred_df.sort_values("Brier_Contribution", ascending=False)
                .head(20)
                .drop(columns=["Brier_Contribution", "Predicted_Margin", "Actual_Margin"])
            )

    if verbose:
        print("\nAverage Brier (Blend):", np.mean(scores))
    return float(np.mean(scores)) if scores else float("nan")


def brier_report(
    all_games,
    season_feats,
    seasons,
    shrink=1.0,
    rating_scale=8.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
    xgb_params=None,
    apply_seed_rules=True,
    apply_calibration=True,
    adjust_strength=None,
    force_season_for_rules=None,
):
    global SEED_RULE_ADJUST_STRENGTH
    prev_strength = SEED_RULE_ADJUST_STRENGTH
    if adjust_strength is not None:
        SEED_RULE_ADJUST_STRENGTH = adjust_strength
    rows = []
    for test_year in seasons:
        train_games = all_games[all_games["Season"] < test_year].copy()
        train_feats = season_feats[season_feats["Season"] < test_year].copy()
        test_games = all_games[
            (all_games["Season"] == test_year) &
            (all_games["is_tourney"] == 1)
        ].copy()
        test_feats = season_feats[season_feats["Season"] == test_year].copy()

        if test_games.empty:
            continue

        coef_importance = feature_importance_regression(train_games, train_feats)
        X_train, y_train, feature_cols, weights, game_weights = build_matchup_training_weighted(
            train_games, train_feats, coef_importance
        )
        model = train_xgboost_model(X_train, y_train, game_weights, params=xgb_params)

        # Historical seed matchup mean for tuning
        train_tourney = train_games[train_games["is_tourney"] == 1].copy()
        team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
        train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
        train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)
        seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
        seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]
        df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
        df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
        df_seed = pd.concat([df_w, df_l])
        seed_matchup_mean = df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()

        def compute_probs(games_df, feats_df):
            feats = feats_df.set_index(["Season", "TeamID"])
            g = games_df.copy()
            feats_team = feats.add_suffix("_team")
            feats_opp = feats.add_suffix("_opp")
            g = g.merge(feats_team, left_on=["Season", "TeamID"], right_index=True, how="left")
            g = g.merge(feats_opp, left_on=["Season", "opp_team_id"], right_index=True, how="left")

            X_team = g[[f"{c}_team" for c in feature_cols]].values
            X_opp = g[[f"{c}_opp" for c in feature_cols]].values
            seed_diff = (g["seed_team"] - g["seed_opp"]).values.reshape(-1, 1)
            if not USE_SEED_FEATURES:
                seed_diff = np.zeros_like(seed_diff)
            off_vs_def = (g["season_adjOE_team"] - g["season_adjDE_opp"]).values.reshape(-1, 1)
            def_vs_off = (g["season_adjDE_team"] - g["season_adjOE_opp"]).values.reshape(-1, 1)

            X_test = np.hstack([X_team - X_opp, seed_diff, off_vs_def, def_vs_off])
            base_weights = np.abs(coef_importance.reindex([f"{c}_diff" for c in feature_cols])).fillna(1.0).values
            weights_array = np.concatenate([base_weights, np.ones(3)])
            X_test_weighted = X_test * weights_array

            probs_raw = model.predict_proba(X_test_weighted)[:, 1]

            # Seed tuning (same as evaluate_model_by_season)
            probs_tuned = probs_raw.copy()

            net_team = g["season_adjOE_team"] - g["season_adjDE_team"]
            net_opp = g["season_adjOE_opp"] - g["season_adjDE_opp"]
            rating_diff = (net_team - net_opp).values
            probs_rating = np.array([_normal_cdf(d / rating_scale) for d in rating_diff])

            probs_shrunk = 0.5 + shrink * (probs_tuned - 0.5)
            probs_shrunk = np.clip(probs_shrunk, 0.01, 0.99)
            probs_blend = blend_weight * probs_shrunk + (1.0 - blend_weight) * probs_rating
            probs_blend = np.clip(probs_blend, 0.01, 0.99)

            if apply_seed_rules:
                seeds_a = g["seed_team"].values
                seeds_b = g["seed_opp"].values
                rounds = g["DayNum"].apply(infer_round_from_daynum).values
                probs_adj = []
                for p, s1, s2, rnd in zip(probs_blend, seeds_a, seeds_b, rounds):
                    season_val = force_season_for_rules if force_season_for_rules is not None else test_year
                    probs_adj.append(adjust_prob_for_seed_rules(p, s1, s2, rnd, season_val))
                probs_blend = np.clip(np.array(probs_adj), 0.01, 0.99)

            return probs_blend, g["Win"].values

        train_probs, train_y = compute_probs(train_tourney, train_feats)
        test_probs, test_y = compute_probs(test_games, test_feats)

        brier_base = brier_score_loss(test_y, test_probs)

        if apply_calibration:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(train_probs, train_y)
            test_probs_cal = iso.predict(test_probs)
            brier_cal = brier_score_loss(test_y, test_probs_cal)
        else:
            brier_cal = np.nan

        rows.append({
            "Season": test_year,
            "Brier": brier_base,
            "Brier_Calibrated": brier_cal,
        })

    df = pd.DataFrame(rows)
    if adjust_strength is not None:
        SEED_RULE_ADJUST_STRENGTH = prev_strength
    if df.empty:
        return df
    avg_row = {
        "Season": "Average",
        "Brier": df["Brier"].mean(),
        "Brier_Calibrated": df["Brier_Calibrated"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df


def precompute_eval_cache(all_games, season_feats):
    years = sorted(season_feats["Season"].unique())
    cache = []

    for test_year in years:
        if test_year <= 2014 or test_year == 2026:
            continue

        # ---------------- TRAIN DATA ----------------
        train_games = all_games[all_games["Season"] < test_year].copy()
        train_feats = season_feats[season_feats["Season"] < test_year].copy()

        # Only tournament games for test
        test_games = all_games[
            (all_games["Season"] == test_year) &
            (all_games["is_tourney"] == 1)
        ].copy()
        test_feats = season_feats[season_feats["Season"] == test_year].copy()

        if test_games.empty:
            continue

        # ---------------- HISTORICAL SEED EXPECTATION ----------------
        train_tourney = train_games[train_games["is_tourney"] == 1].copy()
        team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
        train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
        train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)

        seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
        seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]

        df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
        df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
        df_seed = pd.concat([df_w, df_l])

        seed_matchup_mean = df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()

        # ---------------- FEATURE IMPORTANCE ----------------
        coef_importance = feature_importance_regression(train_games, train_feats)

        # ---------------- TRAIN XGBOOST ----------------
        X_train, y_train, feature_cols, weights, game_weights = build_matchup_training_weighted(
            train_games, train_feats, coef_importance
        )
        model = train_xgboost_model(X_train, y_train, game_weights)

        # ---------------- BUILD TEST MATRIX ----------------
        feats = test_feats.set_index(["Season", "TeamID"])
        g = test_games.copy()
        feats_team = feats.add_suffix("_team")
        feats_opp = feats.add_suffix("_opp")
        g = g.merge(feats_team, left_on=["Season", "TeamID"], right_index=True, how="left")
        g = g.merge(feats_opp, left_on=["Season", "opp_team_id"], right_index=True, how="left")

        X_team = g[[f"{c}_team" for c in feature_cols]].values
        X_opp = g[[f"{c}_opp" for c in feature_cols]].values
        seed_diff = (g["seed_team"] - g["seed_opp"]).values.reshape(-1, 1)
        off_vs_def = (g["season_adjOE_team"] - g["season_adjDE_opp"]).values.reshape(-1, 1)
        def_vs_off = (g["season_adjDE_team"] - g["season_adjOE_opp"]).values.reshape(-1, 1)

        X_test = np.hstack([X_team - X_opp, seed_diff, off_vs_def, def_vs_off])
        base_weights = np.abs(coef_importance.reindex([f"{c}_diff" for c in feature_cols])).fillna(1.0).values
        weights_array = np.concatenate([base_weights, np.ones(3)])
        X_test_weighted = X_test * weights_array

        y_test = g["Win"].values

        # ---------------- PREDICT WIN PROBABILITY ----------------
        probs_raw = model.predict_proba(X_test_weighted)[:, 1]

        # ---------------- SEED-DIFF STATS FOR TUNING ----------------
        seed_diff_flat = seed_diff.flatten()
        seed_diff_stats = (
            pd.DataFrame({"Seed_Diff": seed_diff_flat, "Prob": probs_raw})
            .groupby("Seed_Diff")["Prob"]
            .agg(["mean", "std"])
            .to_dict("index")
        )

        mean_raw_arr = np.array([
            seed_diff_stats.get(sd, {}).get("mean", 0.5) for sd in seed_diff_flat
        ])
        std_raw_arr = np.array([
            seed_diff_stats.get(sd, {}).get("std", 0.0) for sd in seed_diff_flat
        ])
        std_raw_arr = np.where(np.isnan(std_raw_arr), 0.0, std_raw_arr)

        hist_mean_arr = np.array([
            seed_matchup_mean.get(sd, 0.5) for sd in seed_diff_flat
        ])
        hist_std_arr = np.sqrt(hist_mean_arr * (1 - hist_mean_arr))

        # ---------------- FILTER: LOWER TEAMID (KAGGLE FORMAT, DEDUP) ----------------
        mask = g["TeamID"] < g["opp_team_id"]
        rating_diff = (
            (g["season_adjOE_team"] - g["season_adjDE_team"]) -
            (g["season_adjOE_opp"] - g["season_adjDE_opp"])
        ).values

        cache.append({
            "season": test_year,
            "y": y_test[mask],
            "probs_raw": probs_raw[mask],
            "rating_diff": rating_diff[mask],
            "mean_raw": mean_raw_arr[mask],
            "std_raw": std_raw_arr[mask],
            "hist_mean": hist_mean_arr[mask],
            "hist_std": hist_std_arr[mask],
        })

    return cache


def grid_search_brier(all_games, season_feats):
    cache = precompute_eval_cache(all_games, season_feats)
    if not cache:
        print("No cached seasons available for grid search.")
        return None, []

    shrink_vals = [0.85, 0.9, 0.95, 1.0]
    rating_scales = [8.0, 10.0, 11.0, 12.0, 14.0]
    blend_weights = [0.4, 0.55, 0.7, 0.85]
    alpha_vals = [0.4, 0.7, 1.0]
    z_thresholds = [0.0, 0.5, 1.0]

    best = None
    results = []
    for alpha in alpha_vals:
        for z_threshold in z_thresholds:
            for shrink in shrink_vals:
                for rating_scale in rating_scales:
                    for blend_weight in blend_weights:
                        season_scores = []
                        for entry in cache:
                            std_raw = entry["std_raw"]
                            z = np.zeros_like(entry["probs_raw"])
                            valid = std_raw >= 1e-6
                            z[valid] = (entry["probs_raw"][valid] - entry["mean_raw"][valid]) / std_raw[valid]

                            probs_tuned = entry["probs_raw"].copy()
                            use = np.abs(z) >= z_threshold
                            if np.any(use):
                                probs_tuned[use] = entry["hist_mean"][use] + alpha * z[use] * entry["hist_std"][use]
                            probs_tuned = np.clip(probs_tuned, 0.01, 0.99)

                            probs_shrunk = 0.5 + shrink * (probs_tuned - 0.5)
                            probs_shrunk = np.clip(probs_shrunk, 0.01, 0.99)

                            probs_rating = _normal_cdf_vec(entry["rating_diff"] / rating_scale)
                            probs_blend = blend_weight * probs_shrunk + (1.0 - blend_weight) * probs_rating
                            probs_blend = np.clip(probs_blend, 0.01, 0.99)

                            brier = brier_score_loss(entry["y"], probs_blend)
                            season_scores.append(brier)

                        brier = float(np.mean(season_scores)) if season_scores else float("nan")
                        results.append((brier, shrink, rating_scale, blend_weight, alpha, z_threshold))
                        if best is None or brier < best[0]:
                            best = (brier, shrink, rating_scale, blend_weight, alpha, z_threshold)

    results.sort(key=lambda r: r[0])
    print("\nTop 10 parameter sets by Brier:")
    for brier, shrink, rating_scale, blend_weight, alpha, z_threshold in results[:10]:
        print(
            f"Brier={brier:.4f} | shrink={shrink} | rating_scale={rating_scale} | "
            f"blend_weight={blend_weight} | alpha={alpha} | z_threshold={z_threshold}"
        )
    print(
        f"\nBest: Brier={best[0]:.4f} | shrink={best[1]} | rating_scale={best[2]} | "
        f"blend_weight={best[3]} | alpha={best[4]} | z_threshold={best[5]}"
    )
    return best, results


def grid_search_xgb_hyperparams(all_games, season_feats):
    # Fixed best parameters from prior grid
    shrink = 1.0
    rating_scale = 8.0
    blend_weight = 0.55
    seed_alpha = 0.7
    seed_z_threshold = 1.0

    # Limit seasons for speed (recent seasons tend to be most relevant)
    seasons = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

    param_grid = [
        # Local grid around current best: n_estimators=600, max_depth=5, lr=0.03, subsample=0.8, colsample=0.8
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 700, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.025, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.035, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9},
    ]

    results = []
    best = None
    total = len(param_grid)
    for i, params in enumerate(param_grid, 1):
        scores = []
        print(f"\nHyperparam set {i}/{total}: {params}")
        for test_year in seasons:
            train_games = all_games[all_games["Season"] < test_year].copy()
            train_feats = season_feats[season_feats["Season"] < test_year].copy()

            test_games = all_games[
                (all_games["Season"] == test_year) &
                (all_games["is_tourney"] == 1)
            ].copy()
            test_feats = season_feats[season_feats["Season"] == test_year].copy()
            if test_games.empty:
                continue

            # Seed expectations (training)
            train_tourney = train_games[train_games["is_tourney"] == 1].copy()
            team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
            train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
            train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)
            seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
            seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]
            df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
            df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
            df_seed = pd.concat([df_w, df_l])
            seed_matchup_mean = df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()

            # Feature importance + training
            coef_importance = feature_importance_regression(train_games, train_feats)
            X_train, y_train, feature_cols, weights, game_weights = build_matchup_training_weighted(
                train_games, train_feats, coef_importance
            )
            model = train_xgboost_model(X_train, y_train, game_weights, params=params)

            # Test matrix
            feats = test_feats.set_index(["Season", "TeamID"])
            g = test_games.copy()
            feats_team = feats.add_suffix("_team")
            feats_opp = feats.add_suffix("_opp")
            g = g.merge(feats_team, left_on=["Season", "TeamID"], right_index=True, how="left")
            g = g.merge(feats_opp, left_on=["Season", "opp_team_id"], right_index=True, how="left")

            X_team = g[[f"{c}_team" for c in feature_cols]].values
            X_opp = g[[f"{c}_opp" for c in feature_cols]].values
            seed_diff = (g["seed_team"] - g["seed_opp"]).values.reshape(-1, 1)
            off_vs_def = (g["season_adjOE_team"] - g["season_adjDE_opp"]).values.reshape(-1, 1)
            def_vs_off = (g["season_adjDE_team"] - g["season_adjOE_opp"]).values.reshape(-1, 1)

            X_test = np.hstack([X_team - X_opp, seed_diff, off_vs_def, def_vs_off])
            base_weights = np.abs(coef_importance.reindex([f"{c}_diff" for c in feature_cols])).fillna(1.0).values
            weights_array = np.concatenate([base_weights, np.ones(3)])
            X_test_weighted = X_test * weights_array

            probs_raw = model.predict_proba(X_test_weighted)[:, 1]

            # Seed tuning
            seed_diff_flat = seed_diff.flatten()
            seed_diff_stats = (
                pd.DataFrame({"Seed_Diff": seed_diff_flat, "Prob": probs_raw})
                .groupby("Seed_Diff")["Prob"]
                .agg(["mean", "std"])
                .to_dict("index")
            )

            def tune_prob_seed_relative(prob, seed_diff_val):
                stats = seed_diff_stats.get(seed_diff_val, None)
                mean_raw = stats["mean"] if stats else 0.5
                std_raw = stats["std"] if stats and not np.isnan(stats["std"]) else 0.0

                if std_raw < 1e-6:
                    z = 0.0
                else:
                    z = (prob - mean_raw) / std_raw

                if abs(z) < seed_z_threshold:
                    return float(np.clip(prob, 0.01, 0.99))

                hist_mean = seed_matchup_mean.get(seed_diff_val, 0.5)
                hist_std = np.sqrt(hist_mean * (1 - hist_mean))

                tuned = hist_mean + seed_alpha * z * hist_std
                return float(np.clip(tuned, 0.01, 0.99))

            probs_tuned = np.array([tune_prob_seed_relative(p, sd) for p, sd in zip(probs_raw, seed_diff_flat)])
            probs_shrunk = 0.5 + shrink * (probs_tuned - 0.5)
            probs_shrunk = np.clip(probs_shrunk, 0.01, 0.99)

            rating_diff = (
                (g["season_adjOE_team"] - g["season_adjDE_team"]) -
                (g["season_adjOE_opp"] - g["season_adjDE_opp"])
            ).values
            probs_rating = _normal_cdf_vec(rating_diff / rating_scale)
            probs_blend = blend_weight * probs_shrunk + (1.0 - blend_weight) * probs_rating
            probs_blend = np.clip(probs_blend, 0.01, 0.99)

            mask = g["TeamID"] < g["opp_team_id"]
            y = g["Win"].values[mask]
            scores.append(brier_score_loss(y, probs_blend[mask]))

        avg_brier = float(np.mean(scores)) if scores else float("nan")
        results.append((avg_brier, params))
        if best is None or avg_brier < best[0]:
            best = (avg_brier, params)
        print(f"Avg Brier: {avg_brier:.4f}")

    results.sort(key=lambda r: r[0])
    print("\nTop 5 XGB hyperparams by Brier:")
    for brier, params in results[:5]:
        print(f"Brier={brier:.4f} | params={params}")
    print(f"\nBest: Brier={best[0]:.4f} | params={best[1]}")
    return best, results




# --------------------- PATHS ---------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
KAGGLE_DATA_DIR = os.path.join(BASE_DIR, "data", "march-machine-learning-mania-2026")
TEAM_RESULTS_GLOB = os.path.join(BASE_DIR, "data", "*_team_results.csv")
TORVIK_MAP_PATH = os.path.join(BASE_DIR, "analysis", "teamid_to_torvik_mapping.csv")
TOURNEY_RESULTS_PATH = os.path.join(KAGGLE_DATA_DIR, "MNCAATourneyDetailedResults.csv")
COMPACT_RESULTS_PATH = os.path.join(KAGGLE_DATA_DIR, "MNCAATourneyCompactResults.csv")
SEEDS_PATH = os.path.join(KAGGLE_DATA_DIR, "MNCAATourneySeeds.csv")

# --------------------- UTILS ---------------------
def normalize_team_name(name):
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_team_name_lookup():
    teams_path = os.path.join(KAGGLE_DATA_DIR, "MTeams.csv")
    teams = pd.read_csv(teams_path)
    teams["team_norm"] = teams["TeamName"].map(normalize_team_name)
    name_to_id = dict(zip(teams["team_norm"], teams["TeamID"]))
    id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))
    return name_to_id, id_to_name


def print_team_feature_row(season_feats, season, team_name):
    name_to_id, id_to_name = build_team_name_lookup()
    team_id = name_to_id.get(normalize_team_name(team_name))
    if team_id is None:
        print(f"Team not found: {team_name}")
        return
    row = season_feats[(season_feats["Season"] == season) & (season_feats["TeamID"] == team_id)]
    if row.empty:
        print(f"No features for {team_name} in {season}")
        return
    row = row.iloc[0].drop(labels=["Season", "TeamID"])
    print(f"\n=== Features for {id_to_name.get(team_id, team_name)} ({season}) ===")
    print(row.to_string())

def parse_seed(seed_str):
    region = seed_str[0]
    seed_num = int(seed_str[1:3])
    suffix = seed_str[3:] if len(seed_str) > 3 else ""
    return region, seed_num, suffix


def build_seed_bracket(seeds_df, season):
    s = seeds_df[seeds_df["Season"] == season].copy()
    s[["region", "seed_num", "suffix"]] = s["Seed"].apply(
        lambda x: pd.Series(parse_seed(x))
    )
    bracket = {}
    for _, row in s.iterrows():
        region = row["region"]
        seed_num = int(row["seed_num"])
        team_id = int(row["TeamID"])
        bracket.setdefault(region, {}).setdefault(seed_num, []).append(team_id)
    return bracket


def build_seed_matchup_mean(all_games, season_feats, season_max):
    train_games = all_games[all_games["Season"] < season_max].copy()
    train_feats = season_feats[season_feats["Season"] < season_max].copy()
    train_tourney = train_games[train_games["is_tourney"] == 1].copy()
    team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
    train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
    train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)
    seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
    seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]
    df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
    df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
    df_seed = pd.concat([df_w, df_l])
    return df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()


def build_pairwise_prob_cache(
    team_ids,
    feats_2026,
    feature_cols,
    weights,
    model,
):
    feats = feats_2026.set_index("TeamID")
    prob_raw_cache = {}
    seed_diff_list = []
    prob_list = []
    for i in team_ids:
        for j in team_ids:
            if i == j:
                continue
            row_i = feats.loc[i, feature_cols]
            row_j = feats.loc[j, feature_cols]
            if isinstance(row_i, pd.DataFrame):
                row_i = row_i.iloc[0]
            if isinstance(row_j, pd.DataFrame):
                row_j = row_j.iloc[0]
            x_team = row_i.values
            x_opp = row_j.values

            seed_i = feats.loc[i, "seed"]
            seed_j = feats.loc[j, "seed"]
            if isinstance(seed_i, pd.Series):
                seed_i = seed_i.iloc[0]
            if isinstance(seed_j, pd.Series):
                seed_j = seed_j.iloc[0]
            seed_diff = float(seed_i - seed_j)
            if not USE_SEED_FEATURES:
                seed_diff = 0.0

            oe_i = feats.loc[i, "season_adjOE"]
            de_i = feats.loc[i, "season_adjDE"]
            oe_j = feats.loc[j, "season_adjOE"]
            de_j = feats.loc[j, "season_adjDE"]
            if isinstance(oe_i, pd.Series):
                oe_i = oe_i.iloc[0]
            if isinstance(de_i, pd.Series):
                de_i = de_i.iloc[0]
            if isinstance(oe_j, pd.Series):
                oe_j = oe_j.iloc[0]
            if isinstance(de_j, pd.Series):
                de_j = de_j.iloc[0]
            off_vs_def = float(oe_i - de_j)
            def_vs_off = float(de_i - oe_j)
            X = np.append(x_team - x_opp, [seed_diff, off_vs_def, def_vs_off]).reshape(1, -1)
            if X.shape[1] != len(weights):
                min_len = min(X.shape[1], len(weights))
                X = X[:, :min_len]
                weights_use = weights[:min_len]
            else:
                weights_use = weights
            X_weighted = X * weights_use
            prob_raw = float(model.predict_proba(X_weighted)[:, 1][0])
            prob_raw_cache[(i, j)] = prob_raw
            seed_diff_list.append(float(seed_diff))
            prob_list.append(float(prob_raw))
    df = pd.DataFrame({"Seed_Diff": seed_diff_list, "Prob": prob_list})
    df["Seed_Diff"] = pd.to_numeric(df["Seed_Diff"], errors="coerce")
    df["Prob"] = pd.to_numeric(df["Prob"], errors="coerce")
    df = df.dropna(subset=["Seed_Diff", "Prob"])
    seed_diff_stats = df.groupby("Seed_Diff")["Prob"].agg(["mean", "std"]).to_dict("index")
    return prob_raw_cache, seed_diff_stats


def simulate_tournament(
    all_games,
    season_feats,
    model,
    feature_cols,
    weights,
    season=2026,
    n_sims=1000,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
):
    seeds_df = pd.read_csv(SEEDS_PATH)
    bracket = build_seed_bracket(seeds_df, season)

    feats_2026 = season_feats[season_feats["Season"] == season].copy()
    feats_2026 = feats_2026.drop_duplicates(subset=["TeamID"])
    feature_cols_sim = feature_cols[: (len(weights) - 3)]
    team_ids = sorted({tid for region in bracket.values() for seeds in region.values() for tid in seeds})
    prob_raw_cache, seed_diff_stats = build_pairwise_prob_cache(
        team_ids, feats_2026, feature_cols_sim, weights, model
    )
    seed_matchup_mean = build_seed_matchup_mean(all_games, season_feats, season)

    feats = feats_2026.set_index("TeamID")
    net_rating = (feats["season_adjOE"] - feats["season_adjDE"]).to_dict()

    def final_prob(t1, t2, round_name=None):
        prob_raw = prob_raw_cache[(t1, t2)]
        if not USE_SEED_FEATURES:
            prob_tuned = prob_raw
        else:
            seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
            stats = seed_diff_stats.get(seed_diff, None)
            mean_raw = stats["mean"] if stats else 0.5
            std_raw = stats["std"] if stats and not np.isnan(stats["std"]) else 0.0
            if std_raw < 1e-6:
                z = 0.0
            else:
                z = (prob_raw - mean_raw) / std_raw
            if abs(z) < seed_z_threshold:
                prob_tuned = prob_raw
            else:
                hist_mean = seed_matchup_mean.get(seed_diff, 0.5)
                hist_std = np.sqrt(hist_mean * (1 - hist_mean))
                prob_tuned = hist_mean + seed_alpha * z * hist_std
        prob_tuned = float(np.clip(prob_tuned, 0.01, 0.99))
        prob_shrunk = 0.5 + shrink * (prob_tuned - 0.5)
        prob_shrunk = float(np.clip(prob_shrunk, 0.01, 0.99))
        prob_rating = _normal_cdf((net_rating[t1] - net_rating[t2]) / rating_scale)
        prob_final = blend_weight * prob_shrunk + (1.0 - blend_weight) * prob_rating
        prob_final = float(np.clip(prob_final, 0.01, 0.99))
        if round_name is not None:
            seed1 = feats.loc[t1, "seed"]
            seed2 = feats.loc[t2, "seed"]
            if isinstance(seed1, pd.Series):
                seed1 = seed1.iloc[0]
            if isinstance(seed2, pd.Series):
                seed2 = seed2.iloc[0]
            prob_final = adjust_prob_for_seed_rules(prob_final, seed1, seed2, round_name, season)
        return prob_final

    rng = np.random.default_rng()
    round_names = ["R32", "S16", "E8", "F4", "F2", "Champ"]
    counts = {tid: {r: 0 for r in round_names} for tid in team_ids}

    def add_count(team_id, round_name):
        counts[team_id][round_name] += 1

    def resolve_play_ins(region_seeds):
        resolved = {}
        for seed_num, teams in region_seeds.items():
            if len(teams) == 1:
                resolved[seed_num] = teams[0]
            else:
                t1, t2 = teams[0], teams[1]
                p = final_prob(t1, t2)
                winner = t1 if rng.random() < p else t2
                resolved[seed_num] = winner
        return resolved

    def play_round(matchups, round_name):
        winners = []
        for t1, t2 in matchups:
            if t1 == t2:
                winners.append(t1)
                continue
            p = final_prob(t1, t2, round_name=round_name)
            winner = t1 if rng.random() < p else t2
            winners.append(winner)
        return winners

    region_order = ["W", "X", "Y", "Z"]
    round1_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    for _ in range(n_sims):
        region_winners = {}
        for region in region_order:
            region_seeds = resolve_play_ins(bracket.get(region, {}))
            r64 = [(region_seeds[a], region_seeds[b]) for a, b in round1_pairs]
            r32_winners = play_round(r64, "R64")
            for t in r32_winners:
                add_count(t, "R32")

            s16_matchups = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 8, 2)]
            s16_winners = play_round(s16_matchups, "R32")
            for t in s16_winners:
                add_count(t, "S16")

            e8_matchups = [(s16_winners[0], s16_winners[1]), (s16_winners[2], s16_winners[3])]
            e8_winners = play_round(e8_matchups, "S16")
            for t in e8_winners:
                add_count(t, "E8")

            f4_matchups = [(e8_winners[0], e8_winners[1])]
            f4_winner = play_round(f4_matchups, "E8")[0]
            add_count(f4_winner, "F4")
            region_winners[region] = f4_winner

        # Final Four
        f4_winner_left = play_round([(region_winners["W"], region_winners["X"])], "F4")[0]
        add_count(f4_winner_left, "F2")
        f4_winner_right = play_round([(region_winners["Y"], region_winners["Z"])], "F4")[0]
        add_count(f4_winner_right, "F2")

        # Championship
        champ = play_round([(f4_winner_left, f4_winner_right)], "F2")[0]
        add_count(champ, "Champ")

    id_to_name = build_team_name_lookup()[1]
    rows = []
    for team_id, d in counts.items():
        rows.append({
            "TeamID": team_id,
            "Team": id_to_name.get(team_id, str(team_id)),
            **{r: d[r] / n_sims for r in round_names},
        })
    return pd.DataFrame(rows).sort_values("Champ", ascending=False)


def run_interactive_bracket(
    all_games,
    season_feats,
    model,
    feature_cols,
    weights,
    season=2026,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
):
    seeds_df = pd.read_csv(SEEDS_PATH)
    bracket = build_seed_bracket(seeds_df, season)

    feats_season = season_feats[season_feats["Season"] == season].copy()
    feats_season = feats_season.drop_duplicates(subset=["TeamID"])
    feature_cols_sim = feature_cols[: (len(weights) - 3)]
    team_ids = sorted({tid for region in bracket.values() for seeds in region.values() for tid in seeds})
    prob_raw_cache, seed_diff_stats = build_pairwise_prob_cache(
        team_ids, feats_season, feature_cols_sim, weights, model
    )
    seed_matchup_mean = build_seed_matchup_mean(all_games, season_feats, season)

    feats = feats_season.set_index("TeamID")
    seed_lookup = feats["seed"].to_dict()
    net_rating = (feats["season_adjOE"] - feats["season_adjDE"]).to_dict()
    id_to_name = build_team_name_lookup()[1]

    def final_prob(t1, t2):
        prob_raw = prob_raw_cache[(t1, t2)]
        if not USE_SEED_FEATURES:
            prob_tuned = prob_raw
        else:
            seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
            stats = seed_diff_stats.get(seed_diff, None)
            mean_raw = stats["mean"] if stats else 0.5
            std_raw = stats["std"] if stats and not np.isnan(stats["std"]) else 0.0
            if std_raw < 1e-6:
                z = 0.0
            else:
                z = (prob_raw - mean_raw) / std_raw
            if abs(z) < seed_z_threshold:
                prob_tuned = prob_raw
            else:
                hist_mean = seed_matchup_mean.get(seed_diff, 0.5)
                hist_std = np.sqrt(hist_mean * (1 - hist_mean))
                prob_tuned = hist_mean + seed_alpha * z * hist_std
        prob_tuned = float(np.clip(prob_tuned, 0.01, 0.99))
        prob_shrunk = 0.5 + shrink * (prob_tuned - 0.5)
        prob_shrunk = float(np.clip(prob_shrunk, 0.01, 0.99))
        prob_rating = _normal_cdf((net_rating[t1] - net_rating[t2]) / rating_scale)
        prob_final = blend_weight * prob_shrunk + (1.0 - blend_weight) * prob_rating
        return float(np.clip(prob_final, 0.01, 0.99))

    def playin_winner(teams):
        if len(teams) == 1:
            return teams[0]
        t1, t2 = teams[0], teams[1]
        p = final_prob(t1, t2)
        name1 = id_to_name.get(t1, str(t1))
        name2 = id_to_name.get(t2, str(t2))
        print(f"Play-in: A) {name1} vs B) {name2} | P(A wins)={p:.3f}")
        choice = input("Choose A or B (default A): ").strip().lower()
        if choice == "b":
            return t2
        return t1

    def pick_winner(t1, t2, round_name):
        p = final_prob(t1, t2)
        name1 = id_to_name.get(t1, str(t1))
        name2 = id_to_name.get(t2, str(t2))
        seed1 = feats.loc[t1, "seed"]
        seed2 = feats.loc[t2, "seed"]
        if isinstance(seed1, pd.Series):
            seed1 = seed1.iloc[0]
        if isinstance(seed2, pd.Series):
            seed2 = seed2.iloc[0]
        p = adjust_prob_for_seed_rules(p, seed1, seed2, round_name, season)
        net1 = net_rating.get(t1, 0.0)
        net2 = net_rating.get(t2, 0.0)
        oe1 = feats.loc[t1, "season_adjOE"]
        de1 = feats.loc[t1, "season_adjDE"]
        oe2 = feats.loc[t2, "season_adjOE"]
        de2 = feats.loc[t2, "season_adjDE"]
        if isinstance(oe1, pd.Series):
            oe1 = oe1.iloc[0]
        if isinstance(de1, pd.Series):
            de1 = de1.iloc[0]
        if isinstance(oe2, pd.Series):
            oe2 = oe2.iloc[0]
        if isinstance(de2, pd.Series):
            de2 = de2.iloc[0]
        print(f"{round_name}: A) {name1} vs B) {name2} | P(A wins)={p:.3f}")
        print(
            f"   A seed {int(seed1)} | net {net1:.2f} | adjOE {oe1:.2f} | adjDE {de1:.2f}"
        )
        print(
            f"   B seed {int(seed2)} | net {net2:.2f} | adjOE {oe2:.2f} | adjDE {de2:.2f}"
        )
        choice = input("Choose A or B (default A): ").strip().lower()
        if choice == "b":
            return t2
        return t1

    region_order = ["W", "X", "Y", "Z"]
    round1_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    region_winners = {}
    for region in region_order:
        print(f"\n=== Region {region} ===")
        region_seeds = {}
        for seed_num, teams in bracket.get(region, {}).items():
            region_seeds[seed_num] = playin_winner(teams)

        r64 = [(region_seeds[a], region_seeds[b]) for a, b in round1_pairs]
        r32_winners = [pick_winner(t1, t2, "R64") for t1, t2 in r64]

        s16_matchups = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 8, 2)]
        s16_winners = [pick_winner(t1, t2, "R32") for t1, t2 in s16_matchups]

        e8_matchups = [(s16_winners[0], s16_winners[1]), (s16_winners[2], s16_winners[3])]
        e8_winners = [pick_winner(t1, t2, "S16") for t1, t2 in e8_matchups]

        f4_matchup = (e8_winners[0], e8_winners[1])
        region_winner = pick_winner(f4_matchup[0], f4_matchup[1], "E8")
        region_winners[region] = region_winner

    print("\n=== Final Four ===")
    f4_left = pick_winner(region_winners["W"], region_winners["X"], "F4")
    f4_right = pick_winner(region_winners["Y"], region_winners["Z"], "F4")

    print("\n=== Championship ===")
    champ = pick_winner(f4_left, f4_right, "F2")
    print(f"\nChampion: {id_to_name.get(champ, str(champ))}")
    return champ


def list_first_round_games(
    all_games,
    season_feats,
    model,
    feature_cols,
    weights,
    season=2026,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
):
    seeds_df = pd.read_csv(SEEDS_PATH)
    bracket = build_seed_bracket(seeds_df, season)

    feats_season = season_feats[season_feats["Season"] == season].copy()
    feats_season = feats_season.drop_duplicates(subset=["TeamID"])
    feature_cols_sim = feature_cols[: (len(weights) - 3)]
    team_ids = sorted({tid for region in bracket.values() for seeds in region.values() for tid in seeds})
    prob_raw_cache, seed_diff_stats = build_pairwise_prob_cache(
        team_ids, feats_season, feature_cols_sim, weights, model
    )
    seed_matchup_mean = build_seed_matchup_mean(all_games, season_feats, season)

    feats = feats_season.set_index("TeamID")
    seed_lookup = feats["seed"].to_dict()
    net_rating = (feats["season_adjOE"] - feats["season_adjDE"]).to_dict()

    def final_prob(t1, t2):
        prob_raw = prob_raw_cache[(t1, t2)]
        if not USE_SEED_FEATURES:
            prob_tuned = prob_raw
        else:
            seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
            stats = seed_diff_stats.get(seed_diff, None)
            mean_raw = stats["mean"] if stats else 0.5
            std_raw = stats["std"] if stats and not np.isnan(stats["std"]) else 0.0
            if std_raw < 1e-6:
                z = 0.0
            else:
                z = (prob_raw - mean_raw) / std_raw
            if abs(z) < seed_z_threshold:
                prob_tuned = prob_raw
            else:
                hist_mean = seed_matchup_mean.get(seed_diff, 0.5)
                hist_std = np.sqrt(hist_mean * (1 - hist_mean))
                prob_tuned = hist_mean + seed_alpha * z * hist_std
        prob_tuned = float(np.clip(prob_tuned, 0.01, 0.99))
        prob_shrunk = 0.5 + shrink * (prob_tuned - 0.5)
        prob_shrunk = float(np.clip(prob_shrunk, 0.01, 0.99))
        prob_rating = _normal_cdf((net_rating[t1] - net_rating[t2]) / rating_scale)
        prob_final = blend_weight * prob_shrunk + (1.0 - blend_weight) * prob_rating
        return float(np.clip(prob_final, 0.01, 0.99))

    def playin_dist(teams):
        if len(teams) == 1:
            return {teams[0]: 1.0}
        t1, t2 = teams[0], teams[1]
        p = final_prob(t1, t2)
        return {t1: p, t2: 1.0 - p}

    id_to_name = build_team_name_lookup()[1]
    round1_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    rows = []

    for region in ["W", "X", "Y", "Z"]:
        region_seeds = bracket.get(region, {})
        for a, b in round1_pairs:
            teams_a = region_seeds.get(a, [])
            teams_b = region_seeds.get(b, [])
            if not teams_a or not teams_b:
                continue
            dist_a = playin_dist(teams_a)
            dist_b = playin_dist(teams_b)

            # Expected probability higher seed (lower seed number) wins
            if a < b:
                higher_seed = a
            else:
                higher_seed = b

            prob_higher_wins = 0.0
            for ta, pa in dist_a.items():
                for tb, pb in dist_b.items():
                    if a < b:
                        prob_higher = final_prob(ta, tb)
                        higher_team = ta
                        lower_team = tb
                    else:
                        prob_higher = final_prob(tb, ta)
                        higher_team = tb
                        lower_team = ta
                    seed_h = seed_lookup.get(higher_team, np.nan)
                    seed_l = seed_lookup.get(lower_team, np.nan)
                    prob_higher = adjust_prob_for_seed_rules(prob_higher, seed_h, seed_l, "R64", season)
                    prob_higher_wins += pa * pb * prob_higher

            team_a_str = "/".join([id_to_name.get(t, str(t)) for t in teams_a])
            team_b_str = "/".join([id_to_name.get(t, str(t)) for t in teams_b])
            rows.append({
                "Region": region,
                "Matchup": f"{region}{a} vs {region}{b}",
                "SeedA_Teams": team_a_str,
                "SeedB_Teams": team_b_str,
                "Higher_Seed": higher_seed,
                "Prob_Higher_Seed_Wins": prob_higher_wins,
            })

    return pd.DataFrame(rows)


def list_actual_round_matchups(
    all_games,
    season_feats,
    season,
    xgb_params,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
):
    results = pd.read_csv(COMPACT_RESULTS_PATH)
    results = results[results["Season"] == season].copy()

    train_games = all_games[all_games["Season"] < season].copy()
    train_feats = season_feats[season_feats["Season"] < season].copy()
    coef_importance = feature_importance_regression(train_games, train_feats)
    X_weighted, y, feature_cols, weights, game_weights = build_matchup_training_weighted(
        train_games, train_feats, coef_importance
    )
    model = train_xgboost_model(X_weighted, y, game_weights=game_weights, params=xgb_params)

    seeds_df = pd.read_csv(SEEDS_PATH)
    bracket = build_seed_bracket(seeds_df, season)

    feats_season = season_feats[season_feats["Season"] == season].copy()
    feats_season = feats_season.drop_duplicates(subset=["TeamID"])
    feature_cols_sim = feature_cols[: (len(weights) - 3)]
    team_ids = sorted({tid for region in bracket.values() for seeds in region.values() for tid in seeds})
    prob_raw_cache, seed_diff_stats = build_pairwise_prob_cache(
        team_ids, feats_season, feature_cols_sim, weights, model
    )
    seed_matchup_mean = build_seed_matchup_mean(all_games, season_feats, season)

    feats = feats_season.set_index("TeamID")
    seed_lookup = feats["seed"].to_dict()
    net_rating = (feats["season_adjOE"] - feats["season_adjDE"]).to_dict()

    def final_prob(t1, t2):
        prob_raw = prob_raw_cache[(t1, t2)]
        if not USE_SEED_FEATURES:
            prob_tuned = prob_raw
        else:
            seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
            stats = seed_diff_stats.get(seed_diff, None)
            mean_raw = stats["mean"] if stats else 0.5
            std_raw = stats["std"] if stats and not np.isnan(stats["std"]) else 0.0
            if std_raw < 1e-6:
                z = 0.0
            else:
                z = (prob_raw - mean_raw) / std_raw
            if abs(z) < seed_z_threshold:
                prob_tuned = prob_raw
            else:
                hist_mean = seed_matchup_mean.get(seed_diff, 0.5)
                hist_std = np.sqrt(hist_mean * (1 - hist_mean))
                prob_tuned = hist_mean + seed_alpha * z * hist_std
        prob_tuned = float(np.clip(prob_tuned, 0.01, 0.99))
        prob_shrunk = 0.5 + shrink * (prob_tuned - 0.5)
        prob_shrunk = float(np.clip(prob_shrunk, 0.01, 0.99))
        prob_rating = _normal_cdf((net_rating[t1] - net_rating[t2]) / rating_scale)
        prob_final = blend_weight * prob_shrunk + (1.0 - blend_weight) * prob_rating
        return float(np.clip(prob_final, 0.01, 0.99))

    def actual_winner(t1, t2):
        m = results[
            ((results["WTeamID"] == t1) & (results["LTeamID"] == t2)) |
            ((results["WTeamID"] == t2) & (results["LTeamID"] == t1))
        ]
        if m.empty:
            return None
        return int(m.iloc[0]["WTeamID"])

    def resolve_play_ins(region_seeds):
        resolved = {}
        for seed_num, teams in region_seeds.items():
            if len(teams) == 1:
                resolved[seed_num] = teams[0]
            else:
                t1, t2 = teams[0], teams[1]
                winner = actual_winner(t1, t2)
                if winner is None:
                    winner = t1
                resolved[seed_num] = winner
        return resolved

    round1_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    id_to_name = build_team_name_lookup()[1]
    rows = []

    region_order = ["W", "X", "Y", "Z"]
    region_winners = {}

    for region in region_order:
        region_seeds = resolve_play_ins(bracket.get(region, {}))
        r64 = [(region_seeds[a], region_seeds[b]) for a, b in round1_pairs]
        r32_winners = []
        for t1, t2 in r64:
            winner = actual_winner(t1, t2)
            if winner is None:
                winner = t1
            prob_winner = final_prob(winner, t2 if winner == t1 else t1)
            seed_a = seed_lookup.get(t1, np.nan)
            seed_b = seed_lookup.get(t2, np.nan)
            seed_w = seed_lookup.get(winner, np.nan)
            seed_l = seed_b if winner == t1 else seed_a
            upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
            rows.append({
                "Season": season,
                "Round": "R64",
                "TeamA": id_to_name.get(t1, str(t1)),
                "SeedA": seed_a,
                "TeamB": id_to_name.get(t2, str(t2)),
                "SeedB": seed_b,
                "ActualWinner": id_to_name.get(winner, str(winner)),
                "Seed_ActualWinner": seed_w,
                "Upset": upset,
                "PredProb_ActualWinner": prob_winner,
            })
            r32_winners.append(winner)

        s16_matchups = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 8, 2)]
        s16_winners = []
        for t1, t2 in s16_matchups:
            winner = actual_winner(t1, t2)
            if winner is None:
                winner = t1
            prob_winner = final_prob(winner, t2 if winner == t1 else t1)
            seed_a = seed_lookup.get(t1, np.nan)
            seed_b = seed_lookup.get(t2, np.nan)
            seed_w = seed_lookup.get(winner, np.nan)
            seed_l = seed_b if winner == t1 else seed_a
            upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
            rows.append({
                "Season": season,
                "Round": "R32",
                "TeamA": id_to_name.get(t1, str(t1)),
                "SeedA": seed_a,
                "TeamB": id_to_name.get(t2, str(t2)),
                "SeedB": seed_b,
                "ActualWinner": id_to_name.get(winner, str(winner)),
                "Seed_ActualWinner": seed_w,
                "Upset": upset,
                "PredProb_ActualWinner": prob_winner,
            })
            s16_winners.append(winner)

        e8_matchups = [(s16_winners[0], s16_winners[1]), (s16_winners[2], s16_winners[3])]
        e8_winners = []
        for t1, t2 in e8_matchups:
            winner = actual_winner(t1, t2)
            if winner is None:
                winner = t1
            prob_winner = final_prob(winner, t2 if winner == t1 else t1)
            seed_a = seed_lookup.get(t1, np.nan)
            seed_b = seed_lookup.get(t2, np.nan)
            seed_w = seed_lookup.get(winner, np.nan)
            seed_l = seed_b if winner == t1 else seed_a
            upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
            rows.append({
                "Season": season,
                "Round": "S16",
                "TeamA": id_to_name.get(t1, str(t1)),
                "SeedA": seed_a,
                "TeamB": id_to_name.get(t2, str(t2)),
                "SeedB": seed_b,
                "ActualWinner": id_to_name.get(winner, str(winner)),
                "Seed_ActualWinner": seed_w,
                "Upset": upset,
                "PredProb_ActualWinner": prob_winner,
            })
            e8_winners.append(winner)

        f4_matchup = (e8_winners[0], e8_winners[1])
        winner = actual_winner(*f4_matchup)
        if winner is None:
            winner = f4_matchup[0]
        prob_winner = final_prob(winner, f4_matchup[1] if winner == f4_matchup[0] else f4_matchup[0])
        seed_a = seed_lookup.get(f4_matchup[0], np.nan)
        seed_b = seed_lookup.get(f4_matchup[1], np.nan)
        seed_w = seed_lookup.get(winner, np.nan)
        seed_l = seed_b if winner == f4_matchup[0] else seed_a
        upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
        rows.append({
            "Season": season,
            "Round": "E8",
            "TeamA": id_to_name.get(f4_matchup[0], str(f4_matchup[0])),
            "SeedA": seed_a,
            "TeamB": id_to_name.get(f4_matchup[1], str(f4_matchup[1])),
            "SeedB": seed_b,
            "ActualWinner": id_to_name.get(winner, str(winner)),
            "Seed_ActualWinner": seed_w,
            "Upset": upset,
            "PredProb_ActualWinner": prob_winner,
        })
        region_winners[region] = winner

    # Final Four
    f4_left = (region_winners["W"], region_winners["X"])
    f4_right = (region_winners["Y"], region_winners["Z"])
    winner_left = actual_winner(*f4_left) or f4_left[0]
    winner_right = actual_winner(*f4_right) or f4_right[0]
    seed_a = seed_lookup.get(f4_left[0], np.nan)
    seed_b = seed_lookup.get(f4_left[1], np.nan)
    seed_w = seed_lookup.get(winner_left, np.nan)
    seed_l = seed_b if winner_left == f4_left[0] else seed_a
    upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
    rows.append({
        "Season": season,
        "Round": "F4",
        "TeamA": id_to_name.get(f4_left[0], str(f4_left[0])),
        "SeedA": seed_a,
        "TeamB": id_to_name.get(f4_left[1], str(f4_left[1])),
        "SeedB": seed_b,
        "ActualWinner": id_to_name.get(winner_left, str(winner_left)),
        "Seed_ActualWinner": seed_w,
        "Upset": upset,
        "PredProb_ActualWinner": final_prob(winner_left, f4_left[1] if winner_left == f4_left[0] else f4_left[0]),
    })
    seed_a = seed_lookup.get(f4_right[0], np.nan)
    seed_b = seed_lookup.get(f4_right[1], np.nan)
    seed_w = seed_lookup.get(winner_right, np.nan)
    seed_l = seed_b if winner_right == f4_right[0] else seed_a
    upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
    rows.append({
        "Season": season,
        "Round": "F4",
        "TeamA": id_to_name.get(f4_right[0], str(f4_right[0])),
        "SeedA": seed_a,
        "TeamB": id_to_name.get(f4_right[1], str(f4_right[1])),
        "SeedB": seed_b,
        "ActualWinner": id_to_name.get(winner_right, str(winner_right)),
        "Seed_ActualWinner": seed_w,
        "Upset": upset,
        "PredProb_ActualWinner": final_prob(winner_right, f4_right[1] if winner_right == f4_right[0] else f4_right[0]),
    })

    # Championship
    champ_match = (winner_left, winner_right)
    champ = actual_winner(*champ_match) or champ_match[0]
    seed_a = seed_lookup.get(champ_match[0], np.nan)
    seed_b = seed_lookup.get(champ_match[1], np.nan)
    seed_w = seed_lookup.get(champ, np.nan)
    seed_l = seed_b if champ == champ_match[0] else seed_a
    upset = bool(seed_w > seed_l) if np.isfinite(seed_w) and np.isfinite(seed_l) else False
    rows.append({
        "Season": season,
        "Round": "F2",
        "TeamA": id_to_name.get(champ_match[0], str(champ_match[0])),
        "SeedA": seed_a,
        "TeamB": id_to_name.get(champ_match[1], str(champ_match[1])),
        "SeedB": seed_b,
        "ActualWinner": id_to_name.get(champ, str(champ)),
        "Seed_ActualWinner": seed_w,
        "Upset": upset,
        "PredProb_ActualWinner": final_prob(champ, champ_match[1] if champ == champ_match[0] else champ_match[0]),
    })

    return pd.DataFrame(rows)


def compute_seed_matchup_thresholds(
    all_games,
    season_feats,
    seasons,
    xgb_params,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
    target_win_pct=0.95,
    min_games=10,
):
    rows = []
    for season in seasons:
        train_games = all_games[all_games["Season"] < season].copy()
        train_feats = season_feats[season_feats["Season"] < season].copy()
        test_games = all_games[
            (all_games["Season"] == season) &
            (all_games["is_tourney"] == 1)
        ].copy()
        test_feats = season_feats[season_feats["Season"] == season].copy()
        if test_games.empty:
            continue

        # Seed expectations for tuning
        train_tourney = train_games[train_games["is_tourney"] == 1].copy()
        team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
        train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
        train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)
        seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
        seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]
        df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
        df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
        df_seed = pd.concat([df_w, df_l])
        seed_matchup_mean = df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()

        coef_importance = feature_importance_regression(train_games, train_feats)
        X_train, y_train, feature_cols, weights, game_weights = build_matchup_training_weighted(
            train_games, train_feats, coef_importance
        )
        model = train_xgboost_model(X_train, y_train, game_weights, params=xgb_params)

        feats = test_feats.set_index(["Season", "TeamID"])
        g = test_games.copy()
        feats_team = feats.add_suffix("_team")
        feats_opp = feats.add_suffix("_opp")
        g = g.merge(feats_team, left_on=["Season", "TeamID"], right_index=True, how="left")
        g = g.merge(feats_opp, left_on=["Season", "opp_team_id"], right_index=True, how="left")

        X_team = g[[f"{c}_team" for c in feature_cols]].values
        X_opp = g[[f"{c}_opp" for c in feature_cols]].values
        seed_diff = (g["seed_team"] - g["seed_opp"]).values.reshape(-1, 1)
        if not USE_SEED_FEATURES:
            seed_diff = np.zeros_like(seed_diff)
        off_vs_def = (g["season_adjOE_team"] - g["season_adjDE_opp"]).values.reshape(-1, 1)
        def_vs_off = (g["season_adjDE_team"] - g["season_adjOE_opp"]).values.reshape(-1, 1)

        X_test = np.hstack([X_team - X_opp, seed_diff, off_vs_def, def_vs_off])
        base_weights = np.abs(coef_importance.reindex([f"{c}_diff" for c in feature_cols])).fillna(1.0).values
        weights_array = np.concatenate([base_weights, np.ones(3)])
        X_test_weighted = X_test * weights_array

        probs_raw = model.predict_proba(X_test_weighted)[:, 1]

        if USE_SEED_FEATURES:
            seed_diff_flat = seed_diff.flatten()
            seed_diff_stats = (
                pd.DataFrame({"Seed_Diff": seed_diff_flat, "Prob": probs_raw})
                .groupby("Seed_Diff")["Prob"]
                .agg(["mean", "std", "count"])
                .to_dict("index")
            )

            def tune_prob_seed_relative(prob, seed_diff_val):
                stats = seed_diff_stats.get(seed_diff_val, None)
                mean_raw = stats["mean"] if stats else 0.5
                std_raw = stats["std"] if stats and not np.isnan(stats["std"]) else 0.0
                if std_raw < 1e-6:
                    z = 0.0
                else:
                    z = (prob - mean_raw) / std_raw
                if abs(z) < seed_z_threshold:
                    return float(np.clip(prob, 0.01, 0.99))
                hist_mean = seed_matchup_mean.get(seed_diff_val, 0.5)
                hist_std = np.sqrt(hist_mean * (1 - hist_mean))
                tuned = hist_mean + seed_alpha * z * hist_std
                return float(np.clip(tuned, 0.01, 0.99))

            probs_tuned = np.array([
                tune_prob_seed_relative(p, sd) for p, sd in zip(probs_raw, seed_diff_flat)
            ])
        else:
            probs_tuned = probs_raw.copy()

        net_team = g["season_adjOE_team"] - g["season_adjDE_team"]
        net_opp = g["season_adjOE_opp"] - g["season_adjDE_opp"]
        rating_diff = (net_team - net_opp).values
        probs_rating = np.array([_normal_cdf(d / rating_scale) for d in rating_diff])

        probs_shrunk = 0.5 + shrink * (probs_tuned - 0.5)
        probs_shrunk = np.clip(probs_shrunk, 0.01, 0.99)
        probs_blend = blend_weight * probs_shrunk + (1.0 - blend_weight) * probs_rating
        probs_blend = np.clip(probs_blend, 0.01, 0.99)

        y_test = g["Win"].values
        mask = g["TeamID"] < g["opp_team_id"]

        seed_team = g["seed_team"].values
        seed_opp = g["seed_opp"].values

        for i in np.where(mask)[0]:
            t_seed = seed_team[i]
            o_seed = seed_opp[i]
            if pd.isna(t_seed) or pd.isna(o_seed):
                continue
            t_seed = int(t_seed)
            o_seed = int(o_seed)
            higher_seed = min(t_seed, o_seed)
            lower_seed = max(t_seed, o_seed)
            prob_team_wins = probs_blend[i]
            team_is_higher = t_seed < o_seed
            prob_higher = prob_team_wins if team_is_higher else 1.0 - prob_team_wins
            higher_won = y_test[i] if team_is_higher else 1 - y_test[i]
            rows.append({
                "Season": season,
                "HigherSeed": higher_seed,
                "LowerSeed": lower_seed,
                "ProbHigher": prob_higher,
                "HigherWon": higher_won,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    thresholds = []
    for (hs, ls), grp in df.groupby(["HigherSeed", "LowerSeed"]):
        grp = grp.sort_values("ProbHigher", ascending=False).reset_index(drop=True)
        if len(grp) < min_games:
            continue
        grp["cum_wins"] = grp["HigherWon"].cumsum()
        grp["cum_games"] = np.arange(1, len(grp) + 1)
        grp["cum_win_pct"] = grp["cum_wins"] / grp["cum_games"]
        ok = grp[(grp["cum_games"] >= min_games) & (grp["cum_win_pct"] >= target_win_pct)]
        if ok.empty:
            continue
        threshold = ok.iloc[-1]["ProbHigher"]
        thresholds.append({
            "HigherSeed": hs,
            "LowerSeed": ls,
            "Threshold": float(threshold),
            "GamesUsed": int(ok.iloc[-1]["cum_games"]),
            "WinPct": float(ok.iloc[-1]["cum_win_pct"]),
        })

    return pd.DataFrame(thresholds).sort_values(["HigherSeed", "LowerSeed"])

def first_round_for_season(
    all_games,
    season_feats,
    season,
    xgb_params,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
):
    train_games = all_games[all_games["Season"] < season].copy()
    train_feats = season_feats[season_feats["Season"] < season].copy()
    coef_importance = feature_importance_regression(train_games, train_feats)
    X_weighted, y, feature_cols, weights, game_weights = build_matchup_training_weighted(
        train_games, train_feats, coef_importance
    )
    model = train_xgboost_model(X_weighted, y, game_weights=game_weights, params=xgb_params)

    fr = list_first_round_games(
        all_games=all_games,
        season_feats=season_feats,
        model=model,
        feature_cols=feature_cols,
        weights=weights,
        season=season,
        rating_scale=rating_scale,
        shrink=shrink,
        blend_weight=blend_weight,
        seed_alpha=seed_alpha,
        seed_z_threshold=seed_z_threshold,
    )
    return fr


def predict_matchup_by_name(
    team_a,
    team_b,
    season_feats_2026,
    feature_cols,
    weights,
    model,
    rating_scale=8.0,
    shrink=1.0,
    blend_weight=0.55,
    seed_alpha=0.7,
    seed_z_threshold=1.0,
):
    name_to_id, id_to_name = build_team_name_lookup()
    t1 = name_to_id.get(normalize_team_name(team_a))
    t2 = name_to_id.get(normalize_team_name(team_b))
    if t1 is None or t2 is None:
        raise ValueError("One or both team names not found in MTeams.csv.")

    feats = season_feats_2026.set_index("TeamID")
    if t1 not in feats.index or t2 not in feats.index:
        raise ValueError("One or both teams missing 2026 features.")

    x_team = feats.loc[t1, feature_cols].values
    x_opp = feats.loc[t2, feature_cols].values

    seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
    if not USE_SEED_FEATURES:
        seed_diff = 0
    off_vs_def = feats.loc[t1, "season_adjOE"] - feats.loc[t2, "season_adjDE"]
    def_vs_off = feats.loc[t1, "season_adjDE"] - feats.loc[t2, "season_adjOE"]

    X = np.append(x_team - x_opp, [seed_diff, off_vs_def, def_vs_off]).reshape(1, -1)
    X_weighted = X * weights

    prob_raw = float(model.predict_proba(X_weighted)[:, 1][0])

    # Seed-diff tuning using historical seed matchup mean (from 2014-2025 training)
    # Build mapping once from all available seasons prior to 2026
    train_feats = season_feats_2026.copy()
    train_feats = train_feats[train_feats["Season"] < 2026]
    all_games, _ = build_team_season_features(list(range(2014, 2026)))
    train_tourney = all_games[all_games["is_tourney"] == 1].copy()
    team_seed = dict(zip(train_feats["TeamID"], train_feats["seed"]))
    train_tourney["seed_w"] = train_tourney["WTeamID"].map(team_seed)
    train_tourney["seed_l"] = train_tourney["LTeamID"].map(team_seed)
    seed_diff_wl = train_tourney["seed_w"] - train_tourney["seed_l"]
    seed_diff_lw = train_tourney["seed_l"] - train_tourney["seed_w"]
    df_w = pd.DataFrame({"Seed_Diff": seed_diff_wl, "Win": 1})
    df_l = pd.DataFrame({"Seed_Diff": seed_diff_lw, "Win": 0})
    df_seed = pd.concat([df_w, df_l])
    seed_matchup_mean = df_seed.groupby("Seed_Diff")["Win"].mean().to_dict()

    hist_mean = seed_matchup_mean.get(seed_diff, 0.5)
    hist_std = np.sqrt(hist_mean * (1 - hist_mean))
    # Without per-seed std for this single matchup, apply only when extreme
    prob_tuned = prob_raw
    if abs(prob_raw - hist_mean) >= seed_z_threshold * hist_std:
        direction = np.sign(prob_raw - hist_mean)
        prob_tuned = hist_mean + seed_alpha * direction * hist_std
    prob_tuned = float(np.clip(prob_tuned, 0.01, 0.99))

    prob_shrunk = 0.5 + shrink * (prob_tuned - 0.5)
    prob_shrunk = float(np.clip(prob_shrunk, 0.01, 0.99))

    net_t1 = feats.loc[t1, "season_adjOE"] - feats.loc[t1, "season_adjDE"]
    net_t2 = feats.loc[t2, "season_adjOE"] - feats.loc[t2, "season_adjDE"]
    prob_rating = _normal_cdf((net_t1 - net_t2) / rating_scale)

    prob_final = blend_weight * prob_shrunk + (1.0 - blend_weight) * prob_rating
    prob_final = float(np.clip(prob_final, 0.01, 0.99))

    return {
        "team_a": id_to_name.get(t1, str(t1)),
        "team_b": id_to_name.get(t2, str(t2)),
        "prob_team_a_wins": prob_final,
    }

# --------------------- LOAD GAME RESULTS ---------------------
def load_team_schedule_results(team_id, year, details=None, torvik_mapping=None, torvik_stats=None):
    if details is None:
        reg = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, "MRegularSeasonDetailedResults.csv"))
        tourney = pd.read_csv(TOURNEY_RESULTS_PATH)
        reg["is_tourney"] = 0
        tourney["is_tourney"] = 1
        details = pd.concat([reg, tourney], ignore_index=True)
    if torvik_mapping is None:
        torvik_mapping = pd.read_csv(TORVIK_MAP_PATH)
    if torvik_stats is None:
        torvik_stats = pd.read_csv(os.path.join(DATA_DIR, f"{year}_team_results.csv"))

    torvik_stats["team_norm"] = torvik_stats["team"].map(normalize_team_name)
    torvik_mapping["team_norm"] = torvik_mapping["team"].map(normalize_team_name)
    id_to_name = dict(zip(torvik_mapping["TeamID"], torvik_mapping["team"]))
    team_name = id_to_name.get(team_id)

    wins = details.loc[
        (details["WTeamID"] == team_id) & (details["Season"] == year)
    ].copy()
    losses = details.loc[
        (details["LTeamID"] == team_id) & (details["Season"] == year)
    ].copy()

    wins["Win"] = 1
    losses["Win"] = 0

    wins["opp_team"] = wins["LTeamID"].map(id_to_name)
    losses["opp_team"] = losses["WTeamID"].map(id_to_name)

    wins["team"] = team_name
    losses["team"] = team_name

    results = pd.concat([wins, losses])
    results["team_norm"] = results["team"].map(normalize_team_name)
    results["opp_team_norm"] = results["opp_team"].map(normalize_team_name)

    results["adjOE"] = np.where(
        results["Win"] == 1,
        100 * results["WScore"] / (results["WFGA"] - results["WOR"] + results["WTO"] + 0.475 * results["WFTA"]),
        100 * results["LScore"] / (results["LFGA"] - results["LOR"] + results["LTO"] + 0.475 * results["LFTA"]),
    )

    results["adjDE"] = np.where(
        results["Win"] == 1,
        100 * results["LScore"] / (results["LFGA"] - results["LOR"] + results["LTO"] + 0.475 * results["LFTA"]),
        100 * results["WScore"] / (results["WFGA"] - results["WOR"] + results["WTO"] + 0.475 * results["WFTA"]),
    )

    results["TO%"] = np.where(
        results["Win"] == 1,
        100 * results["WTO"] / (results["WFGA"] - results["WOR"] + results["WTO"] + 0.475 * results["WFTA"]),
        100 * results["LTO"] / (results["LFGA"] - results["LOR"] + results["LTO"] + 0.475 * results["LFTA"]),
    )
    results["season_TO%"] = results["TO%"].mean()

    # ---------- 3PT PERCENTAGE (GAME LEVEL) ----------
    three_fgm = np.where(results["Win"] == 1, results["WFGM3"], results["LFGM3"])
    three_fga = np.where(results["Win"] == 1, results["WFGA3"], results["LFGA3"])
    results["three_pt_pct"] = np.where(three_fga > 0, three_fgm / three_fga, np.nan)

    # ---------- OPPONENT 3PT PERCENTAGE / RATE (GAME LEVEL) ----------
    opp_three_fgm = np.where(results["Win"] == 1, results["LFGM3"], results["WFGM3"])
    opp_three_fga = np.where(results["Win"] == 1, results["LFGA3"], results["WFGA3"])
    opp_fga = np.where(results["Win"] == 1, results["LFGA"], results["WFGA"])
    results["opp_three_pt_pct"] = np.where(opp_three_fga > 0, opp_three_fgm / opp_three_fga, np.nan)
    results["opp_three_pt_rate"] = np.where(opp_fga > 0, opp_three_fga / opp_fga, np.nan)

    # ---------- TEAM SEASON STATS ----------
    results = results.merge(
        torvik_stats[["team_norm", "adjoe", "adjde", "sos", "ncsos"]],
        left_on="team_norm",
        right_on="team_norm",
        how="left"
    )

    results = results.rename(columns={
        "adjoe": "season_adjOE",
        "adjde": "season_adjDE",
        "sos": "season_sos",
        "ncsos": "season_ncsos",
    })

    # ---------- OPPONENT SEASON STATS ----------
    results = results.merge(
        torvik_stats[["team_norm", "adjoe", "adjde"]],
        left_on="opp_team_norm",
        right_on="team_norm",
        how="left",
        suffixes=("", "_opp")
    )

    results = results.rename(columns={
        "adjoe": "opp_season_adjOE",
        "adjde": "opp_season_adjDE"
    })

    results = results.drop(columns=["team_norm", "opp_team_norm"])

    return results[[
        "Win",
        "team",
        "opp_team",
        "WTeamID",
        "LTeamID",
        "DayNum",
        "WScore",
        "LScore",
        "adjOE",
        "adjDE",
        "TO%",
        "three_pt_pct",
        "opp_three_pt_pct",
        "opp_three_pt_rate",
        "season_TO%",
        "season_adjOE",
        "season_adjDE",
        "season_sos",
        "season_ncsos",
        "opp_season_adjOE",
        "opp_season_adjDE",
        "is_tourney"
    ]]

# --------------------- BUILD SEASON FEATURES ---------------------
def build_team_season_features(years):
    reg = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, "MRegularSeasonDetailedResults.csv"))
    tourney = pd.read_csv(TOURNEY_RESULTS_PATH)
    reg["is_tourney"] = 0
    tourney["is_tourney"] = 1
    details = pd.concat([reg, tourney], ignore_index=True)
    torvik_mapping = pd.read_csv(TORVIK_MAP_PATH)
    seeds = pd.read_csv(SEEDS_PATH)
    seeds["seed"] = seeds["Seed"].str[1:3].astype(int)
    all_rows = []

    for year in years:
        torvik_path = os.path.join(DATA_DIR, f"{year}_team_results.csv")
        if not os.path.exists(torvik_path):
            continue
        torvik_stats = pd.read_csv(torvik_path)
        team_ids = pd.unique(
            details.loc[details["Season"] == year, ["WTeamID", "LTeamID"]].values.ravel()
        )
        for team_id in team_ids:
            games = load_team_schedule_results(
                team_id,
                year,
                details=details,
                torvik_mapping=torvik_mapping,
                torvik_stats=torvik_stats,
            )
            if games.empty:
                continue
            games = games.copy()
            games["TeamID"] = team_id
            games["Season"] = year

            # ----------------- CHANGED -----------------
            # Map opponent names to TeamIDs to fix merge issues
            team_name_to_id = dict(zip(torvik_mapping["team"], torvik_mapping["TeamID"]))
            games["opp_team_id"] = games["opp_team"].map(team_name_to_id)
            # -------------------------------------------

            # Opponent quality score and quartiles
            games["opp_quality"] = games["opp_season_adjOE"] - games["opp_season_adjDE"]
            all_rows.append(games)

    all_games = pd.concat(all_rows, ignore_index=True)

    # Build season-level features
    season_feats = all_games.groupby(["Season", "TeamID"]).agg(
        adjOE=("adjOE", "mean"),
        adjDE=("adjDE", "mean"),
        to_pct=("TO%", "mean"),
        three_pt_pct=("three_pt_pct", "mean"),
        three_pt_pct_std=("three_pt_pct", "std"),
        opp_three_pt_pct=("opp_three_pt_pct", "mean"),
        opp_three_pt_rate=("opp_three_pt_rate", "mean"),
        season_to_pct=("season_TO%", "mean"),
        season_adjOE=("season_adjOE", "mean"),
        season_adjDE=("season_adjDE", "mean"),
        season_sos=("season_sos", "mean"),
        season_ncsos=("season_ncsos", "mean"),
    )

    season_feats = season_feats.reset_index()

    season_feats = season_feats.merge(
        seeds[["Season","TeamID","seed"]],
        on=["Season","TeamID"],
        how="left"
    )

    season_feats["net_rating"] = season_feats["season_adjOE"] - season_feats["season_adjDE"]
    # Use net rating instead of OE*DE product
    season_feats["efficiency_product"] = season_feats["net_rating"]
    season_feats["seed"] = season_feats["seed"].fillna(20)
    season_feats["season_sos"] = season_feats["season_sos"].fillna(0)
    season_feats["season_ncsos"] = season_feats["season_ncsos"].fillna(0)
    season_feats["opp_three_pt_pct"] = season_feats["opp_three_pt_pct"].fillna(0)
    season_feats["opp_three_pt_rate"] = season_feats["opp_three_pt_rate"].fillna(0)

    # ---------- 3PT HOTNESS (LAST 5 GAMES) ----------
    all_games_sorted = all_games.sort_values(["Season", "TeamID", "DayNum"])
    last5_3p = (
        all_games_sorted.groupby(["Season", "TeamID"])["three_pt_pct"]
        .apply(lambda s: s.tail(5).mean())
        .reset_index(name="three_pt_pct_last5")
    )
    season_feats = season_feats.merge(
        last5_3p,
        on=["Season", "TeamID"],
        how="left"
    )
    season_feats["three_pt_hot"] = season_feats["three_pt_pct_last5"] - season_feats["three_pt_pct"]

    season_feats["three_pt_pct"] = season_feats["three_pt_pct"].fillna(0)
    season_feats["three_pt_pct_std"] = season_feats["three_pt_pct_std"].fillna(0)
    season_feats["three_pt_pct_last5"] = season_feats["three_pt_pct_last5"].fillna(0)
    season_feats["three_pt_hot"] = season_feats["three_pt_hot"].fillna(0)

    # Performance vs opponent quality quartiles per season
    all_games["opp_q"] = all_games.groupby("Season")["opp_quality"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop")
    )
    opp_perf = (
        all_games.groupby(["Season", "TeamID", "opp_q"])["Win"]
        .mean()
        .unstack(fill_value=0)
    )
    for q in [0, 1, 2, 3]:
        if q not in opp_perf.columns:
            opp_perf[q] = 0.0
    opp_perf = opp_perf.sort_index(axis=1)
    opp_perf.columns = [f"win_pct_opp_q{int(c)+1}" for c in opp_perf.columns]

    opp_games = (
        all_games.groupby(["Season", "TeamID", "opp_q"])["Win"]
        .size()
        .unstack(fill_value=0)
    )
    for q in [0, 1, 2, 3]:
        if q not in opp_games.columns:
            opp_games[q] = 0
    opp_games = opp_games.sort_index(axis=1)
    opp_games.columns = [f"games_opp_q{int(c)+1}" for c in opp_games.columns]

    # Wins per quartile for shrinkage
    opp_wins = opp_perf * opp_games
    opp_wins.columns = [c.replace("win_pct", "wins") for c in opp_wins.columns]

    season_feats = season_feats.merge(
        opp_perf.reset_index(),
        on=["Season", "TeamID"],
        how="left"
    ).fillna(0)
    season_feats = season_feats.merge(
        opp_games.reset_index(),
        on=["Season", "TeamID"],
        how="left"
    ).fillna(0)
    season_feats = season_feats.merge(
        opp_wins.reset_index(),
        on=["Season", "TeamID"],
        how="left"
    ).fillna(0)

    # Shrink win% toward season win rate when games are few
    k = 4.0
    season_win = all_games.groupby(["Season", "TeamID"])["Win"].mean().reset_index(name="season_win_pct")
    season_feats = season_feats.merge(season_win, on=["Season", "TeamID"], how="left").fillna(0)
    for q in [1, 2, 3, 4]:
        games_col = f"games_opp_q{q}"
        wins_col = f"wins_opp_q{q}"
        if games_col not in season_feats.columns:
            season_feats[games_col] = 0.0
        if wins_col not in season_feats.columns:
            season_feats[wins_col] = 0.0
    for q in [1, 2, 3, 4]:
        wins_col = f"wins_opp_q{q}"
        games_col = f"games_opp_q{q}"
        adj_col = f"win_pct_opp_q{q}_adj"
        season_feats[adj_col] = (
            season_feats[wins_col] + k * season_feats["season_win_pct"]
        ) / (season_feats[games_col] + k)

    # Drop raw quartile win% to avoid misleading small-sample signals
    season_feats = season_feats.drop(
        columns=[
            "win_pct_opp_q1",
            "win_pct_opp_q2",
            "win_pct_opp_q3",
            "win_pct_opp_q4",
        ],
        errors="ignore",
    )
    season_feats = season_feats.reset_index()

    # Conference tournament wins/losses
    conf = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, "MConferenceTourneyGames.csv"))
    conf_w = conf.groupby(["Season", "WTeamID"]).size().reset_index(name="conf_tourney_wins")
    conf_l = conf.groupby(["Season", "LTeamID"]).size().reset_index(name="conf_tourney_losses")
    conf_w = conf_w.rename(columns={"WTeamID": "TeamID"})
    conf_l = conf_l.rename(columns={"LTeamID": "TeamID"})
    season_feats = season_feats.merge(conf_w, on=["Season", "TeamID"], how="left")
    season_feats = season_feats.merge(conf_l, on=["Season", "TeamID"], how="left")
    season_feats["conf_tourney_wins"] = season_feats["conf_tourney_wins"].fillna(0).astype(int)
    season_feats["conf_tourney_losses"] = season_feats["conf_tourney_losses"].fillna(0).astype(int)
    season_feats["conf_tourney_games"] = (
        season_feats["conf_tourney_wins"] + season_feats["conf_tourney_losses"]
    )
    season_feats["conf_tourney_win_pct"] = np.where(
        season_feats["conf_tourney_games"] > 0,
        season_feats["conf_tourney_wins"] / season_feats["conf_tourney_games"],
        0.0,
    )

    # Drop redundant / noisy columns and merge artifacts
    redundant_cols = [
        "adjOE",
        "adjDE",
        "season_to_pct",
        "efficiency_product",
        "three_pt_pct_std",
        "three_pt_pct_last5",
        "three_pt_hot",
        "games_opp_q1",
        "games_opp_q2",
        "games_opp_q3",
        "games_opp_q4",
        "wins_opp_q1",
        "wins_opp_q2",
        "wins_opp_q3",
        "wins_opp_q4",
        "season_win_pct",
    ]
    season_feats = season_feats.drop(columns=redundant_cols, errors="ignore")

    # Remove any merge artifact columns like *_x / *_y
    artifact_cols = [c for c in season_feats.columns if c.endswith("_x") or c.endswith("_y")]
    if artifact_cols:
        season_feats = season_feats.drop(columns=artifact_cols, errors="ignore")

    return all_games, season_feats

# --------------------- BUILD MATCHUP TRAINING (weighted) ---------------------
def build_matchup_training_weighted(all_games, season_feats, coef_importance):
    feats = season_feats.set_index(["Season", "TeamID"])
    g = all_games.copy()

    # Pre-rename columns to avoid conflicts
    feats_team = feats.add_suffix("_team")
    feats_opp = feats.add_suffix("_opp")

    g = g.merge(
        feats_team,
        left_on=["Season", "TeamID"],
        right_index=True,
        how="left"
    )

    g = g.merge(
        feats_opp,
        left_on=["Season", "opp_team_id"],
        right_index=True,
        how="left"
    )

    feature_cols = [c for c in feats.columns]
    X_team = g[[f"{c}_team" for c in feature_cols]].values
    X_opp = g[[f"{c}_opp" for c in feature_cols]].values

    seed_team = g["seed_team"].values
    seed_opp = g["seed_opp"].values
    seed_diff = (seed_team - seed_opp).reshape(-1,1)
    if not USE_SEED_FEATURES:
        seed_diff = np.zeros_like(seed_diff)

    # OFFENSE vs DEFENSE INTERACTIONS
    off_vs_def = (
        g["season_adjOE_team"] -
        g["season_adjDE_opp"]
    ).values.reshape(-1,1)

    def_vs_off = (
        g["season_adjDE_team"] -
        g["season_adjOE_opp"]
    ).values.reshape(-1,1)

    X = np.hstack([
        X_team - X_opp,
        seed_diff,
        off_vs_def,
        def_vs_off
    ])
    y = g["Win"].values

    feature_cols = list(feats.columns)

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    is_tourney = g.loc[mask, "is_tourney"].values

    # symmetric matchups
    X_rev = -X
    y_rev = 1 - y
    is_tourney_rev = is_tourney.copy()

    X = np.vstack([X, X_rev])
    y = np.concatenate([y, y_rev])
    is_tourney = np.concatenate([is_tourney, is_tourney_rev])

    # weighting: regular=1.0, NCAA=2.0
    game_weights = np.where(is_tourney == 1, 2.0, 1.0)

    # --------------------- FEATURE WEIGHTING ---------------------
    base_weights = np.abs(
        coef_importance.reindex([f"{c}_diff" for c in feature_cols])
    ).fillna(1.0).values

    extra_weights = np.ones(3)  # seed_diff, off_vs_def, def_vs_off

    weights = np.concatenate([base_weights, extra_weights])
    X_weighted = X * weights

    return X_weighted, y, feature_cols, weights, game_weights



# --------------------- TRAIN CALIBRATED XGBOOST ---------------------
def train_xgboost_model(X_weighted, y, game_weights, params=None):
    if params is None:
        params = {}
    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        eval_metric="logloss",
    )
    model.fit(X_weighted, y, sample_weight=game_weights)

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(X_weighted, y, sample_weight=game_weights)
    return calibrated_model

# --------------------- PREDICT 2026 MATCHUP MATRIX (weighted) ---------------------
def predict_2026_matchup_matrix_weighted(model, season_feats_2026, weights):

    feats = season_feats_2026.set_index("TeamID")
    feature_cols = [c for c in feats.columns if c not in ["Season", "TeamID"]]

    teams = feats.index.values
    n = len(teams)

    prob_matrix = pd.DataFrame(np.nan, index=teams, columns=teams)

    X_rows = []
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):

            t1, t2 = teams[i], teams[j]

            seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
            if not USE_SEED_FEATURES:
                seed_diff = 0

            off_vs_def = (
                feats.loc[t1, "season_adjOE"] -
                feats.loc[t2, "season_adjDE"]
            )

            def_vs_off = (
                feats.loc[t1, "season_adjDE"] -
                feats.loc[t2, "season_adjOE"]
            )

            x = np.append(
                feats.loc[t1, feature_cols].values -
                feats.loc[t2, feature_cols].values,
                [
                    seed_diff,
                    off_vs_def,
                    def_vs_off
                ]
            )

            X_rows.append(x)
            pairs.append((t1, t2))

    X = np.vstack(X_rows)

    # apply feature weights
    X_weighted = X * weights

    # predict all games at once
    probs = model.predict_proba(X_weighted)[:, 1]

    for k, (t1, t2) in enumerate(pairs):

        p = probs[k]

        prob_matrix.loc[t1, t2] = p
        prob_matrix.loc[t2, t1] = 1 - p

    return prob_matrix
# --------------------- FEATURE IMPORTANCE REGRESSION ---------------------
def feature_importance_regression(all_games, season_feats):
    feats = season_feats.set_index(["Season", "TeamID"])
    g = all_games.copy()
    
    # Pre-rename columns
    feats_team = feats.add_suffix("_team")
    feats_opp = feats.add_suffix("_opp")
    
    g = g.merge(
        feats_team,
        left_on=["Season", "TeamID"],
        right_index=True,
        how="left"
    )
    
    g = g.merge(
        feats_opp,
        left_on=["Season", "opp_team_id"],
        right_index=True,
        how="left"
    )
    
    feature_cols = [c for c in feats.columns]
    X_team = g[[f"{c}_team" for c in feature_cols]].values
    X_opp = g[[f"{c}_opp" for c in feature_cols]].values
    X = X_team - X_opp
    y = g["Win"].values

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    
    y = y[mask]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    
    coef = pd.Series(model.coef_[0], index=[f"{c}_diff" for c in feature_cols])
    coef = coef.sort_values(key=abs, ascending=False)
    
    return coef

# --------------------- MAIN ---------------------
def main():
    years = list(range(2014, 2027))
    all_games, season_feats = build_team_season_features(years)

    RUN_INTERACTIVE = False
    RUN_SIM = False
    RUN_FIRST_ROUND = False
    RUN_BRACKET_PICK = True
    RUN_ACTUAL_ROUNDS = False
    RUN_SEED_THRESHOLDS = False
    RUN_HYPERPARAM_SEARCH = False
    RUN_EVAL = False
    RUN_BRIER_REPORT = False
    if RUN_HYPERPARAM_SEARCH:
        # Hyperparameter grid search for XGBoost (Brier-focused)
        grid_search_xgb_hyperparams(all_games, season_feats)
        return

    train_games = all_games[all_games["Season"] <= 2025].copy()
    train_feats = season_feats[season_feats["Season"] <= 2025].copy()

    # Compute regression-based feature importance
    coef_importance = feature_importance_regression(train_games, train_feats)

    # Tuned parameters from grid search
    shrink = 1.0
    rating_scale = 8.0
    blend_weight = 0.55
    seed_alpha = 0.7
    seed_z_threshold = 1.0
    xgb_params = {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.035,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Evaluate with tuned parameters
    if RUN_EVAL:
        evaluate_model_by_season(
            all_games,
            season_feats,
            shrink=shrink,
            rating_scale=rating_scale,
            blend_weight=blend_weight,
            seed_alpha=seed_alpha,
            seed_z_threshold=seed_z_threshold,
            xgb_params=xgb_params,
            verbose=True,
        )
        return

    if RUN_BRIER_REPORT:
        strengths = [1.25, 1.5, 2, 3, 5, 10]
        rows = []
        for strength in strengths:
            report = brier_report(
                all_games,
                season_feats,
                seasons=[2018, 2019, 2021, 2022, 2023, 2024, 2025],
                shrink=shrink,
                rating_scale=rating_scale,
                blend_weight=blend_weight,
                seed_alpha=seed_alpha,
                seed_z_threshold=seed_z_threshold,
                xgb_params=xgb_params,
                apply_seed_rules=True,
                apply_calibration=False,
                adjust_strength=strength,
            )
            avg_row = report[report["Season"] == "Average"].iloc[0]
            rows.append({
                "AdjustStrength": strength,
                "AvgBrier": avg_row["Brier"],
            })
        grid = pd.DataFrame(rows).sort_values("AvgBrier")
        print("\n=== Adjust Strength Grid (Rules, no calibration) ===")
        print(grid.to_string(index=False))
        return

    # Build weighted matchup training
    X_weighted, y, feature_cols, weights, game_weights = build_matchup_training_weighted(
        train_games, train_feats, coef_importance
    )

    # Train weighted XGBoost
    model = train_xgboost_model(X_weighted, y, game_weights=game_weights, params=xgb_params)

    # Interactive single-matchup mode
    if RUN_INTERACTIVE:
        while True:
            team_a = input("Team A (or blank to quit): ").strip()
            if not team_a:
                break
            team_b = input("Team B: ").strip()
            if not team_b:
                break
            result = predict_matchup_by_name(
                team_a,
                team_b,
                season_feats_2026=season_feats[season_feats["Season"] == 2026].copy(),
                feature_cols=feature_cols,
                weights=weights,
                model=model,
                rating_scale=rating_scale,
                shrink=shrink,
                blend_weight=blend_weight,
                seed_alpha=seed_alpha,
                seed_z_threshold=seed_z_threshold,
            )
            print(f"{result['team_a']} vs {result['team_b']}: {result['prob_team_a_wins']:.4f}")
        return

    if RUN_BRACKET_PICK:
        run_interactive_bracket(
            all_games=all_games,
            season_feats=season_feats,
            model=model,
            feature_cols=feature_cols,
            weights=weights,
            season=2026,
            rating_scale=rating_scale,
            shrink=shrink,
            blend_weight=blend_weight,
            seed_alpha=seed_alpha,
            seed_z_threshold=seed_z_threshold,
        )
        return

    if RUN_FIRST_ROUND:
        for yr in [2026]:
            fr = first_round_for_season(
                all_games=all_games,
                season_feats=season_feats,
                season=yr,
                xgb_params=xgb_params,
                rating_scale=rating_scale,
                shrink=shrink,
                blend_weight=blend_weight,
                seed_alpha=seed_alpha,
                seed_z_threshold=seed_z_threshold,
            )
            print(f"\n=== First Round {yr} (trained on < {yr}) ===")
            print(fr.sort_values(["Region", "Matchup"]).to_string(index=False))
        # fall through to allow actual rounds too if enabled

    if RUN_ACTUAL_ROUNDS:
        brier_scores = []
        for yr in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
            ar = list_actual_round_matchups(
                all_games=all_games,
                season_feats=season_feats,
                season=yr,
                xgb_params=xgb_params,
                rating_scale=rating_scale,
                shrink=shrink,
                blend_weight=blend_weight,
                seed_alpha=seed_alpha,
                seed_z_threshold=seed_z_threshold,
            )
            print(f"\n=== Actual Round Matchups {yr} (trained on < {yr}) ===")
            print(ar.to_string(index=False))
            brier_year = float(np.mean((1.0 - ar["PredProb_ActualWinner"].values) ** 2))
            brier_scores.append(brier_year)
            print(f"Brier {yr}: {brier_year:.6f}")
        if brier_scores:
            print(f"\nAverage Brier (2018-2025): {float(np.mean(brier_scores)):.6f}")
        return

    if RUN_SEED_THRESHOLDS:
        thresholds = compute_seed_matchup_thresholds(
            all_games=all_games,
            season_feats=season_feats,
            seasons=[2018, 2019, 2021, 2022, 2023, 2024, 2025],
            xgb_params=xgb_params,
            rating_scale=rating_scale,
            shrink=shrink,
            blend_weight=blend_weight,
            seed_alpha=seed_alpha,
            seed_z_threshold=seed_z_threshold,
            target_win_pct=0.95,
            min_games=10,
        )
        print("\n=== Seed Matchup Thresholds (>=95% win rate) ===")
        print(thresholds.to_string(index=False))
        return

    # Tournament simulation
    if RUN_SIM:
        sim_df = simulate_tournament(
            all_games=all_games,
            season_feats=season_feats,
            model=model,
            feature_cols=feature_cols,
            weights=weights,
            season=2026,
            n_sims=10000,
            rating_scale=rating_scale,
            shrink=shrink,
            blend_weight=blend_weight,
            seed_alpha=seed_alpha,
            seed_z_threshold=seed_z_threshold,
        )
        print(sim_df.to_string(index=False))
        return

    # Predict 2026 matchups
    feats_2026 = season_feats[season_feats["Season"] == 2026].copy()
    prob_matrix_2026 = predict_2026_matchup_matrix_weighted(model, feats_2026, weights)

    print(prob_matrix_2026.head(10))

    # Build 2026 submission files from sample templates
    sample_paths = [
        os.path.join(KAGGLE_DATA_DIR, "SampleSubmissionStage1.csv"),
        os.path.join(KAGGLE_DATA_DIR, "SampleSubmissionStage2.csv"),
    ]
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            out_path = os.path.join(BASE_DIR, f"predictions_{os.path.basename(sample_path)}")
            build_submission_from_sample(
                model=model,
                season_feats_2026=feats_2026,
                feature_cols=feature_cols,
                weights=weights,
                sample_path=sample_path,
                out_path=out_path,
                rating_scale=rating_scale,
                shrink=shrink,
                blend_weight=blend_weight,
            )
            print(f"Wrote submission: {out_path}")

    # Optional: feature importance
    print(coef_importance.head(10))


if __name__ == "__main__":
    main()
