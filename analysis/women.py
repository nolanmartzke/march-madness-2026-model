import os
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# --------------------- PATHS ---------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KAGGLE_DATA_DIR = os.path.join(BASE_DIR, "data", "march-machine-learning-mania-2026")
WREG_PATH = os.path.join(KAGGLE_DATA_DIR, "WRegularSeasonDetailedResults.csv")
WTOURNEY_PATH = os.path.join(KAGGLE_DATA_DIR, "WNCAATourneyDetailedResults.csv")
WSEEDS_PATH = os.path.join(KAGGLE_DATA_DIR, "WNCAATourneySeeds.csv")
WTEAMS_PATH = os.path.join(KAGGLE_DATA_DIR, "WTeams.csv")
WCOMPACT_RESULTS_PATH = os.path.join(KAGGLE_DATA_DIR, "WNCAATourneyCompactResults.csv")


USE_SEED_FEATURES = True
WOMEN_HOME_ADV_BOOST = 0.03  # applied only in R64/R32 when a 1-4 seed hosts


def normalize_team_name(name):
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_team_name_lookup():
    teams = pd.read_csv(WTEAMS_PATH)
    teams["team_norm"] = teams["TeamName"].map(normalize_team_name)
    name_to_id = dict(zip(teams["team_norm"], teams["TeamID"]))
    id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))
    return name_to_id, id_to_name


def build_seed_bracket_w(seeds_df, season):
    seeds_df = seeds_df[seeds_df["Season"] == season].copy()
    bracket = {}
    for _, row in seeds_df.iterrows():
        seed = row["Seed"]
        team_id = int(row["TeamID"])
        if not isinstance(seed, str) or len(seed) < 3:
            continue
        region = seed[0]
        try:
            seed_num = int(seed[1:3])
        except ValueError:
            continue
        bracket.setdefault(region, {}).setdefault(seed_num, []).append(team_id)
    return bracket


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


def apply_home_boost(p, seed1, seed2, round_name):
    if round_name not in {"R64", "R32"}:
        return p
    if seed1 is None or seed2 is None:
        return p
    seed1 = int(seed1)
    seed2 = int(seed2)
    if seed1 == seed2:
        return p
    if seed1 <= 4 and seed1 < seed2:
        return p + WOMEN_HOME_ADV_BOOST * (1.0 - p)
    if seed2 <= 4 and seed2 < seed1:
        return p - WOMEN_HOME_ADV_BOOST * p
    return p


def load_team_schedule_results(team_id, year, details=None, id_to_name=None):
    if details is None:
        reg = pd.read_csv(WREG_PATH)
        tourney = pd.read_csv(WTOURNEY_PATH)
        reg["is_tourney"] = 0
        tourney["is_tourney"] = 1
        details = pd.concat([reg, tourney], ignore_index=True)

    if id_to_name is None:
        teams = pd.read_csv(WTEAMS_PATH)
        id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))
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
    results["opp_team_id"] = np.where(results["Win"] == 1, results["LTeamID"], results["WTeamID"])

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

    # 3PT%
    three_fgm = np.where(results["Win"] == 1, results["WFGM3"], results["LFGM3"]).astype(float)
    three_fga = np.where(results["Win"] == 1, results["WFGA3"], results["LFGA3"]).astype(float)
    results["three_pt_pct"] = np.divide(
        three_fgm,
        three_fga,
        out=np.full_like(three_fgm, np.nan, dtype=float),
        where=three_fga > 0,
    )

    # Opponent 3PT% and 3PT rate (3PA / FGA)
    opp_three_fgm = np.where(results["Win"] == 1, results["LFGM3"], results["WFGM3"]).astype(float)
    opp_three_fga = np.where(results["Win"] == 1, results["LFGA3"], results["WFGA3"]).astype(float)
    opp_fga = np.where(results["Win"] == 1, results["LFGA"], results["WFGA"]).astype(float)
    results["opp_three_pt_pct"] = np.divide(
        opp_three_fgm,
        opp_three_fga,
        out=np.full_like(opp_three_fgm, np.nan, dtype=float),
        where=opp_three_fga > 0,
    )
    results["opp_three_pt_rate"] = np.divide(
        opp_three_fga,
        opp_fga,
        out=np.full_like(opp_three_fga, np.nan, dtype=float),
        where=opp_fga > 0,
    )

    return results[[
        "Win",
        "team",
        "opp_team",
        "opp_team_id",
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
        "is_tourney",
    ]]


def build_team_season_features(years):
    print("Building women team season features...")
    reg = pd.read_csv(WREG_PATH)
    tourney = pd.read_csv(WTOURNEY_PATH)
    reg["is_tourney"] = 0
    tourney["is_tourney"] = 1
    details = pd.concat([reg, tourney], ignore_index=True)
    print(f"Loaded {len(details):,} women game rows")

    teams = pd.read_csv(WTEAMS_PATH)
    id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

    seeds = pd.read_csv(WSEEDS_PATH)
    seeds["seed"] = seeds["Seed"].str[1:3].astype(int)

    if years is not None:
        details = details[details["Season"].isin(years)].copy()
        print(f"Filtered to seasons {min(years)}–{max(years)} -> {len(details):,} rows")

    # Build game-level rows in vectorized form (much faster than per-team loop)
    win_rows = details.copy()
    loss_rows = details.copy()

    win_rows["Win"] = 1
    loss_rows["Win"] = 0

    win_rows["TeamID"] = win_rows["WTeamID"]
    win_rows["opp_team_id"] = win_rows["LTeamID"]
    loss_rows["TeamID"] = loss_rows["LTeamID"]
    loss_rows["opp_team_id"] = loss_rows["WTeamID"]

    win_rows["team"] = win_rows["TeamID"].map(id_to_name)
    loss_rows["team"] = loss_rows["TeamID"].map(id_to_name)
    win_rows["opp_team"] = win_rows["opp_team_id"].map(id_to_name)
    loss_rows["opp_team"] = loss_rows["opp_team_id"].map(id_to_name)

    win_rows["adjOE"] = 100 * win_rows["WScore"] / (
        win_rows["WFGA"] - win_rows["WOR"] + win_rows["WTO"] + 0.475 * win_rows["WFTA"]
    )
    loss_rows["adjOE"] = 100 * loss_rows["LScore"] / (
        loss_rows["LFGA"] - loss_rows["LOR"] + loss_rows["LTO"] + 0.475 * loss_rows["LFTA"]
    )

    win_rows["adjDE"] = 100 * win_rows["LScore"] / (
        win_rows["LFGA"] - win_rows["LOR"] + win_rows["LTO"] + 0.475 * win_rows["LFTA"]
    )
    loss_rows["adjDE"] = 100 * loss_rows["WScore"] / (
        loss_rows["WFGA"] - loss_rows["WOR"] + loss_rows["WTO"] + 0.475 * loss_rows["WFTA"]
    )

    win_rows["TO%"] = 100 * win_rows["WTO"] / (
        win_rows["WFGA"] - win_rows["WOR"] + win_rows["WTO"] + 0.475 * win_rows["WFTA"]
    )
    loss_rows["TO%"] = 100 * loss_rows["LTO"] / (
        loss_rows["LFGA"] - loss_rows["LOR"] + loss_rows["LTO"] + 0.475 * loss_rows["LFTA"]
    )

    win_rows["three_pt_pct"] = np.divide(
        win_rows["WFGM3"].astype(float),
        win_rows["WFGA3"].astype(float),
        out=np.full(len(win_rows), np.nan, dtype=float),
        where=win_rows["WFGA3"] > 0,
    )
    loss_rows["three_pt_pct"] = np.divide(
        loss_rows["LFGM3"].astype(float),
        loss_rows["LFGA3"].astype(float),
        out=np.full(len(loss_rows), np.nan, dtype=float),
        where=loss_rows["LFGA3"] > 0,
    )

    win_rows["opp_three_pt_pct"] = np.divide(
        win_rows["LFGM3"].astype(float),
        win_rows["LFGA3"].astype(float),
        out=np.full(len(win_rows), np.nan, dtype=float),
        where=win_rows["LFGA3"] > 0,
    )
    loss_rows["opp_three_pt_pct"] = np.divide(
        loss_rows["WFGM3"].astype(float),
        loss_rows["WFGA3"].astype(float),
        out=np.full(len(loss_rows), np.nan, dtype=float),
        where=loss_rows["WFGA3"] > 0,
    )

    win_rows["opp_three_pt_rate"] = np.divide(
        win_rows["LFGA3"].astype(float),
        win_rows["LFGA"].astype(float),
        out=np.full(len(win_rows), np.nan, dtype=float),
        where=win_rows["LFGA"] > 0,
    )
    loss_rows["opp_three_pt_rate"] = np.divide(
        loss_rows["WFGA3"].astype(float),
        loss_rows["WFGA"].astype(float),
        out=np.full(len(loss_rows), np.nan, dtype=float),
        where=loss_rows["WFGA"] > 0,
    )

    win_rows["season_TO%"] = win_rows.groupby(["Season", "TeamID"])["TO%"].transform("mean")
    loss_rows["season_TO%"] = loss_rows.groupby(["Season", "TeamID"])["TO%"].transform("mean")

    keep_cols = [
        "Season",
        "TeamID",
        "opp_team_id",
        "team",
        "opp_team",
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
        "is_tourney",
        "Win",
    ]
    all_games = pd.concat([win_rows[keep_cols], loss_rows[keep_cols]], ignore_index=True)
    print(f"Built all_games: {len(all_games):,} rows")

    season_feats = all_games.groupby(["Season", "TeamID"]).agg(
        adjOE=("adjOE", "mean"),
        adjDE=("adjDE", "mean"),
        to_pct=("TO%", "mean"),
        three_pt_pct=("three_pt_pct", "mean"),
        three_pt_pct_std=("three_pt_pct", "std"),
        opp_three_pt_pct=("opp_three_pt_pct", "mean"),
        opp_three_pt_rate=("opp_three_pt_rate", "mean"),
        season_to_pct=("season_TO%", "mean"),
    )
    season_feats = season_feats.reset_index()
    print(f"Built season_feats: {len(season_feats):,} rows")
    season_feats = season_feats.merge(
        seeds[["Season", "TeamID", "seed"]],
        on=["Season", "TeamID"],
        how="left"
    )
    season_feats["seed"] = season_feats["seed"].fillna(20)
    season_feats["net_rating"] = season_feats["adjOE"] - season_feats["adjDE"]
    season_feats["three_pt_pct"] = season_feats["three_pt_pct"].fillna(0)
    season_feats["three_pt_pct_std"] = season_feats["three_pt_pct_std"].fillna(0)
    season_feats["opp_three_pt_pct"] = season_feats["opp_three_pt_pct"].fillna(0)
    season_feats["opp_three_pt_rate"] = season_feats["opp_three_pt_rate"].fillna(0)

    return all_games, season_feats


def feature_importance_regression(all_games, season_feats):
    feats = season_feats.set_index(["Season", "TeamID"])
    g = all_games.copy()

    feats_team = feats.add_suffix("_team")
    feats_opp = feats.add_suffix("_opp")

    g = g.merge(feats_team, left_on=["Season", "TeamID"], right_index=True, how="left")
    g = g.merge(feats_opp, left_on=["Season", "opp_team_id"], right_index=True, how="left")

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


def build_matchup_training_weighted(all_games, season_feats, coef_importance):
    feats = season_feats.set_index(["Season", "TeamID"])
    g = all_games.copy()

    feats_team = feats.add_suffix("_team")
    feats_opp = feats.add_suffix("_opp")

    g = g.merge(feats_team, left_on=["Season", "TeamID"], right_index=True, how="left")
    g = g.merge(feats_opp, left_on=["Season", "opp_team_id"], right_index=True, how="left")

    feature_cols = [c for c in feats.columns]
    X_team = g[[f"{c}_team" for c in feature_cols]].values
    X_opp = g[[f"{c}_opp" for c in feature_cols]].values

    seed_diff = (g["seed_team"] - g["seed_opp"]).to_numpy().reshape(-1, 1)
    if not USE_SEED_FEATURES:
        seed_diff = np.zeros_like(seed_diff)

    X = np.hstack([X_team - X_opp, seed_diff])
    y = g["Win"].values

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

    game_weights = np.where(is_tourney == 1, 2.0, 1.0)

    base_weights = np.abs(coef_importance.reindex([f"{c}_diff" for c in feature_cols])).fillna(1.0).values
    extra_weights = np.ones(1)
    weights = np.concatenate([base_weights, extra_weights])
    X_weighted = X * weights

    return X_weighted, y, feature_cols, weights, game_weights


def train_xgboost_model(X_weighted, y, game_weights, params=None):
    if params is None:
        params = {}
    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 600),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.035),
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


def build_submission_from_sample(
    model,
    season_feats_2026,
    feature_cols,
    weights,
    sample_path,
    out_path,
):
    sample = pd.read_csv(sample_path)
    ids = sample["ID"].str.split("_", expand=True)
    season = ids[0].astype(int)
    team1 = ids[1].astype(int)
    team2 = ids[2].astype(int)

    feats = season_feats_2026.set_index("TeamID")
    pred = np.full(len(sample), 0.5, dtype=float)

    seeds = pd.read_csv(WSEEDS_PATH)
    seeds["seed_num"] = seeds["Seed"].str[1:3].astype(int)
    seeds_map = (
        seeds[["Season", "TeamID", "Seed", "seed_num"]]
        .set_index(["Season", "TeamID"])
        .to_dict("index")
    )

    def seed_info(season_val, team_id):
        info = seeds_map.get((int(season_val), int(team_id)))
        if not info:
            return None, None
        return info["Seed"], info["seed_num"]

    def matchup_round(seed_a, seed_b):
        if seed_a is None or seed_b is None:
            return None
        a = int(seed_a)
        b = int(seed_b)
        pair = (min(a, b), max(a, b))
        r64_pairs = {(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)}
        if pair in r64_pairs:
            return "R64"
        r32_sets = [
            ({1}, {8, 9}),
            ({4}, {5, 12, 13}),
            ({3}, {6, 11, 14}),
            ({2}, {7, 10, 15}),
        ]
        for hi_set, lo_set in r32_sets:
            if (a in hi_set and b in lo_set) or (b in hi_set and a in lo_set):
                return "R32"
        return None

    def apply_home_boost(p, is_home):
        if WOMEN_HOME_ADV_BOOST <= 0:
            return p
        if is_home:
            return p + WOMEN_HOME_ADV_BOOST * (1.0 - p)
        return p - WOMEN_HOME_ADV_BOOST * p

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
        X = np.hstack([x_team - x_opp, seed_diff])
        X_weighted = X * weights
        probs_raw = model.predict_proba(X_weighted)[:, 1]
        probs = np.clip(probs_raw, 0.01, 0.99)

        # Apply home-court boost for 1-4 seeds in R64/R32 within same region
        for i, idx in enumerate(np.where(known)[0]):
            season_val = season.iloc[idx]
            t1 = team1.iloc[idx]
            t2 = team2.iloc[idx]
            seed_str_1, seed_num_1 = seed_info(season_val, t1)
            seed_str_2, seed_num_2 = seed_info(season_val, t2)
            if seed_str_1 is None or seed_str_2 is None:
                continue
            if seed_str_1[0] != seed_str_2[0]:
                continue
            rnd = matchup_round(seed_num_1, seed_num_2)
            if rnd not in {"R64", "R32"}:
                continue
            if seed_num_1 <= 4 or seed_num_2 <= 4:
                # boost the higher home seed (1-4) if present
                if seed_num_1 <= 4 and seed_num_1 < seed_num_2:
                    probs[i] = apply_home_boost(probs[i], True)
                elif seed_num_2 <= 4 and seed_num_2 < seed_num_1:
                    probs[i] = apply_home_boost(probs[i], False)
                # if both <=4 (rare), no boost
            probs[i] = float(np.clip(probs[i], 0.01, 0.99))

        pred[known] = probs

    out_df = pd.DataFrame({"ID": sample["ID"], "Pred": pred})
    out_df.to_csv(out_path, index=False)
    return out_path


def merge_men_women_predictions(sample_path, men_pred_path, women_pred_path, out_path):
    sample = pd.read_csv(sample_path)
    men = pd.read_csv(men_pred_path)
    women = pd.read_csv(women_pred_path)
    wteams = pd.read_csv(WTEAMS_PATH)
    w_ids = set(wteams["TeamID"])

    ids = sample["ID"].str.split("_", expand=True)
    team1 = ids[1].astype(int)
    team2 = ids[2].astype(int)
    is_women = team1.isin(w_ids) | team2.isin(w_ids)

    merged = sample.merge(men, on="ID", how="left", suffixes=("", "_men"))
    merged = merged.merge(women, on="ID", how="left", suffixes=("", "_women"))

    pred = merged["Pred_men"]
    # Override with women prediction only for women matchups
    pred = pred.where(~is_women, merged["Pred_women"])

    out_df = pd.DataFrame({"ID": merged["ID"], "Pred": pred.fillna(0.5)})
    out_df.to_csv(out_path, index=False)
    return out_path


def list_actual_round_matchups_w(
    all_games,
    season_feats,
    season,
    xgb_params,
):
    results = pd.read_csv(WCOMPACT_RESULTS_PATH)
    results = results[results["Season"] == season].copy()

    train_games = all_games[all_games["Season"] < season].copy()
    train_feats = season_feats[season_feats["Season"] < season].copy()
    coef_importance = feature_importance_regression(train_games, train_feats)
    X_train, y_train, feature_cols, weights, game_weights = build_matchup_training_weighted(
        train_games, train_feats, coef_importance
    )
    model = train_xgboost_model(X_train, y_train, game_weights=game_weights, params=xgb_params)

    seeds_df = pd.read_csv(WSEEDS_PATH)
    bracket = build_seed_bracket_w(seeds_df, season)

    feats_season = season_feats[season_feats["Season"] == season].copy()
    feats_season = feats_season.drop_duplicates(subset=["TeamID"])
    feature_cols_sim = feature_cols[: (len(weights) - 1)]
    team_ids = sorted({tid for region in bracket.values() for seeds in region.values() for tid in seeds})

    feats = feats_season.set_index("TeamID")
    id_to_name = build_team_name_lookup()[1]

    def final_prob(t1, t2, round_name=None):
        x_team = feats.loc[t1, feature_cols].values
        x_opp = feats.loc[t2, feature_cols].values
        seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
        if not USE_SEED_FEATURES:
            seed_diff = 0
        X = np.append(x_team - x_opp, [seed_diff]).reshape(1, -1)
        X_weighted = X * weights
        prob = float(model.predict_proba(X_weighted)[:, 1][0])
        if round_name is not None:
            seed1 = feats.loc[t1, "seed"]
            seed2 = feats.loc[t2, "seed"]
            prob = apply_home_boost(prob, seed1, seed2, round_name)
        return float(np.clip(prob, 0.01, 0.99))

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
                winner = actual_winner(t1, t2) or t1
                resolved[seed_num] = winner
        return resolved

    round1_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    rows = []

    for region in ["W", "X", "Y", "Z"]:
        region_seeds = resolve_play_ins(bracket.get(region, {}))
        r64 = [(region_seeds[a], region_seeds[b]) for a, b in round1_pairs]
        r32_winners = []
        for t1, t2 in r64:
            winner = actual_winner(t1, t2) or t1
            prob_winner = final_prob(winner, t2 if winner == t1 else t1, "R64")
            rows.append({
                "Season": season,
                "Round": "R64",
                "TeamA": id_to_name.get(t1, str(t1)),
                "TeamB": id_to_name.get(t2, str(t2)),
                "ActualWinner": id_to_name.get(winner, str(winner)),
                "PredProb_ActualWinner": prob_winner,
            })
            r32_winners.append(winner)

        s16_matchups = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 8, 2)]
        s16_winners = []
        for t1, t2 in s16_matchups:
            winner = actual_winner(t1, t2) or t1
            prob_winner = final_prob(winner, t2 if winner == t1 else t1, "R32")
            rows.append({
                "Season": season,
                "Round": "R32",
                "TeamA": id_to_name.get(t1, str(t1)),
                "TeamB": id_to_name.get(t2, str(t2)),
                "ActualWinner": id_to_name.get(winner, str(winner)),
                "PredProb_ActualWinner": prob_winner,
            })
            s16_winners.append(winner)

        e8_matchups = [(s16_winners[0], s16_winners[1]), (s16_winners[2], s16_winners[3])]
        e8_winners = []
        for t1, t2 in e8_matchups:
            winner = actual_winner(t1, t2) or t1
            prob_winner = final_prob(winner, t2 if winner == t1 else t1, "S16")
            rows.append({
                "Season": season,
                "Round": "S16",
                "TeamA": id_to_name.get(t1, str(t1)),
                "TeamB": id_to_name.get(t2, str(t2)),
                "ActualWinner": id_to_name.get(winner, str(winner)),
                "PredProb_ActualWinner": prob_winner,
            })
            e8_winners.append(winner)

        f4_matchup = (e8_winners[0], e8_winners[1])
        winner = actual_winner(*f4_matchup) or f4_matchup[0]
        prob_winner = final_prob(winner, f4_matchup[1] if winner == f4_matchup[0] else f4_matchup[0], "E8")
        rows.append({
            "Season": season,
            "Round": "E8",
            "TeamA": id_to_name.get(f4_matchup[0], str(f4_matchup[0])),
            "TeamB": id_to_name.get(f4_matchup[1], str(f4_matchup[1])),
            "ActualWinner": id_to_name.get(winner, str(winner)),
            "PredProb_ActualWinner": prob_winner,
        })

    return pd.DataFrame(rows)


def simulate_women_tournament(
    all_games,
    season_feats,
    model,
    feature_cols,
    weights,
    season=2026,
    n_sims=1000,
):
    seeds_df = pd.read_csv(WSEEDS_PATH)
    bracket = build_seed_bracket_w(seeds_df, season)

    feats_season = season_feats[season_feats["Season"] == season].copy()
    feats_season = feats_season.drop_duplicates(subset=["TeamID"])
    feats = feats_season.set_index("TeamID")
    team_ids = sorted({tid for region in bracket.values() for seeds in region.values() for tid in seeds})

    def final_prob(t1, t2, round_name=None):
        x_team = feats.loc[t1, feature_cols].values
        x_opp = feats.loc[t2, feature_cols].values
        seed_diff = feats.loc[t1, "seed"] - feats.loc[t2, "seed"]
        if not USE_SEED_FEATURES:
            seed_diff = 0
        X = np.append(x_team - x_opp, [seed_diff]).reshape(1, -1)
        X_weighted = X * weights
        prob = float(model.predict_proba(X_weighted)[:, 1][0])
        if round_name is not None:
            seed1 = feats.loc[t1, "seed"]
            seed2 = feats.loc[t2, "seed"]
            prob = apply_home_boost(prob, seed1, seed2, round_name)
        return float(np.clip(prob, 0.01, 0.99))

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
            p = final_prob(t1, t2, round_name)
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

        f4_left = play_round([(region_winners["W"], region_winners["X"])], "F4")[0]
        add_count(f4_left, "F2")
        f4_right = play_round([(region_winners["Y"], region_winners["Z"])], "F4")[0]
        add_count(f4_right, "F2")

        champ = play_round([(f4_left, f4_right)], "F2")[0]
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


def main():
    print("Starting women pipeline...")
    years = list(range(2014, 2027))
    all_games, season_feats = build_team_season_features(years)

    RUN_WOMEN_SIM = True
    RUN_WOMEN_ACTUAL_ROUNDS = False

    print("Training women model...")

    train_games = all_games[all_games["Season"] <= 2025].copy()
    train_feats = season_feats[season_feats["Season"] <= 2025].copy()

    coef_importance = feature_importance_regression(train_games, train_feats)
    X_weighted, y, feature_cols, weights, game_weights = build_matchup_training_weighted(
        train_games, train_feats, coef_importance
    )
    model = train_xgboost_model(X_weighted, y, game_weights=game_weights)
    print("Model trained.")

    if RUN_WOMEN_ACTUAL_ROUNDS:
        for yr in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
            ar = list_actual_round_matchups_w(
                all_games=all_games,
                season_feats=season_feats,
                season=yr,
                xgb_params=None,
            )
            print(f"\n=== Women Actual Round Matchups {yr} (trained on < {yr}) ===")
            print(ar.to_string(index=False))
        return

    if RUN_WOMEN_SIM:
        print("Running women tournament simulation...")
        sim_df = simulate_women_tournament(
            all_games=all_games,
            season_feats=season_feats,
            model=model,
            feature_cols=feature_cols,
            weights=weights,
            season=2026,
            n_sims=10000,
        )
        print(sim_df.to_string(index=False))
        return

    print("Generating women submissions...")
    feats_2026 = season_feats[season_feats["Season"] == 2026].copy()
    sample_paths = [
        os.path.join(KAGGLE_DATA_DIR, "SampleSubmissionStage1.csv"),
        os.path.join(KAGGLE_DATA_DIR, "SampleSubmissionStage2.csv"),
    ]
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            out_path = os.path.join(BASE_DIR, f"women_{os.path.basename(sample_path)}")
            build_submission_from_sample(
                model=model,
                season_feats_2026=feats_2026,
                feature_cols=feature_cols,
                weights=weights,
                sample_path=sample_path,
                out_path=out_path,
            )
            print(f"Wrote women preds: {out_path}")

    # Merge into final submission files if men predictions already exist
    for sample_path in sample_paths:
        if not os.path.exists(sample_path):
            continue
        women_pred_path = os.path.join(BASE_DIR, f"women_{os.path.basename(sample_path)}")
        men_pred_path = os.path.join(BASE_DIR, f"predictions_{os.path.basename(sample_path)}")
        if not os.path.exists(women_pred_path):
            continue
        if not os.path.exists(men_pred_path):
            print(f"Skipping merge; missing men preds: {men_pred_path}")
            continue
        out_path = men_pred_path
        merge_men_women_predictions(
            sample_path=sample_path,
            men_pred_path=men_pred_path,
            women_pred_path=women_pred_path,
            out_path=out_path,
        )
        print(f"Wrote combined submission: {out_path}")


if __name__ == "__main__":
    main()
