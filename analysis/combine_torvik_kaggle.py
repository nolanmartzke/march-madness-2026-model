import pandas as pd
import re
from rapidfuzz import process, fuzz

# --- Load files ---
teams = pd.read_csv("../data/march-machine-learning-mania-2026/MTeams.csv")                 # Kaggle: TeamID, TeamName
torvik = pd.read_csv("../data/2025_team_results.csv")

teams = teams[teams['LastD1Season'] == 2026]          # Your Torvik file: Team (or TeamName), metrics...

# Adjust column names here:
KAGGLE_NAME_COL = "TeamName"
TORVIK_NAME_COL = "team"   # or whatever your Torvik name column is

# --- Name normalizer ---
def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""

    s = s.lower().strip()

    # replace ampersand
    s = s.replace("&", "and")

    # remove parentheses
    s = re.sub(r"\(.*?\)", "", s)

    # normalize st.
    s = s.replace("st.", "st")

    # ---- CS handling ----
    # cs at beginning -> cal state
    s = re.sub(r"^cs\s+", "cal state ", s)

    # remove punctuation
    s = re.sub(r"[^a-z0-9\s]", "", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # ---- saint/state handling ----

    # st before word -> saint
    s = re.sub(r"\bst\s+(?=[a-z])", "saint ", s)

    # st at end -> state
    s = re.sub(r"\bst$", "state", s)

    return s

teams["name_norm"] = teams[KAGGLE_NAME_COL].map(normalize_name)
torvik["name_norm"] = torvik[TORVIK_NAME_COL].map(normalize_name)

# --- 1) Exact merge on normalized name ---
exact = teams.merge(
    torvik,
    on="name_norm",
    how="left",
    suffixes=("_kaggle", "_torvik")
)

matched_exact = exact[~exact[TORVIK_NAME_COL].isna()].copy()
unmatched = exact[exact[TORVIK_NAME_COL].isna()].copy()

print("Exact matched:", len(matched_exact), "Unmatched:", len(unmatched))

# --- 2) Fuzzy match the unmatched ---
torvik_choices = torvik["name_norm"].tolist()
torvik_lookup = torvik.set_index("name_norm")

def best_fuzzy_match(name_norm: str):
    if not name_norm:
        return (None, 0)
    match, score, _ = process.extractOne(name_norm, torvik_choices, scorer=fuzz.WRatio)
    return (match, score)

unmatched["fuzzy_norm"], unmatched["fuzzy_score"] = zip(*unmatched["name_norm"].map(best_fuzzy_match))

# Choose a threshold: 90 is usually safe; 85 if your data is messy
THRESH = 77
fuzzy_good = unmatched[unmatched["fuzzy_score"] >= THRESH].copy()
fuzzy_bad  = unmatched[unmatched["fuzzy_score"] < THRESH].copy()

# Pull torvik rows for fuzzy_good
fuzzy_good = fuzzy_good.drop(columns=torvik.columns.difference(["name_norm", TORVIK_NAME_COL]), errors="ignore")

# IMPORTANT: drop the existing empty 'team' column so merge doesn't suffix it
fuzzy_good = fuzzy_good.drop(columns=[TORVIK_NAME_COL], errors="ignore")

fuzzy_good = fuzzy_good.merge(
    torvik,
    left_on="fuzzy_norm",
    right_on="name_norm",
    how="left"
)

print("Fuzzy matched (>=THRESH):", len(fuzzy_good))
print("Still unmatched:", len(fuzzy_bad))

# --- 3) Manual overrides for remaining (you will fill this in) ---
# key = kaggle normalized name, value = torvik normalized name
MANUAL = {
    # "mississippi": "ole miss",
    # "miami fl": "miami",
    # "uc santa barbara": "ucsb",
    'iupui': 'iu indy',
    'utrgv': 'ut rio grande valley',
    'ulm': 'louisiana monroe',
    'tam c christi': 'texas aandm corpus chris',
    'fgcu': 'florida gulf coast',
    'pfw': 'purdue fort wayne',
    'wku': 'western kentucky',
    'mtsu': 'middle tennessee',
    'etsu': 'east tennessee state',
}

fuzzy_bad["manual_norm"] = fuzzy_bad["name_norm"].map(MANUAL)

manual_good = fuzzy_bad[~fuzzy_bad["manual_norm"].isna()].copy()

# IMPORTANT: drop existing empty 'team' column before merge
manual_good = manual_good.drop(columns=[TORVIK_NAME_COL], errors="ignore")

manual_good = manual_good.merge(
    torvik,
    left_on="manual_norm",
    right_on="name_norm",
    how="left"
)
final_bad = fuzzy_bad[fuzzy_bad["manual_norm"].isna()].copy()

print(final_bad[[ "TeamID", "TeamName", "name_norm", "fuzzy_norm", "fuzzy_score"]]
      .sort_values("fuzzy_score", ascending=False)
      .head(30))

# --- Combine all matched ---
final_map = pd.concat([matched_exact, fuzzy_good, manual_good], ignore_index=True)

# Keep only the columns we want
final_map = final_map[["TeamID", TORVIK_NAME_COL]]

# Remove duplicates just in case
final_map = final_map.drop_duplicates(subset="TeamID")

# Save
final_map.to_csv("teamid_to_torvik_mapping.csv", index=False)

print("Saved: teamid_to_torvik_mapping.csv")