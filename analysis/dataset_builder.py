import re
import time
import random
import requests
import pandas as pd
from io import StringIO

data_2014 =pd.read_csv('../data/2014_team_results.csv')
data_2015 =pd.read_csv('../data/2015_team_results.csv')
data_2016 =pd.read_csv('../data/2016_team_results.csv')
data_2017 =pd.read_csv('../data/2017_team_results.csv')
data_2018 =pd.read_csv('../data/2018_team_results.csv')
data_2019 =pd.read_csv('../data/2019_team_results.csv')
data_2021 =pd.read_csv('../data/2021_team_results.csv')
data_2022 =pd.read_csv('../data/2022_team_results.csv')
data_2023 =pd.read_csv('../data/2023_team_results.csv')
data_2024 =pd.read_csv('../data/2024_team_results.csv')
data_2025 =pd.read_csv('../data/2025_team_results.csv')
data_2026 =pd.read_csv('../data/2026_team_results.csv')

START_YEAR = 2014
END_YEAR = 2025  # change as needed (2020 will be blank)

TORVIK_OVERRIDES = {
    "NC State": "N.C. State",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "UConn": "Connecticut",
    "St. Mary's": "Saint Mary's",
    "St. Joseph's": "Saint Joseph's",
    "St. Peter's": "Saint Peter's",
    "Mt. St. Mary's": "Mount St. Mary's",
    "Louisiana-Lafayette": "Louisiana",
}

def to_torvik_name(x: str) -> str:
    s = str(x).strip()

    # strip refs + records
    s = re.sub(r"\[[^\]]*\]", "", s)            # [1]
    s = re.sub(r"\s+\d+–\d+.*$", "", s)         # "32–2 ..."
    s = s.replace("\u2013", "-").strip()        # en-dash

    # common name standardization
    if s in TORVIK_OVERRIDES:
        return TORVIK_OVERRIDES[s]

    s = re.sub(r"^St\.\s+", "Saint ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch_html(url: str, tries: int = 5, timeout: int = 25) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    last_err = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(1.2 + attempt + random.random())
    raise RuntimeError(f"Failed to fetch {url}. Last error: {last_err}")

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handles MultiIndex columns from read_html
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([str(p) for p in col if str(p) != "nan"]).strip() for col in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df

def extract_seeds_from_wikipedia(year: int) -> pd.DataFrame:
    url = f"https://en.wikipedia.org/wiki/{year}_NCAA_Division_I_men%27s_basketball_tournament"
    html = fetch_html(url)

    tables = pd.read_html(StringIO(html))
    rows = []

    bad = {"East Region", "West Region", "South Region", "Midwest Region"}

    for raw in tables:
        df = flatten_columns(raw)
        cols_lower = [c.lower() for c in df.columns]

        # identify seed column (sometimes "Seed", sometimes "Seed*" etc.)
        seed_cols = [df.columns[i] for i, c in enumerate(cols_lower) if c.startswith("seed")]
        if not seed_cols:
            continue
        seed_col = seed_cols[0]

        # identify team/school column
        team_cols = [df.columns[i] for i, c in enumerate(cols_lower) if ("school" in c or "team" in c)]
        if not team_cols:
            continue
        team_col = team_cols[0]

        tmp = df[[seed_col, team_col]].dropna()

        for seed_val, team_val in tmp.itertuples(index=False):
            team = to_torvik_name(team_val)
            if not team or team in bad:
                continue

            # seed might be "1", "1*", "16a", etc. -> keep numeric part
            m = re.search(r"\d+", str(seed_val))
            if not m:
                continue
            seed = int(m.group())

            rows.append((team, seed))

    seed_df = pd.DataFrame(rows, columns=["team", "seed"])

    # Each team should have exactly one seed; if duplicates exist, keep the most common
    seed_df = (
        seed_df.groupby("team")["seed"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )

    # sanity check
    if len(seed_df) < 60:
        raise RuntimeError(f"Only found {len(seed_df)} seeded teams for {year}")

    return seed_df

seeds_by_year = {}

for year in range(START_YEAR, END_YEAR + 1):
    if year == 2020:
        seeds_by_year[year] = pd.DataFrame(columns=["team", "seed"])
        continue

    seed_df = extract_seeds_from_wikipedia(year)
    seeds_by_year[year] = seed_df

    seed_df.to_csv(f"../data/seeds_{year}.csv", index=False)

def add_seed_and_filter(df: pd.DataFrame, year: int) -> pd.DataFrame:
    seed_df = seeds_by_year[year]

    out = df.merge(seed_df, on="team", how="inner")  # inner = keep only tourney teams
    # Optional: ensure seed is int
    out["seed"] = out["seed"].astype(int)
    return out

data_2014 = add_seed_and_filter(data_2014, 2014)
data_2015 = add_seed_and_filter(data_2015, 2015)
data_2016 = add_seed_and_filter(data_2016, 2016)
data_2017 = add_seed_and_filter(data_2017, 2017)
data_2018 = add_seed_and_filter(data_2018, 2018)
data_2019 = add_seed_and_filter(data_2019, 2019)
data_2021 = add_seed_and_filter(data_2021, 2021)
data_2022 = add_seed_and_filter(data_2022, 2022)
data_2023 = add_seed_and_filter(data_2023, 2023)
data_2024 = add_seed_and_filter(data_2024, 2024)
data_2025 = add_seed_and_filter(data_2025, 2025)

data_2014.to_csv('../data/2014_team_results_with_seeds.csv', index=False)

# Add year column to each dataset
data_2014["year"] = 2014
data_2015["year"] = 2015
data_2016["year"] = 2016
data_2017["year"] = 2017
data_2018["year"] = 2018
data_2019["year"] = 2019
data_2021["year"] = 2021
data_2022["year"] = 2022
data_2023["year"] = 2023
data_2024["year"] = 2024
data_2025["year"] = 2025

# Combine all years into one dataset with all statistics + seeds
combined = pd.concat([data_2014, data_2015, data_2016, data_2017, data_2018, data_2019, data_2021, data_2022, data_2023, data_2024, data_2025], ignore_index=True)
combined.to_csv('../data/combined_with_seeds.csv', index=False)

# Example filter:
# df_year = df[df["Team"].isin(tourney_teams_by_year[2014])]