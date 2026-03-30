import pandas as pd
import glob
import os

TORVIK_MAP_PATH = "../analysis/teamid_to_torvik_mapping.csv"
DATA_PATH = "../data/[0-9]*_team_results.csv"
mapping = pd.read_csv(TORVIK_MAP_PATH)

for file in glob.glob(DATA_PATH):

    df = pd.read_csv(file)

    # fix torvik column shift
    if df["team"].str.contains("-").any():
        df = df.shift(1, axis=1)
        df.columns = [
            "rank","team","conf","record","adjoe","oe_rank","adjde","de_rank",
            "barthag","rank2","proj_w","proj_l","pro_con_w","pro_con_l","con_rec",
            "sos","ncsos","consos","proj_sos","proj_noncon_sos","proj_con_sos",
            "elite_sos","elite_noncon_sos","opp_oe","opp_de","opp_proj_oe",
            "opp_proj_de","con_adj_oe","con_adj_de","qual_o","qual_d",
            "qual_barthag","qual_games","fun","conpf","conpa","conposs",
            "conoe","conde","consosremain","conf_win","wab","wab_rk",
            "adjt","TeamID"
        ]

    # remove existing TeamID if already present
    if "TeamID" in df.columns:
        df = df.drop(columns=["TeamID"])

    # merge IDs
    df = df.merge(
        mapping[["TeamID", "team"]],
        on="team",
        how="left"
    )

    # ignore conference rows (all caps abbreviations)
    missing = df[df["TeamID"].isna()]["team"]
    missing = missing[~missing.str.fullmatch(r"[A-Z0-9]+")].unique()

    if len(missing) > 0:
        print(f"{file} missing IDs:", missing)

    # save new file instead of overwriting
    output = f"../data/updated_{os.path.basename(file)}"
    df.to_csv(output, index=False)

    print("Processed:", file, "->", output)