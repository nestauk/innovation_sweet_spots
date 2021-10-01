import pandas as pd


def find_hit_column(df):
    for col in df.columns:
        if "hit" in col.lower():
            return col


def get_verified_docs(sheet_names, fpath=REVIEWED_DOCS_PATH):
    dfs = pd.DataFrame()
    for SHEET_NAME in sheet_names:
        df = pd.read_excel(fpath, sheet_name=SHEET_NAME)
        hit_column = find_hit_column(df)
        df = df[df[hit_column] != 0]
        df = df[COLS]
        df["tech_category"] = SHEET_NAME
        dfs = dfs.append(df, ignore_index=True)
    return dfs
