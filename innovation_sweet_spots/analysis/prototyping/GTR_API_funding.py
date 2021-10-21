# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import requests
from time import sleep
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots import PROJECT_DIR
from tqdm.notebook import tqdm
import csv
import pandas as pd


# %%
def get_fund_from_api_response(r):
    if r.json()["totalSize"] != 0:
        return r.json()["fund"][0]["valuePounds"]["amount"]
    else:
        return 0


# %%
# Import GTR data
gtr_projects = gtr.get_gtr_projects()

# %%
# check i
j = len(
    pd.read_csv(PROJECT_DIR / "outputs/GTR_funds.csv", names=["i", "doc_id", "amount"])
)
selected_docs = gtr_projects.project_id.iloc[j:].to_list()
print(j)

# %%
# open the file in the write mode
with open(PROJECT_DIR / "outputs/GTR_funds.csv", "a") as f:
    writer = csv.writer(f)
    all_responses = []
    for i, doc_id in tqdm(enumerate(selected_docs), total=len(selected_docs)):
        i_ = i + j
        r = requests.get(
            f"https://gtr.ukri.org/gtr/api/projects/{doc_id}/funds",
            headers={"Accept": "application/vnd.rcuk.gtr.json-v7"},
        )
        fund_amount = get_fund_from_api_response(r)
        row = [i_, doc_id, fund_amount]
        all_responses.append(row)
        writer.writerow(row)
        sleep(0.05)

# %%
df_funding = pd.read_csv(
    PROJECT_DIR / "outputs/GTR_funds.csv", names=["i", "doc_id", "amount"]
)

# %%
gtr_projects.merge(df_funding, left_on="project_id", right_on="doc_id", how="left")
