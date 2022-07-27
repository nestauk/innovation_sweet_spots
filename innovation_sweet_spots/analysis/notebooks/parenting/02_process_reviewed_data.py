# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Processing reviewed company data
#
# This script fetches manually reviewed company data, processes manual comments and returns a table with three columns:
# - **"id"**: Identifiers for the relevant companies (to be included in the further analysis)
# - **"user"**: A tag for the primary user of the company's product or service (ie, `Parents` or `Children`)
# - **"interesting"**: A `True` or `False` tag indicating whether the manual reviewer thought the company was particulary interesting (eg, if it's using emerging technologies or have an interesting business model).

# %%
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd

# Location of the reviewed data tables
INPUTS_PATH = PROJECT_DIR / "outputs/parenting/cb_companies/reviewed"
# Location to export the final table
OUTPUTS_DIR = PROJECT_DIR / "outputs/parenting/cb_companies"

# Custom companies to add
# (these didn't make through in the earlier selection process and were noticed later)
CUSTOM_COMPANIES = {
    "95487399-812c-d898-a435-c9494023cbbc": "Children",
    "58a73d1f-036b-4f21-875d-dfa3f3ef93be": "Parents",
}


def map_child_comments_to_user(txt):
    """Helper function to map custom comments to primary user categories"""
    cats = {
        "Sports": "Children",
        "General": "Children",
        "Learning": "Children",
        "Numerical / coding": "Children",
        "Parental support": "Parents",
        "Child care": "Parents",
        "Numerical / stem": "Children",
        "Learning / Play": "Children",
        "Parental support / Activities": "Parents",
        "Education management": "Parents",
        "Older kids": "Children",
        "Tech": "Children",
        "Child Care ": "Parents",
        "Sharing memories": "Parents",
        "Sharing memories ": "Parents",
        "Parental suport": "Parents",
        "Literacy": "Children",
    }
    if type(txt) is str:
        return cats[txt]
    else:
        return "Children"


def map_parent_comments_to_user(txt):
    """Helper function to map custom comments to primary user categories"""
    cats = {
        "Helping babies to sleep": "Parents",
        "Literacy": "Children",
        "Media": "Children",
        "Reading stories": "Children",
        "Reading stories / Parental support": "Children",
        "Sharing memories": "Parents",
        "Play": "Children",
        "Activities": "Parents",
        "child care": "Parents",
        "Community": "Parents",
        "Educational management": "Parents",
        "Learning": "Children",
        "Parental support": "Parents",
        "Share memories": "Parents",
        "Stories": "Children",
        "Activities / outdoors / Play": "Parents",
        "Child care": "Parents",
        "Education management": "Parents",
        "Educational": "Children",
        "Educational / Education management": "Parents",
        "Educational / platform": "Parents",
        "Educational / special needs": "Children",
        "Learning play": "Children",
        "Parental support / activities": "Parents",
        "Parental support / community": "Parents",
        "Pregancy / health": "Parents",
        "Pregnancy": "Parents",
        "Robots": "Children",
        "Toys / Play": "Children",
        "Finance": "Parents",
        "Kids products / retail": "Parents",
        "Fertility": "Parents",
        "Adoption": "Parents",
        "Educational / health": "Parents",
        "Learning / special needs": "Parents",
        "Learning play / Outdoors": "Children",
        "Parental support ": "Parents",
        "Parental support / co-parenting": "Parents",
        "Parental support / Community": "Parents",
        "Play, activities": "Children",
        "Activities / Play": "Children",
        "Parental support  / Community": "Parents",
        "Parental support / Activities": "Parents",
        "Play / games": "Children",
        "Clothes / Kids products": "Parents",
        "Helping babies sleep": "Parents",
        "Parental support / health": "Parents",
        "Robots / hardware": "Children",
        "Robots / Tracking babies rhythms": "Children",
        "Tracking babies rhythms": "Parents",
        "Parental support / Child care": "Parents",
        "Parental support / Communities": "Parents",
    }
    if type(txt) is str:
        return cats[txt]
    else:
        return "Parents"


# %%
if __name__ == "__main__":

    ## Load reviewed data tables
    # Reviewed parenting companies
    reviewed_df_parenting = pd.read_csv(
        INPUTS_PATH
        / "cb_companies_parenting_v2022_04_27 - cb_companies_parenting_v2022_04_27.csv"
    )
    # Reviewed early years child education companies
    reviewed_df_child_ed = pd.read_csv(
        INPUTS_PATH
        / "cb_companies_child_ed_v2022_04_27 - cb_companies_child_ed_v2022_04_27.csv"
    )
    # Process the tables (select only 'relevant' companies, and map them to the primary users)
    companies_parenting_df = reviewed_df_parenting.query(
        'relevancy == "relevant"'
    ).assign(user=lambda df: df.comment.apply(map_parent_comments_to_user))
    # Mark which companies were tagged as particularly 'interesting' by the manual reviewer
    companies_parenting_df["interesting"] = (
        companies_parenting_df["Unnamed: 16"].str.lower().str.contains("interesting")
    )
    companies_child_ed_df = reviewed_df_child_ed.query(
        'relevancy == "relevant" or comment == "potentially relevant"'
    ).assign(user=lambda df: df.comment.apply(map_child_comments_to_user))
    companies_child_ed_df["interesting"] = (
        companies_child_ed_df.interesting.isnull() == False
    )
    # Company identifiers
    companies_ids = set(companies_parenting_df.id.to_list()).union(
        set(companies_child_ed_df.id.to_list())
    )
    # Add any custom companies
    companies_ids = companies_ids.union(set(list(CUSTOM_COMPANIES.keys())))
    # Mapping from company identifiers to primary users
    id_to_user = (
        pd.concat(
            [
                companies_parenting_df[["id", "user", "interesting"]],
                companies_child_ed_df[["id", "user", "interesting"]],
                pd.DataFrame(
                    data={
                        "id": list(CUSTOM_COMPANIES.keys()),
                        "user": list(CUSTOM_COMPANIES.values()),
                        "interesting": True,
                    }
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates("id")
        .query("id in @companies_ids")
        .fillna({"interesting": False})
        .reset_index(drop=True)
    )

    id_to_user.to_csv(OUTPUTS_DIR / "cb_companies_ids_reviewed.csv", index=False)
