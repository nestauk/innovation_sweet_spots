# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %%
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup, NavigableString, Tag
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
from innovation_sweet_spots.utils.google_sheets import upload_to_google_sheet
import utils


def extract_website_link(html_element: Tag) -> str:
    """Extract the website link from the html element"""
    website_link = html_element.find("a", href=True)
    if website_link:
        # Replace `////` with http:// and return the string
        if website_link["href"].startswith("////"):
            return website_link["href"].replace("////", "https://")
        else:
            return website_link["href"]
    else:
        return None


def extract_image_link(html_element: Tag) -> str:
    """Extract the image link from the html element"""
    image_link = html_element.find("img", src=True)
    if image_link:
        return image_link["src"]
    else:
        return None


# %%
# Import a website text (obtained using 'Inspect Element')
with open(
    PROJECT_DIR / "inputs/data/misc/2023_childcare/holoniq_taxonomy.txt", "r"
) as f:
    webpage = f.read()

# Pass a string with html text into beautiful soup
page_soup = BeautifulSoup(webpage, "html.parser")

# %%
rows = page_soup.find_all("div", {"class": "row"})

h2_tags = []
h3_tags = []
websites = []
image_links = []

for row in rows:
    # Get all children h2 tags
    h2_tags.append([h2.text for h2 in row.find_all("h2", {"id": "install"})])
    # Get all children col tags
    company_cols = row.find_all("div", {"class": "col-sm-2"})
    # Get all children h3 tags
    h3_tags.append([[h3.text for h3 in col.find_all("h3")] for col in company_cols])
    # Get all children website links
    websites.append([extract_website_link(col) for col in company_cols])
    #  Get all children image links
    image_links.append([extract_image_link(col) for col in company_cols])

# %%
# Create a pandas dataframe that contains the data and explode the lists, and then drop the empty rows
df = (
    pd.DataFrame(
        {
            "h3_tags": h3_tags,
            "h2_tags": h2_tags,
            "websites": websites,
            "image_links": image_links,
            "no": range(len(h3_tags)),
        }
    )
    .explode(["h3_tags", "websites", "image_links"])
    .dropna(subset=["h3_tags"])
    # Format lists into strings
    .assign(
        h3_tags=lambda x: x["h3_tags"].apply(lambda x: str(x[0])),
        h2_tags=lambda x: x["h2_tags"].apply(lambda x: str(x[0])),
        websites=lambda x: x["websites"].astype(str),
        image_links=lambda x: x["image_links"].astype(str),
    )
    # Remove rows that contain the string 'Add to HolonIQ'
    .loc[lambda x: ~x["h3_tags"].str.contains("Add to HolonIQ")]
    .sort_values("no")
    .drop(columns=["no"])
    # Rename columns
    .rename(columns={"h3_tags": "company_name", "h2_tags": "category"})
    # Reorder columns
    .reindex(columns=["category", "company_name", "websites", "image_links"])
    .reset_index(drop=True)
)


# %%
df

# %%
upload_to_google_sheet(
    df, google_sheet_id=utils.GOOGLE_SHEET_ID, wks_name=utils.GOOGLE_SHEET_TAB
)

# %%
df.to_csv(
    PROJECT_DIR / "inputs/data/misc/2023_childcare/holoniq_taxonomy.csv", index=False
)

# %%
