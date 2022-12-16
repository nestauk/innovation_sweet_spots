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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import altair as alt
from vega_datasets import data

source = data.barley.url

alt.Chart(source).mark_point().encode(
    x='site:O',
    y='yield:Q',
    xOffset='year:N',
    color='year:N',
)

# %%
import altair as alt
import pandas as pd

# create dataframe
df = pd.DataFrame([['Action', 5, 'F'], 
                   ['Crime', 10, 'F'], 
                   ['Action', 3, 'M'], 
                   ['Crime', 9, 'M']], 
                  columns=['Genre', 'Rating', 'Gender'])

chart = alt.Chart(df).mark_bar().encode(
   column=alt.Column(
       'Genre', 
       header=alt.Header(orient='bottom')
    ),
   x=alt.X('Gender', axis=alt.Axis(ticks=False, labels=False, title='')),
   y=alt.Y('Rating', axis=alt.Axis(grid=False)),
   color='Gender'
).configure_view(
    stroke=None,
)

chart

# %%
chart = alt.Chart(df).mark_bar().encode(
   x=alt.X('Genre', axis=alt.Axis(labelAngle=0)),
   xOffset=alt.XOffset('Gender:N'),
   y=alt.Y('Rating', axis=alt.Axis(grid=False)),
   color='Gender'
).configure_view(
    stroke=None,
)

chart

# %%
