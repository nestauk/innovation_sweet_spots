# Analysis modules

See examples below on using the analysis modules (going forward, this should be transformed into a proper documentation)

## Data wrangling

Module `wrangling_utils` helps fetching and linking data related to research projects and businesses.

### Research projects

```python
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler

# Prepare a small sample of projects
gtr_projects = gtr.get_gtr_projects().head(3)

# Initiate a data wrangler instance
GtR = GtrWrangler()
```

Adding data on funding amounts and the start and end dates of funding (the first run might take longer as it needs to load in funding data)

```python
GtR.get_funding_data(gtr_projects)
```

Add data on organisations that participate in the projects and the organisation locations

```python

GtR.get_organisations_and_locations(gtr_projects)
```
