# Analysis modules

See quick examples below on using the analysis modules (going forward, this should be transformed into a more proper documentation).

To try out these functionalities more interactively, check the jupyter notebooks in `examples` folder (Note: you will need to convert them from .py to .ipynb files using [jupytext](https://github.com/mwouts/jupytext)).

## Contents

- [**Data wrangling**](#data_wrangling)
  - [Research project data](#wrangling_research)
  - [Company data](#wrangling_companies)
- [**Querying data**](#querying)
  - [Search terms](#query_terms)

## Data wrangling<a name="data_wrangling"></a>

Module `wrangling_utils` helps fetching data related to research projects and businesses.

### Research project data<a name="wrangling_research"></a>

```python
from innovation_sweet_spots.getters import gtr
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler

# Prepare a small sample of projects
gtr_projects = gtr.get_gtr_projects().head(3)

# Initiate a data wrangler instance
GtR = GtrWrangler()
```

#### Funding

Apply `get_funding_data()` on a list of projects, to get the awarded funding amounts and the start and end dates of the funding (the first run might take longer as it needs to load in funding data).

```python
GtR.get_funding_data(gtr_projects)
```

#### Research topics

Use `get_research_topics()` to find the research topics assigned to the projects in the GtR database. Note, however, that many projects (approximately half) are `Unclassified`.

```python
GtR.get_research_topics(gtr_projects)
```

To view all existing research topic labels, you can check `GtR.gtr_topics`. You can also retrieve projects belonging to specific GtR research topics by running `get_projects_in_research_topics()`. For example:

```python
GtR.get_projects_in_research_topics(research_topics=['International Business', 'Classical Literature'])
```

#### Organisations

Use `get_organisations_and_locations()` to get organisations that participate in the projects and the organisations' locations.

```python
GtR.get_organisations_and_locations(gtr_projects)
```

Note that that organisations can have different types of roles in a project - this is specified by the `organisation_relation` variable, and the explanations of the different possible roles is provided in the [GtR API documentation](https://gtr.ukri.org/resources/GtR-2-API-v1.7.5.pdf) (see page 9).

Location data is available at the level of continent, country, address as well as latitude and longitude.

Presently, it is not possible to automatically distinguish different types of organisations (eg, public sector, private sector, non-profits etc.) but we are planning to look into this. Also, we are planning to match the organisations in GtR dataset with organisation data in Crunchbase.

#### People

Use `get_persons()` to get information about the people participating in the projects.

```python
GtR.get_persons(gtr_projects)
```

Note that that people can have different types of roles in a project - this is specified by the `person_relation` variable, and the explanations of the different possible roles is provided in the [GtR API documentation](https://gtr.ukri.org/resources/GtR-2-API-v1.7.5.pdf) (see page 9).

---

### Company data<a name="wrangling_companies"></a>

```python
from innovation_sweet_spots.getters import crunchbase as cb
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

# Initiate a data wrangler instance
CB = CrunchbaseWrangler()
```

#### Industries

Crunchbase organises their companies by industries and industry groups. To see all industries you can run

```python
CB.industries
```

To find companies in specific Crunchbase industries, use `get_companies_in_industries()`. Note: this might take a minute when running for the first time. For example:

```python
cb_orgs = CB.get_companies_in_industries(industries_names=["parenting"])
```

A company can be in several industries (eg, "parenting" and "social media"). To check all industries these companies are in, you can run `get_company_industries()`.

```python
CB.get_company_industries(cb_orgs)
```

See the notebooks in the `examples` folder for more examples of selecting companies by industries or industry groups.

#### Investments

To find companies' funding rounds (ie, investment deals) use `get_funding_rounds()`:

```python
CB.get_funding_rounds(cb_orgs)
```

To list the investors who have invested in a specific set of companies, use `get_organisation_investors()`

```python
CB.get_organisation_investors(cb_orgs)
```

#### People

Use `get_company_persons()` to find people associated with specific companies:

```python
cb_org_persons = CB.get_company_persons(cb_orgs)
```

See also notebooks in `examples` for further examples how to get the university education data on Crunchbase.

## Querying data<a name="querying"></a>

### Search terms<a name="query_terms"></a>

`analysis.query_terms` module helps to do a simple search using key words and phrases.

To get started, you need to load a "corpus" with preprocessed documents, which is provided by the `getters.preprocessed` module. Presently, it's possible to load in the preprocessed documents used in the pilot project (NB: Pilot project focussed only on the UK).

```python
from innovation_sweet_spots.getters.preprocessed import (
    get_pilot_gtr_corpus,
    get_pilot_crunchbase_corpus,
)
from innovation_sweet_spots.analysis.query_terms import QueryTerms

# Define search terms
SEARCH_TERMS = [
    ["heat pump"],
]

Query = QueryTerms(corpus=get_pilot_gtr_corpus())
query_df = Query.find_matches(SEARCH_TERMS, return_only_matches=True)
```

The returned dataframe contains a column for each search term, as well as a summary `has_any_terms` column, and the document ids. To view more data on the documents, you can use the `wrangler_utils` module.

```python
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler
GTR = GtrWrangler()
GTR.add_project_data(query_df, id_column="id", columns=["title", "start"])
```
