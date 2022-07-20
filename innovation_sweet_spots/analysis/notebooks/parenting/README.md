# Mapping parenting tech: Venture capital trends

Analysis scripts and notebooks, developed for Nesta's [Mapping parenting technologies project](https://www.nesta.org.uk/project/mapping-parenting-technology/). This analysis re-used code developed for Nesta's [Innovation Sweet Spots](https://www.nesta.org.uk/feature/innovation-sweet-spots/) project, which is why it is located in this repo.

To learn about innovation trends around digital technologies for parenting and early years education (for 0-5 year olds), we used [Crunchbase](https://crunchbase.com/) business information platform to analyse companies and venture capital investments in this space. The final list of the identified early years education and parenting company names can be found [here](https://docs.google.com/spreadsheets/d/1KELT0mLeMC565blGOxHj7soLx4qK6pzdUx7xqjNy29k/edit#gid=2002056421).

Unfotunately, Cruchbase data is proprietary and we cannot share granular, company-level details. Therefore, unless you have access to Crunchbase database snapshots, the code in this folder serves more as a reference for the performed analysis.

_In addition, for this project we also analysed trends around children and parenting apps on the Play Store. See [this repo](https://github.com/nestauk/mapping_parenting_tech) to find the corresponding analysis code and results._

## Contents

- **01_select_companies**: Script that selects potentially relevant companies (the selection criteria are also explained further below). This could be run as a notebook, by converting the .py file into .ipynb using jupytext, or as a script using the command line.
- **02_process_reviewed_data** Script that fetches the manually reviewed company data table and processes manual comments.
- **03_company_analysis**: Notebook for preparing analyses and producing graphs used in the final output.
- **utils**: Helper functions and variables

The _deprecated/_ folder contains notebooks with interim data explorations, and can be avoided.

## Methdology

### Identification of companies by industries and keywords

To identify companies related to parenting and early years education, we used industry labels assigned to companies on Crunchbase as well as their text descriptions. For example, we selected all companies that have the label 'parenting'; we also selected companies that have keywords such as 'preschool' in their description. The full selection criteria are described below.

#### Early years education: Industries

First approach selected companies related to children AND education. Specifically, we used a set of Crunchbase industry categories related to children `CHILDREN_INDUSTRIES` and another set related to education `EDUCATION_INDUSTRIES`.

We also specified another set with education-related industries `INDUSTRIES_TO_REMOVE` that are, however, irrelevant as they desribe later stages of education.

We then looked for companies in `EDUCATION_INDUSTRIES` AND `CHILDREN_INDUSTRIES` AND NOT in `INDUSTRIES_TO_REMOVE`.

The sets are shown below, and they're defined in the `utils.py` helper module.

```
CHILDREN_INDUSTRIES = ["child care", "children", "underserved children", "family", "baby"]

EDUCATION_INDUSTRIES = ["education", "edtech", "e-learning", "edutainment", "language learning", "mooc", "music education", "personal development", "skill assessment", "stem education", "tutoring", "training", "primary education", "continuing education", "charter schools"]

INDUSTRIES_TO_REMOVE = ["secondary education", "higher education", "universities", "vocational education", "corporate training", "college recruiting"]
```

#### Early years education: Keywords

As the second approach, we used a set of keywords (search terms) related to children `CHILDREN_TERMS` and to learning `LEARNING_TERMS`. We looked for companies that have both `CHILDREN_TERMS` AND `LEARNING_TERMS` in their short and long descriptions. Prior to the keyword search, the descriptions were preprocessed by the flow in `pipeline/preprocessing/flow_tokenise_cb.py`. Some of the search terms are stemmed/shortened so that they capture multiple forms of the same word.

```
CHILDREN_TERMS = ["baby", "babies", "infant", "child", "toddler", "kid ", "kids ", "son ", "sons ", "daughter", "boy", "girl"]


LEARNING_TERMS = ['learn', 'educat', 'develop', 'study', 'preschool', 'pre school', 'kindergarten', 'pre k ', 'montessori', 'literacy', ' numeracy', 'math', 'phonics', 'early year']

```

#### Parenting

Another set of companies was selected using a much simpler approach, by choosing all companies that are in the `"parenting"` Crunchbase industry.

#### Manual review

Among the companies selected by the approaches outlined above, we considered those that recorded at least one investment round on Crunchbase. We then manually reviewed the companies to only select those that are likely to be relevant to early years (ie, for 0-5 year olds) and excluded those focussing solely on products and services for older children. If a company had products or services for both younger and older, school-age children, we included them. However, in those cases it is not possible to discern what fraction of the investment is specifically for developing the early years products.

During the manual review, each company was also assigned either a 'Parents' or 'Children' tag to indicate the primary user of the products or services developed by the companies.

The final list of companies considered in this analysis can be found [here](https://docs.google.com/spreadsheets/d/1KELT0mLeMC565blGOxHj7soLx4qK6pzdUx7xqjNy29k/edit#gid=2002056421).

Note that for this analysis, we used a data snapshot last updated in March 2022. If you would repeat this process with more recently updated data then you might get a different (larger) set of companies as the database is continually updated.

### Venture capital analysis

#### Deal types

In the published figures, we considered only early stage investment deals (eg, seed, crowdfunding and series) and excluded late stage deals such as acquisitions, IPO or post-IPO investments. The late stage investments are characteristic to more mature companies and can have very large amounts that would skew the investment trends.

See below the types of deals included in the analysis:

```
"angel",
"convertible_note",
"equity_crowdfunding",
"non_equity_assistance",
"pre_seed",
"product_crowdfunding",
"secondary_market",
"seed",
"series_a",
"series_b",
"series_c",
"series_d",
"series_e",
"series_unknown"
```

#### Digital technologies

For the digital technology analysis, we again used Crunchbase industry labels. We used the broader industry groups specified by Crunchbase, and selected all industries that fall into the industry groups specified below (and defined in `utils.py`). There were in total almost 300 different digital industry groups considered in the analysis.

```
"information technology",
"hardware",
"software",
"mobile",
"consumer electronics",
"music and audio",
"gaming",
"design",
"privacy and security",
"messaging and telecommunications",
"internet services",
"artificial intelligence",
"media and entertainment",
"platforms",
"data and analytics",
"apps",
"video",
"content and publishing",
"advertising"

```

## Caveats

It is important to note that the analysis largely relies on the company labels on Crunchbase, which we do not expect to be fully complete. For the digital technology categories, the growth estimates might also be influenced by popularity and maturity of the technologies. For example, categories like 'internet’ or ‘information technology’ are likely becoming redundant as most companies depend on these technologies and they do not serve as reliable differentiators anymore.

As we were interested in reporting investment trends, we have only included companies that have at least one record of raised investment. Not all investment, however, is recorded on the Crunchbase platform and, therefore, our results
are likely to be underestimates.
