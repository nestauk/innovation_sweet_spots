# :satellite: Innovation Sweet Spots

**_Open-source code for data-driven horizon scanning_**

## :wave: Welcome!

Innovation Sweet Spots is an experimental, data-driven horizon scanning project, led by Nesta's [Discovery Hub](https://www.nesta.org.uk/project/discovery-hub/). Read more about our motivation on [Medium](https://medium.com/@nesta_uk/in-search-of-innovation-sweet-spots-can-data-science-help-us-see-through-tech-hype-1f140f50c18b), and check out our [first report on green technologies](https://www.nesta.org.uk/data-visualisation-and-interactive/innovation-sweet-spots/). The project code has also been used to analyse [parenting technologies](https://www.nesta.org.uk/project/mapping-parenting-technology/whats-next-for-parenting-tech/), and we're soon publishing a new report on [food tech and innovation](https://www.nesta.org.uk/project/innovation-sweet-spots-food-innovation-obesity-and-food-environments/).

We are building upon Nesta's [Data Analytics Practice](https://www.nesta.org.uk/project/data-analytics/) expertise and previous work on [innovation mapping](https://www.nesta.org.uk/feature/innovation-methods/innovation-mapping/), leveraging data science and machine-learning methods to track the trajectory of innovations and technologies for social good.

By combining insights across several large [datasets](#datasets) that are commonly only analysed in isolation, we paint a multi-dimensional picture of the innovations indicating the resources they are attracting and how they are perceived.

_NB: This codebase and documentation is still under development, where some of the code is still living in Jupyter notebooks whereas some utilities have already been neatly factored out into modules. Please [contact us](mailto:karlis.kanders@nesta.org.uk) if you're interested in re-using parts of the codebase, and we'll be happy to help._

## :hammer_and_wrench: Installation (simple instructions for the tutorial)
Clone this branch of the repo
```
git clone --branch discourse_tutorial_blog https://github.com/nestauk/innovation_sweet_spots.git
cd innovation_sweet_spots
```

Set up a conda environment
```
conda create --name tutorial python=3.9
conda activate tutorial
```

Install the required packages
```
pip install -r requirements.txt
pip install -e .
```

Open the notebook `innovation_sweet_spots/analysis/examples/tutorials/Data_driven_discourse_analysis.ipynb` in your favourite development environment.

## :floppy_disk: Datasets

To uncover research, investment and public discourse trends, we are presently using the following data:

- **[Gateway to Research (GtR)](https://gtr.ukri.org/)**: Research projects funded by UKRI
- **[Crunchbase](https://crunchbase.com/)** or **[Dealroom](https://dealroom.co)**: Venture capital investment databases
- **[The Guardian news](https://open-platform.theguardian.com/)**: to the best of our knowledge, the only major UK newspaper to make its text freely available for research.
- **[Hansard](https://zenodo.org/record/4066772#.YXCN1kbYrlw)**: Records of parliamentary debates

All these datasets except Crunchbase and Dealroom are freely available. Note, however, that this project accesses some of these large datasets (namely GtR and Crunchbase) via our internal Nesta database and as such are intended for internal use.

In the future, we might add other datasets to our approach.

<details>
  <summary>Click to read data access guidelines</summary>

<br>

_NB: This information is slightly out of date and will be updated soon_

### Research project and company data

To download GtR and Crunchbase datasets from Nesta database, you will first need to decrypt the config files (if you don't have the key, reach out to Karlis).

```shell
$ git stash
$ git-crypt unlock /path/to/key
```

The most recent version of the Gateway to Research (GtR) and Crunchbase datasets can then be fetched by running the command below. Note that you need to be connected via Nesta's VPN when accessing the database.

```shell
$ python innovation_sweet_spots/pipeline/fetch_daps1_data/flow.py --no-pylint --environment=conda run
```

### The Guardian news

We are using Guardian API to search for articles with specific key terms. For accessing the API, you you'll need to proceed as follows:

- Request an API key from Guardian website ([see here](https://open-platform.theguardian.com/documentation/))
- Store it somewhere safe on your local machine (outside the repo) in a `.txt` file
- Specify the path to this file in `.env` file, by adding a new line with `export GUARDIAN_API_KEY=path/to/file`
- Use the functions in `innovation_sweet_spots.getters.guardian`

To see examples of using our public discourse analysis tools, check `innovation_sweet_spots/analysis/examples/public_discourse_analysis`.

### Hansard

Please ask Karlis to access the Hansard dataset. More details coming soon...

## :handshake: Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
