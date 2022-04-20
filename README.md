# :satellite: Innovation Sweet Spots

**_Open-source code for data-driven horizon scanning_**

## :wave: Welcome!

Innovation Sweet Spots is an experimental, data-driven horizon scanning project, led by Nesta's [Discovery Hub](https://www.nesta.org.uk/project/discovery-hub/). Read more about our motivation on [Medium](https://medium.com/@nesta_uk/in-search-of-innovation-sweet-spots-can-data-science-help-us-see-through-tech-hype-1f140f50c18b), and check out our [first report on green technologies](https://www.nesta.org.uk/data-visualisation-and-interactive/innovation-sweet-spots/).

We are building upon Nesta's [Data Analytics Practice](https://www.nesta.org.uk/project/data-analytics/) expertise and previous work on [innovation mapping](https://www.nesta.org.uk/feature/innovation-methods/innovation-mapping/), leveraging data science and machine-learning methods to track the trajectory of innovations and technologies for social good.

By combining insights across several large [datasets](#datasets) that are commonly only analysed in isolation, we paint a multi-dimensional picture of the innovations indicating the resources they are attracting and how they are perceived.

_NB: The codebase and these guidelines are still under development, with several parts of the analyses being presently refactored into modules._

## :hammer_and_wrench: Installation

**Step 1.** Check that you meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter). In brief, you should:

- Install the following components:
  - [gh](https://formulae.brew.sh/formula/gh), GitHub command line tool
  - [direnv](https://formulae.brew.sh/formula/direnv#default), for using environment variables
  - [git-crypt](https://github.com/AGWA/git-crypt/blob/master/INSTALL.md#installing-on-mac-os-x), tool for encryption of sensitive files
- Have a Nesta AWS account, and install and configure your [AWS Command Line Interface](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html)

**Step 2.** Run the following command from the repo root folder:

```
make install
```

This will configure the development environment:

- Setup the conda environment with the name `innovation_sweet_spots`
- Configure pre-commit actions (for example, running a code formatter before each commit)
- Configure metaflow

The expected output:

```
conda env create -q -n innovation_sweet_spots -f environment.yaml
Collecting package metadata (repodata.json): ...working... done
Solving environment: ...working... done
Preparing transaction: ...working... done
Verifying transaction: ...working... done
Executing transaction: ...working... done
/Library/Developer/CommandLineTools/usr/bin/make -s pip-install
source bin/conda_activate.sh && conda_activate &&  pre-commit install --install-hooks
pre-commit installed at .git/hooks/pre-commit
source bin/conda_activate.sh && conda_activate &&  /bin/bash ./bin/install_metaflow_aws.sh
INSTALL COMPLETE
```

**Step 3.** Activate the newly created conda environment and you're good to go!

```shell
$ conda activate innovation_sweet_spots
```

## :floppy_disk: Datasets

To uncover research, investment and public discourse trends, we are presently using the following data:

- **[Gateway to Research (GtR)](https://gtr.ukri.org/)**: Research projects funded by UKRI
- **[Crunchbase](https://crunchbase.com/)**: Global company directory
- **[The Guardian news](https://open-platform.theguardian.com/)**: to the best of our knowledge, the only major UK newspaper to make its text freely available for research.
- **[Hansard](https://zenodo.org/record/4066772#.YXCN1kbYrlw)**: Records of parliamentary debates

All these datasets except Crunchbase are freely available. Note, however, that this project accesses some of these large datasets (namely GtR and Crunchbase) via our internal Nesta database and as such are intended for internal use.

In the future, we might add other datasets to our approach.

<details>
  <summary>Click to read data access guidelines</summary>

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

Coming soon...

  </details>

## :handshake: Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
