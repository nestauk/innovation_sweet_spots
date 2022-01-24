# Innovation Sweet Spots

## Setup

Check that you meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter). In brief, you should:

- Install `git-crypt`
- Have a Nesta AWS account configured with `awscli`

In terminal, run `make install` to configure the development environment. This will do the following:

- Setup the conda environment with the name `innovation_sweet_spots`
- Configure pre-commit (for example, running a code formatter before each commit)
- Configure metaflow to use AWS

You should then activate the newly created conda environment and install the repository package:

```shell
$ conda activate innovation_sweet_spots
$ pip install -e .
```

### Data access

To uncover research, investment and public discourse trends, we are using the following data:

- **[Gateway to Research (GtR)](https://gtr.ukri.org/)**: Research projects funded by UKRI
- **[Crunchbase](https://crunchbase.com/)**: Global company directory
- **[The Guardian news](https://open-platform.theguardian.com/)**: to the best of our knowledge, the only major UK newspaper to make its text freely available for research.
- **[Hansard](https://zenodo.org/record/4066772#.YXCN1kbYrlw)**: Records of parliamentary debates

All these datasets except Crunchbase are freely available. Note, however, that this project accesses some of these large datasets (namely GtR and Crunchbase) via our internal Nesta database.

#### Research project and company data

To download GtR and Crunchbase datasets from Nesta database, you will first need to decrypt the config files (if you don't have the key, reach out to Karlis).

```shell
$ git stash
$ git-crypt unlock /path/to/key
```

The most recent version of the Gateway to Research (GtR) and Crunchbase datasets can then be fetched by running the command below. Note that you need to be connected via Nesta's VPN when accessing the database.

```shell
$ python innovation_sweet_spots/pipeline/fetch_daps1_data/flow.py --no-pylint --environment=conda run
```

#### The Guardian news

Coming soon...

#### Hansard

Coming soon...

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
