# Innovation Sweet Spots

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter), in brief:
  - Install: `git-crypt`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS
- Run `conda config --add channels conda-forge`

### Data access

To download input data from Nesta database, you will first need to decrypt the config files (if you don't have the key, reach out to Karlis)

```
$ git stash
$ git-crypt unlock /path/to/key
```

#### Research project and business data

The GtR and CB data can be fetched by running `make fetch-daps1`. Note that at the moment
there are two data fetching pipelines (the other using `innovation_sweet_spots.getters.inputs`)
that will have to be integrated in the near future.

#### Parliamentary speech data

For downloading Hansard data, run the following command

```
python -c "from innovation_sweet_spots.getters.inputs import download_hansard_data; download_hansard_data();"
```

#### News data

In case of the news dataset, we're not downloading any large data beforehand. Instead, you can search for a specific term via Guardian API. For accessing the API, you you'll need to:

- Request an API key from Guardian website ([see here](https://open-platform.theguardian.com/documentation/))
- Store it somewhere safe on your local machine (outside the repo) in a `.txt` file
- Specify the path to this file in `.env` file, by adding a new line with `export GUARDIAN_API_KEY=path/to/file`
- Use the functionalities in `innovation_sweet_spots.getters.guardian`

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
