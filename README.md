# Innovation Sweet Spots

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter), in brief:
  - Install: `git-crypt`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

### Data access

To download input data from Nesta database, you will first need to decrypt the config files (if you don't have the key, reach out to Karlis)

```
$ git stash
$ git-crypt unlock /path/to/key
```

The most recent version of the input data can then be fetched by running

```shell
python innovation_sweet_spots/getters/inputs.py
```

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
