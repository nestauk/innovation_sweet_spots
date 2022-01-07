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

**NB: This is subject to refactoring in the near future!**

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
