The library tomotopy does not currently run on M1 macs. This can be overcome by creating a Rosetta virtual environment and installing tomotopy there.
Follow these steps:

1. Go to `applications` in mac and duplicate the terminal application (or iterm2 or whatever you use)
2. Rename the duplicate terminal application to something like “Rosetta terminal”
3. Right click on the duplicate terminal application, click get info, select `Open using Rosetta`
4. Put the duplicate terminal application in dock
5. Open the Rosetta terminal
6. Type `arch` in the terminal, if it returns `i386` that confirms you’re on Rosetta, if it says `arm64` you’re still on M1
7. Make a new virtual environment using this:

```bash
CONDA_SUBDIR=osx-64 conda create -n <virtual_env_name> python=3.8.13
conda activate <virtual_env_name>
conda config --env --set subdir osx-64
```

8. Now you can install tomotopy in the Rosetta environment with `pip install tomotopy`

Now that the environment is created, if you want to use the environment in the future, you just need to open the Rosetta terminal and `run conda activate <virtual_env_name>`.
