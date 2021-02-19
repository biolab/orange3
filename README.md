<p align="center">
    <a href="https://orange.biolab.si/download">
    <img src="https://raw.githubusercontent.com/irgolic/orange3/README-shields/distribute/orange-title.png" alt="Orange Data Mining" height="200">
    </a>
</p>
<p align="center">
    <a href="https://orange.biolab.si/download" alt="Latest release">
        <img src="https://img.shields.io/github/v/release/biolab/orange3?label=download" />
    </a>
    <a href="https://orange3.readthedocs.io/en/latest/?badge=latest" alt="Documentation">
        <img src="https://readthedocs.org/projects/orange3/badge/?version=latest">
    </a>
    <a href="https://discord.gg/FWrfeXV" alt="Discord">
        <img src="https://img.shields.io/discord/633376992607076354?logo=discord&color=7389D8&logoColor=white&label=Discord">                                                                                                                                                                                                                                                  </a>
</p>

# Orange Data Mining
[Orange] is a data mining and visualization toolbox for novice and expert alike. To explore data with Orange, one requires __no programming or in-depth mathematical knowledge__. We believe that workflow-based data science tools democratize data science by hiding complex underlying mechanics and exposing intuitive concepts. Anyone who owns data, or is motivated to peek into data, should have the means to do so.

<p align="center">
    <a href="https://orange.biolab.si/download">
    <img src="https://raw.githubusercontent.com/irgolic/orange3/README-shields/distribute/orange-example-tall.png" alt="Example Workflow">
    </a>
</p>

[Orange]: https://orange.biolab.si/


## Installing

### Easy installation

For easy installation, [![Download](https://img.shields.io/github/v/release/biolab/orange3?label=download)](https://orange.biolab.si/download) the latest released Orange version from our website. To install an add-on, head to `Options -> Add-ons...` in the menu bar.

### Installing with Conda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS. Open the Anaconda prompt and type the following (copy individual lines; skip those starting with `#`).

```Shell
# Add conda-forge to your channels for access to the latest release
conda config --add channels conda-forge

# Create and activate an environment for orange
conda create python=3 --name orange3
conda activate orange3

# Install orange
conda install orange3
```

For installation of any add-ons, use `conda install orange3-<addon name>`. See specific add-on repositories for details.

#### Running

Open Anaconda prompt and activate the environment by `conda activate orange3`. Then run `orange-canvas` or `python3 -m Orange.canvas`. Add `--help` for a list of program options.

Starting up may take a while the first time.

### Installing with pip

To install Orange with pip, run the following.

```Shell
# Install build requirements via your system's package manager
sudo apt install virtualenv build-essential python3-dev

# Create an environment for Orange and its dependencies
virtualenv --python=python3 --system-site-packages orange3venv

# Activate the environment
source orange3venv/bin/activate

# Install Orange
pip install orange3
```

#### Running

Activate the environment by `source orange3venv/bin/activate`. Then run `orange-canvas` or `python3 -m Orange.canvas`. Add `--help` for a list of program options.

Starting up may take a while the first time.

### Installing with winget (Windows only)

To install Orange with [winget](https://docs.microsoft.com/en-us/windows/package-manager/winget/), run:

```Shell
winget install --id  UniversityofLjubljana.Orange 
```

## Developing

[![GitHub Actions](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fbiolab%2Forange3%2Fbadge&label=build)](https://actions-badge.atrox.dev/biolab/orange3/goto) [![codecov](https://img.shields.io/codecov/c/github/biolab/orange3)](https://codecov.io/gh/biolab/orange3) [![Contributor count](https://img.shields.io/github/contributors-anon/biolab/orange3)](https://github.com/biolab/orange3/graphs/contributors) [![Latest GitHub commit](https://img.shields.io/github/last-commit/biolab/orange3)](https://github.com/biolab/orange3/commits/master)

Want to write your own widget? [Use the Orange3 example addon template.](https://github.com/biolab/orange3-example-addon)

Want to get involved? Join us on [![Discord](https://img.shields.io/discord/633376992607076354?logo=discord&color=7389D8&logoColor=white&label=Discord)](https://discord.gg/FWrfeXV), introduce yourself in #general! 

Take a look at our [contributing guide](https://github.com/irgolic/orange3/blob/README-shields/CONTRIBUTING.md) and [style guidelines](https://github.com/biolab/orange-widget-base/wiki/Widget-UI).

Check out our widget development [![docs](https://readthedocs.org/projects/orange-widget-base/badge/?version=latest)](https://orange-widget-base.readthedocs.io/en/latest/?badge=latest) for a comprehensive guide on writing Orange widgets.

### The Orange ecosystem

The development of core Orange is primarily split into three repositories:

[biolab/orange-canvas-core](https://www.github.com/biolab/orange-canvas-core) implements the canvas,  
[biolab/orange-widget-base](https://www.github.com/biolab/orange-widget-base) is a handy widget GUI library,  
[biolab/orange3](https://www.github.com/biolab/orange3) brings it all together and implements the base data mining toolbox.	

Additionally, add-ons implement additional widgets for more specific use cases. [Anyone can write an add-on.](https://github.com/biolab/orange3-example-addon) Some of our first-party add-ons:

- [biolab/orange3-text](https://www.github.com/biolab/orange3-text)
- [biolab/orange3-bioinformatics](https://www.github.com/biolab/orange3-bioinformatics)
- [biolab/orange3-timeseries](https://www.github.com/biolab/orange3-timeseries)    
- [biolab/orange3-single-cell](https://www.github.com/biolab/orange3-single-cell)    
- [biolab/orange3-imageanalytics](https://www.github.com/biolab/orange3-imageanalytics)    
- [biolab/orange3-educational](https://www.github.com/biolab/orange3-educational)    
- [biolab/orange3-geo](https://www.github.com/biolab/orange3-geo)    
- [biolab/orange3-associate](https://www.github.com/biolab/orange3-associate)    
- [biolab/orange3-network](https://www.github.com/biolab/orange3-network)
- [biolab/orange3-explain](https://www.github.com/biolab/orange3-explain)

### Setting up for Orange core development

Fork the repository by pressing the fork button in top-right corner of this page. Then execute the following lines (copy them one by one!):

```Shell
conda create python=3 --name orange3
conda activate orange3

git clone ssh://git@github.com/<YOUR-USERNAME>/orange3

pip install -e orange3
```


Now you're ready to work with git. See GitHub's guides on [pull requests](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests), [forks](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/working-with-forks) if you're unfamiliar. If you're having trouble, get in touch on [![Discord](https://img.shields.io/discord/633376992607076354?logo=discord&color=7389D8&logoColor=white&label=Discord)](https://discord.gg/FWrfeXV).

#### Running

Run orange with `python -m Orange.canvas` (after activating the conda environment).

`python -m Orange.canvas -l 2 --no-splash --no-welcome` will skip the splash screen and welcome window, and output more debug info. Use `-l 4` for more.

Add `--clear-widget-settings` to clear the widget settings before start.

To explore the dark side of the orange, try `--style=fusion:breeze-dark`

Argument `--help` lists all available options.

To run tests, use `unittest Orange.tests Orange.widgets.tests`


### Setting up for development of all components

If you wish to also contribute to base components (the widget base and the canvas), these two repositories must also be cloned from Github instead of being installed as dependency of Orange3 (which happens above).

First, fork all repositories to which you want to contribute. Then type:

```Shell
conda create python=3 --name orange3
conda activate orange3

git clone ssh://git@github.com/<YOUR-USERNAME>/orange-widget-base
pip install -e orange-widget-base

git clone ssh://git@github.com/<YOUR-USERNAME>/orange-canvas-core
pip install -e orange-canvas-core

git clone ssh://git@github.com/<YOUR-USERNAME>/orange3
pip install -e orange3

# Same for any add-on repositories
```

It's important to install `orange-base-widget` and `orange-canvas-core` before `orange3` to ensure that `orange3` will use your local versions.
