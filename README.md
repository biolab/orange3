<p align="center">
    <a href="https://orange.biolab.si/download">
    <img src="https://raw.githubusercontent.com/irgolic/orange3/README-shields/distribute/orange-title.png" alt="Orange Data Mining" height="200">
    </a>
</p>
<p align="center">
    <a href="https://orange.biolab.si/download" alt="Latest release">
        <img src="https://img.shields.io/github/v/release/biolab/orange3?label=download" /></a>
    <a href="https://orange3.readthedocs.io/en/latest/?badge=latest" alt="Documentation">
        <img src="https://readthedocs.org/projects/orange3/badge/?version=latest"></a>
</p>

# Orange
[Orange] is a data mining and visualization toolbox for novice and expert alike. To explore data with Orange, one requires __no__ programming or in-depth mathematical knowledge. We believe that workflow-based data science tools democratize data science by hiding complex underlying mechanics and exposing intuitive concepts. Anyone who owns data, or is motivated to peek into data, should have the means to do so.

<p align="center">
    <a href="https://orange.biolab.si/download">
    <img src="https://raw.githubusercontent.com/irgolic/orange3/README-shields/distribute/orange-example-tall.png" alt="Example Workflow">
    </a>
</p>

[Orange]: https://orange.biolab.si/

## Contributing

[![GitHub Actions](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fbiolab%2Forange3%2Fbadge&label=build)](https://actions-badge.atrox.dev/biolab/orange3/goto)
[![codecov](https://img.shields.io/codecov/c/github/biolab/orange3)](https://codecov.io/gh/biolab/orange3)
[![Contributor count](https://img.shields.io/github/contributors-anon/biolab/orange3)](https://github.com/biolab/orange3/graphs/contributors)
[![Latest GitHub commit](https://img.shields.io/github/last-commit/biolab/orange3)](https://github.com/biolab/orange3/commits/master)

Want to get involved? Join us on [![Discord](https://img.shields.io/discord/633376992607076354?logo=discord&color=7389D8&logoColor=white&label=Discord)](https://discord.gg/FWrfeXV), introduce yourself in #general!

Take a look at our [contributing guide](https://github.com/irgolic/orange3/blob/README-shields/CONTRIBUTING.md), it might answer some questions, and it outlines the standards we adhere to.

Check out our widget development [![docs](https://readthedocs.org/projects/orange-widget-base/badge/?version=latest)](https://orange-widget-base.readthedocs.io/en/latest/?badge=latest) for a comprehensive guide on writing Orange widgets.

If you're looking for a good starting point, check out our [![good first issues](https://img.shields.io/github/issues/biolab/orange3/good%20first%20issue?label=good%20first%20issues)](https://github.com/biolab/orange3/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).


### The Orange Ecosystem

The development of Orange is primarily split into three repositories:

[biolab/orange-canvas-core](https://www.github.com/biolab/orange-canvas-core) implements canvas elements,  
[biolab/orange-widget-base](https://www.github.com/biolab/orange-widget-base) implements a widget window's interface elements,  
[biolab/orange3](https://www.github.com/biolab/orange3) brings it all together and implements the base data mining toolbox.	

Additionally, add-ons implement additional widgets for more specific use cases. [Anyone can write an add-on.](https://github.com/biolab/orange3-example-addon) Below is a list of our first-party add-ons:

[biolab/orange3-text](https://www.github.com/biolab/orange3-text)    
[biolab/orange3-bioinformatics](https://www.github.com/biolab/orange3-bioinformatics)    
[biolab/orange3-timeseries](https://www.github.com/biolab/orange3-timeseries)    
[biolab/orange3-single-cell](https://www.github.com/biolab/orange3-single-cell)    
[biolab/orange3-imageanalytics](https://www.github.com/biolab/orange3-imageanalytics)    
[biolab/orange3-educational](https://www.github.com/biolab/orange3-educational)    
[biolab/orange3-geo](https://www.github.com/biolab/orange3-geo)    
[biolab/orange3-associate](https://www.github.com/biolab/orange3-associate)    
[biolab/orange3-network](https://www.github.com/biolab/orange3-network)

### Setting up

1. Set up a __virtual environment__. We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  
`conda create python=3 --name orange3`
2. __Fork__ your chosen repository.  
Press the fork button in top-right corner of the page
3. __Clone__ it.   
`git clone ssh://git@github.com/<your-username>/<repo-name>`
4. __Install__ it.  
`pip install -e .` or `python setup.py develop`

Now you're ready to work with git. See GitHub's guides on [pull requests](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests), [forks](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/working-with-forks) if you're unfamiliar.  
If you're having trouble, get in touch on [![Discord](https://img.shields.io/discord/633376992607076354?logo=discord&color=7389D8&logoColor=white&label=Discord)](https://discord.gg/FWrfeXV).

## Installing

For easy installation, [![Download](https://img.shields.io/github/v/release/biolab/orange3?label=download)](https://orange.biolab.si/download) the latest released Orange version from our website.

### Installing with Miniconda / Anaconda

Orange requires Python 3.6 or newer.

First, install [Miniconda] for your OS. Create virtual environment for Orange:

```Shell
conda create python=3 --name orange3
```
In your Anaconda Prompt add conda-forge to your channels:

```Shell
conda config --add channels conda-forge
```

This will enable access to the latest Orange release. Then install Orange3:

```Shell
conda install orange3
```

[Miniconda]: https://docs.conda.io/en/latest/miniconda.html

To install the add-ons, follow a similar recipe:

```Shell
conda install orange3-<addon name>
```

See specific add-on repositories for details.

### Installing with pip

To install Orange with pip, run the following.

```Shell
# Install some build requirements via your system's package manager
sudo apt install virtualenv build-essential python3-dev

# Create a separate Python environment for Orange and its dependencies ...
virtualenv --python=python3 --system-site-packages orange3venv
# ... and make it the active one
source orange3venv/bin/activate

# Install Orange
pip install orange3
```

### Installing with winget (Windows only)

To install Orange with [winget](https://docs.microsoft.com/en-us/windows/package-manager/winget/), run:

```Shell
winget install --id  UniversityofLjubljana.Orange 
```

### Starting Orange GUI

To start Orange GUI from the command line, run:

```Shell
orange-canvas
# or
python3 -m Orange.canvas
```

Append `--help` for a list of program options.
