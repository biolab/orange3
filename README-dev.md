Orange development
==================

The source code of [Orange] is versioned in [Git] and hosted on [GitHub]. 
If you want to contribute to this open-source project you will have to use git. However, for minor experimentation with the source code you can also get by without. 

[Orange]: https://orange.biolab.si/
[Git]: https://git-scm.com/
[GitHub]: https://github.com/biolab/orange

Prerequisites
-------------

[Orange] is written mostly in Python, therefore you'll need [Python 3] version 3.6 or newer.

You will also need a C/C++ compiler. On Windows, you can get one by installing [Visual Studio].
A slightly more "minimalistic" option is to install only its [Build Tools].

[Python 3]: https://www.python.org
[Visual Studio]: https://visualstudio.microsoft.com/vs/
[Build Tools]: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Source code
-----------

Get the source code by cloning the git repository

    git clone https://github.com/biolab/orange3.git

or, alternatively, download and unpack the [ZIP archive] of the source code from [GitHub].

[ZIP archive]: https://github.com/biolab/orange3/archive/master.zip

Building
--------

Consider using virtual environments to avoid package conflicts. 

Install the required Python packages

    pip install -r requirements.txt
    
and run the setup script with a development option, which will link to the source code instead of creating a new package in Python's site-packages.

    python setup.py develop
    
Verify the installation by importing the Orange package from Python and loading an example Iris dataset.

    >>> import Orange
    >>> print(Orange.data.Table("iris")[0])
    [5.1, 3.5, 1.4, 0.2 | Iris-setosa]

Using the graphic user interface requires some additional packages.

    pip install -r requirements-gui.txt

To start Orange GUI from the command line, run:

    python3 -m Orange.canvas

Contributing
------------

If you've made improvements that you want to contribute, you'll need your own fork of the [GitHub] repository. After committing and pushing changes to your fork, you can create a pull request. We will review your contribution and hopefully merge it after any potential corrections. 

You can view the list of open [pull requests] and known [issues] on GitHub.

[pull requests]: https://github.com/biolab/orange3/pulls
[issues]: https://github.com/biolab/orange3/issues
