Set Enrichment
==============

![Databases widget icon]

Updates local systems biology databases, like gene ontologies,
annotations, gene names, protein interaction networks, and similar.

Signals
-------

Inputs:

:   -   (None)

Outputs:

:   -   (None)

Description
-----------

Many widgets in Orange bioinformatics add-on rely on information on
genes, gene sets, pathways, and alike. This information is stored on
your local computer when the widget requires them for the first time.
The corresponding data comes from different web resources, and is either
preprocessed and then stored on Orange server, or accessed directly from
a dedicated web site.

Orange does not change the data on your local computer, and with time
this becomes different to the newest version of the online data sets.
Databases widget can update the data on your local machine, and can also
be used to manage (remove or add) any locally stored systems biology
data set.

![Databases widget]

To get a more detailed information on the particular database that
requires an update, hover on its Update button.

![Databases widget][1]

  [Databases widget icon]: ../../orangecontrib/bio/widgets/icons/Databases.svg
  [Databases widget]: images/databases-stamped.png
  [1]: images/databases-hover.png