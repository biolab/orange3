PIPA
====

![Widget icon]

Signals
-------

Inputs:

:   -   None

Outputs:

:   -   

        Selected microarrays (ExampleTable)

        :   Selected experiments. Each annotated column contains results
            of a single experiment or, if the corresponding option is
            chosen, the average of multiple replicates.

Description
-----------

![PIPA widget]

The PIPA widget lists accessible experiments, which can be filtered with
the “Search” box at the top. The selected experiments will appear on the
output when the “Commit” button is clicked. You can connect the output
of the PIPA widget to any Orange widget which accepts ExampleTable as
input. The widget will automatically save (cache) downloaded data and
you will therefore be able to analyse them offline.

To select multiple experiments click them while holding the “Control”
key. For frequent combinations of selections use the “Experiment Sets”
feature: select experiments and click on the “+” button.

The logarithmic transformation is computed as the binary logarithm of
the (value + 1). If username and passwords are not given, only the
public experiments will be accessible.

Examples
--------

In the schema below we connected PIPA to Data Table, GO Enrichment
Analysis, and Distance Map (through Attribute Distance) widgets.

![Example schema with PIPA widget]

The Data Table widget below contains the output from the PIPA widget.
Each column contains gene expressions of a single experiment. The labels
are shown in the table header.

![Output of PIPA in a data table.]

The Distance Map widget shows distances between experiments. The
distances are measured with Attribute Distance widget, which was set to
compute Pearson distances (as (1-PearsonCorrelation)/2).

![Distances between experiments.]

  [Widget icon]: ../../orangecontrib/bio/widgets/icons/PIPA.png
  [PIPA widget]: images/PIPA.*
  [Example schema with PIPA widget]: images/PIPA_schema.*
  [Output of PIPA in a data table.]: images/PIPA_datatable.*
  [Distances between experiments.]: images/PIPA_distance.*