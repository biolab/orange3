
Datasets
========

Load a dataset from an online repository.

Outputs
    Data
        output dataset


**Datasets** widget retrives selected dataset from the server and sends it to the output. File is downloaded to the local memory and thus instantly available even without the internet connection. Each dataset is provided with a description and information on the data size, number of instances, number of variables, target and tags.

.. figure:: images/Datasets-stamped.png

1. Information on the number of datasets available and the number of them downloaded to the local memory.
2. Content of available datasets. Each dataset is described with the size, number of instances and variables, type of the target variable and tags.
3. Formal description of the selected dataset.
4. If *Send Data Automatically* is ticked, selected dataset is communicated automatically. Alternatively, press *Send Data*.

Example
-------

Orange workflows can start with **Datasets** widget instead of **File** widget. In the example below, the widget retrieves a dataset from an online repository (Kickstarter data), which is subsequently sent to both the :doc:`Data Table <../data/datatable>` and the :doc:`Distributions <../visualize/distributions>`.

.. figure:: images/Datasets-Workflow.png
