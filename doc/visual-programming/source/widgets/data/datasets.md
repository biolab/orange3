Datasets
========

Load a dataset from an online repository.

**Outputs**

- Data: output dataset

The **Datasets** widget retrieves selected dataset from a server upon double-click and sends it to the output. The file is downloaded to the local disk and thus will be available even without internet connection. Each dataset is provided with a description and information on the data size, number of instances, number of variables, target and tags.

![](images/Datasets-stamped.png)

1. Search box to filter dataset on titles, variables, targets and tags. 
2. Change the language of displayed datasets.
3. Select the domain of displayed datasets.
4. Content of available datasets. Each dataset is described with the size, number of instances and variables, type of the target variable and tags.
5. Formal description of the selected dataset.

Example
-------

Orange workflows can start with the **Datasets** widget instead of the **File** widget. In the example below, the widget retrieves a dataset from an online repository (Kickstarter data), which is subsequently sent to both the [Data Table](../data/datatable) and the [Distributions](../visualize/distributions).

![](images/Datasets-Workflow.png)
