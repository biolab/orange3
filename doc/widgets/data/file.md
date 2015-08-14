File
====

![File icon](icons/file.png)

Reads attribute-value data from an input file.

Signals
-------

**Inputs**:

 - (None)

**Outputs**:

 - **Data**

  Attribute-valued data from the input file.

Description
-----------

**File** widget [**reads the input data file**](http://docs.orange.biolab.si/data/rst/index.html) (data table with data instances)
and sends the data set to its output channel. History of the most recently opened files is maintained in the widget. The widget also includes a directory with sample data sets that come pre-installed with Orange.

The widget reads data from Excel (**.xlsx**), simple tab-delimited (**.txt**) or comma-separated
files (**.csv**).

![File widget with loaded Iris data set](images/File-stamped.png)

1. Browse for a data file.
2. Browse through previously opened data files, or load any of the sample ones.
3. Reloads currently selected data file.
4. Information on the loaded data set: data set size, number and types of data features.
5. Allows you to distinguish between columns with the same name across files (otherwise columns with the same name will be considered as the same attribute).

Example
-------

Most Orange workflows would probably start with the **File** widget. In the
schema below, the widget is used to read the data that is sent to both the
**Data Table** and the **Box Plot** widget.

![Example schema with File widget](images/File-Workflow.png)

Loading your data
-----------------

-   Orange can import any comma, .xlsx or tab-delimited data file. Use [File] (http://docs.orange.biolab.si/widgets/rst/data/file.html#file)
    widget and then, if needed, select class and meta attributes in
    [Select Columns] (http://docs.orange.biolab.si/widgets/rst/data/selectattributes.html#select-attributes) widget.
-   To specify the domain and the type of the attribute, attribute names
    can be preceded with a label followed by a hash. Use c for class
    and m for meta attribute, i to ignore a column, and C, D, S for
    continuous, discrete and string attribute types. Examples: C\#mpg,
    mS\#name, i\#dummy. Make sure to set **Import Options** in [File] (http://docs.orange.biolab.si/widgets/rst/data/file.html#file)
    widget and set the header to **Orange simplified header**.
-   Orange's native format is a tab-delimited text file with three
    header rows. The first row contains attribute names, the second the
    type (**continuous**, **discrete** or **string**), and the third
    the optional element (**class**, **meta** or **string**).

![image](images/spreadsheet-simple-head1.png)

Read more on loading your data [here](http://docs.orange.biolab.si/data/rst/index.html).
