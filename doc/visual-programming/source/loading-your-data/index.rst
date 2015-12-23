Loading your Data
=================

Orange comes with its `own data format <http://docs.orange.biolab.si/reference/rst/Orange.data.formats.html#tab-delimited>`_, but can
also handle native Excel (.xlsx or .xls), comma- or tab-delimited data files. The input data
set is usually a table, with data instances (samples) in rows and
data attributes in columns. Attributes can be of different type
(continuous, discrete, and strings) and kind (input features, meta attributes, and class). Data attribute type and kind can be provided
in the data table header. This can be changed later, after reading the
data with `Select Columns <http://docs.orange.biolab.si/widgets/rst/data/selectattributes.html#select-attributes>`_ widget.

In a Nutshell
-------------

-   Orange can import any comma- or tab-delimited data file, or Excel's native files or Google Sheets document. Use `File <http://docs.orange.biolab.si/3/visual-programming/widgets/data/file.html>`_
    widget and then, if needed, define the class and meta attributes in
    `Select Columns <http://docs.orange.biolab.si/3/visual-programming/widgets/data/selectcolumns.html>`_ widget.
-   Attribute names in the column header
    can be preceded with a label followed by a hash. Use c for class
    and m for meta attribute, i to ignore a column, and C, D, S for
    continuous, discrete and string attribute types. Examples: C\#mpg,
    mS\#name, i\#dummy.
-   An alternative to the hash notation is Orange's native format with three
    header rows: the first with attribute names, the second specifying
    the type (**continuous**, **discrete** or **string**), and the third
    proving information on the kind of attribute (**class**, **meta** or **string**).

An Example: Data from Excel
---------------------------

Here is an example data set (download it from :download:`sample.xlsx <sample.xlsx>`) as entered in Excel:

.. image:: spreadsheet1.png
    :width: 600 px
    :align: center

The file contains a header row, eight data instances (rows) and seven data attributes (columns). Empty cells in the table denote missing data entries. Rows represent genes; their function (class) is provided in the first column and their name in the second. The remaining columns store measurements that characterize each gene. With this data, we could, say, develop a classifier that would predict gene function from its characteristic measurements.

Let us start with a simple workflow that reads the data and displays it in a table:

.. image:: file-data-table-workflow.png
    :align: center

To load the data, open the File widget (double click on the icon of the widget), click on the file browser icon ("...") and locate the downloaded file (from :download:`sample.xlsx <sample.xlsx>`) on your disk:

.. image:: loadingyourdata.png
    :width: 401 px
    :align: center

The **File** widget sends the data to the **Data Table** widget. Double click the **Data Table** widget to see its contents:

.. image:: table-widget.png
    :width: 900 px
    :align: center

Orange correctly assumed that a column with gene names is a meta information, which is displayed in **Data Table** in columns shaded with light-gray. It has wrongly guessed that `heat 20`, the last non-meta column in our data file, is a class column. To correct this in Orange, we can feed the data to the `Select Columns <http://docs.orange.biolab.si/widgets/rst/data/selectattributes.html#select-attributes>`_ widget:

.. image:: select-columns-schema.png
    :align: center

Opening the `Select Columns <http://docs.orange.biolab.si/widgets/rst/data/selectattributes.html#select-attributes>`_ widget reveals Orange's classification of attributes. We would like all our continuous attributes to be data features, gene function to be our target variable and gene names considered as meta attributes. We can obtain this by dragging the attribute names around the boxes in **Select Columns**:

.. image:: select-columns-start.png
    :width: 413 px
    :align: center

To correctly reassign attribute types, drag attribute named `function`
to a **Class** box, and attribute named `gene` to a **Meta Attribute**
box. The `Select Columns <http://docs.orange.biolab.si/widgets/rst/data/selectattributes.html#select-attributes>`_ widget should now look like this:

.. image:: select-columns-reassigned.png
    :width: 413 px
    :align: center

Change of attribute types in *Select Columns* widget should be confirmed
by clicking the **Apply** button. The data from this widget is fed into
`Data Table <http://docs.orange.biolab.si/widgets/rst/data/datatable.html#data-table>`_ that now renders the data just the way we intended:

.. image:: data-table-with-class1.png
    :width: 548 px
    :align: center

We could also define the domain for this data set in a different way.
Say, we could make the data set ready for regression, and use `heat 0`
as a continuous class variable, keep gene function and name as meta
variables, and remove `heat 10` and `heat 20` from the data set:

.. image:: select-columns-regression.png
    :width: 413 px
    :align: center

By setting the attributes as above, the rendering of the data in the
Data Table widget gives the following output:

.. image:: data-table-regression1.png
    :width: 506 px
    :align: center

Header with Attribute Type Information
--------------------------------------

Consider again the :download:`sample.xlsx <sample.xlsx>` data set. This time 
we will augment the names of the attributes with prefixes
that define attribute type (continuous, discrete, string) and kind (class or meta attribute)
Prefixes are separated from the attribute name with a hash sign ("\#"). Prefixes for the attribute kind are:

-   c: class attribute
-   m: meta attribute
-   i: ignore the attribute

and for the type:

-   C: Continuous
-   D: Discrete
-   S: String

This is how the header with augmented attribute names looks like in
Excel (:download:`sample-head.xlsx <sample-head.xlsx>`):

.. image:: spreadsheet-simple-head1.png
    :width: 414 px
    :align: center

We can again use a **File** widget to load this data set and then render it in the **Data Table**:

.. image:: select-cols-simplified-header.png
    :width: 509 px
    :align: center

Notice that the attributes we have ignored (label "i" in the
attribute name) are not present in the data set.

Three-Row Header Format
-----------------------

Orange's legacy native data format is a tab-delimited text file with three header rows. The first row lists the attribute names, the second row defines their type (continuous, discrete and string, or abbreviated c, d and s), and the third row an optional kind (class, meta, or ignore). Here is an example:

.. image:: excel-with-tab1.png
    :width: 585 px
    :align: center

Data from Google Sheets
-----------------------

Orange can read data from Google Sheets, as long as it conforms to the data presentation rules we have presented above. In Google Sheets, copy the shareable link (Share button, then Get shareable link) and paste it in the `Data File / URL` box of the File widget. For a taste, here's one such link you can use: `http://bit.ly/1J12Tdp <http://bit.ly/1J12Tdp>`_, and the way we have entered it in the **File** widget:

.. image:: file-google-sheets.png
    :width: 402 px
    :align: center

Data from LibreOffice
---------------------

If you are using LibreOffice, simply save your files in Excel (.xlsx or .xls) format (available from the drop-down menu under *Save As Type*).

.. image:: saving-tab-delimited-files.png
    :align: center

