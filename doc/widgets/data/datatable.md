Data Table
==========

![Data Table icon](icons/data-table.png)

Displays attribute-value data in a spreadsheet.

Signals
-------

**Inputs**:

- **Data**
 
  Attribute-valued data set.

**Outputs**:

- **Selected Data**

  Selected data instances.

Description
-----------

**Data Table** widget receives one or more data sets in its input and
presents them as a spreadsheet. Columns in white are regular attributes with
either discrete or continuous values. Columns in grey are class attribues, while
columns in light navy are meta attributes. Data instances may be sorted by any
attribute value. 

The widget also supports manual selection of data instances.

![Data table with Iris data set](images/DataTable-stamped.png)

1.  Info on the current data set size and the number and types of
    attributes.
2.  Use '*Restore Order*' button to
    reorder data instance after attribute-based sorting.
3.  Values of continuous attributes can be visualized with bars; colors can be attributed to different classes.
4.  Data instances (rows) can be selected and sent to the widget's
    output channel.
5.  While *Auto-send is on*, all changes will be automatically communicated to other widgets. 
    Alternatively press '*Send Selected Rows*'.

Example
-------

We used two **File** widgets to read the *Iris* and *Glass* data set (provided in
Orange distribution), and send them to the **Data Table** widget.

![Example data table schema](images/DataTable-Schema.png)

Selected data instances in the first **Data Table** are passed to the second
**Data Table**. Notice that we can select which data set to view (iris or
glass). Changing from one data set to another alters the communicated
selection of the data instances if "*Commit on any change*" is selected.

<img src="images/DataTable-Example.png" alt="image" width="600">
