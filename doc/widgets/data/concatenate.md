Concatenate
===========

![image](icons/concatenate.png)

Concatenates data from multiple sources.

Signals
-------

**Inputs**:

- **Primary Data**

  Data set that defines the attribute set.

- **Additional Data**

  Additional data set.

**Outputs**:

- **Data**

Description
-----------

The widget concatenates multiple sets of instances (data sets). The merge is
“vertical”, in a sense that two sets of 10 and 5 instances yield a new
set of 15 instances.

![image](images/Concatenate-stamped.png)

1. Set the attribute merging method.
2. Add identification od source data sets to the output data set..

If one of the tables is connected to the widget as the primary
table, the resulting table will contain its own attributes. If there
is no primary table, the attributes can be either a union of all
attributes that appear in the tables specified as “Additional Tables”,
or their intersection, that is, a list of attributes common to all
the connected tables.

Example
-------

The widget can be used, for instance, for merging the data from
three separate files, as shown below.

![image](images/Concatenate-Example.png)
