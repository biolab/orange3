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
2. Add identification od source data sets to the output data set.

If one of the tables is connected to the widget as the primary
table, the resulting table will contain its own attributes. If there
is no primary table, the attributes can be either a union of all
attributes that appear in the tables specified as "*Additional Tables*",
or their intersection, that is, a list of attributes common to all
the connected tables.

Example
-------

The widget can be used for merging the data from
two separate files, as shown below. Let's say we have two data sets with the same attributes,
one containing instances from the first experiment and the other instances from the second
experiment and we wish to join the two data tables together. We use **Concatenate** widget
to merge data sets by attributes (appending new rows under existing attributes).

Below we used a modified *Zoo* data set. In the [first](zoo-first.tab) **File** widget we loaded only animals beginning with letters A 
and B and in the [second](zoo-second.tab) one only animals beginning with the letter C. Upon concatenation we observe the new data
in the **Data Table** widget, where we see the complete table with animals from A to C.

<img src="images/Concatenate-Example.png" alt="image" width="600">
