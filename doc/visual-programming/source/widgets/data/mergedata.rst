Merge Data
==========

.. figure:: icons/merge-data.png

Merges two data sets, based on values of selected attributes.

Signals
-------

**Inputs**:

-  **Data A**

   Attribute-valued data set.

-  **Data B**

   Attribute-valued data set.

**Outputs**:

-  **Merged Data**

   Instances from input data A to which attributes from input data B are
   added.

Description
-----------

The **Merge Data** widget is used to horizontally merge two data sets, based
on values of selected attributes. In the input, two data sets are
required, A and B. The widget allows selection of an attribute from each
domain, which will be used to perform the merging. The widget produces
one output. It corresponds to instances from the input data A
to which attributes from B are appended, and B+A to instances from B to
which attributes from A are appended.

Merging is done by values of selected (merging) attributes. First,
the value of the merging attribute from A is taken and instances from B
are searched for matching values. If more than a single instance from B
is found, the first one is taken and horizontally merged with the
instance from A. If no instance from B matches the criterion, unknown
values are assigned to the appended attributes.

.. figure:: images/MergeData-stamped.png

1. List of comparable attributes from Data A
2. List of comparable attributes from Data B
3. Information on Data A
4. Information on Data B
5. If checked, instances from B without the match are excluded form the output.
   If not checked, instances from B without the match are assigned
   unknown values to the appended attributes.
6. Produce a report.

Example
-------

Merging two data sets results in appending new attributes to the
original file, based on a selected common attribute. In the example
below, we wanted to merge the **zoo.tab** file containing only factual
data with :download:`zoo-with-images.tab <../data/zoo-with-images.tab>`
containing images. Both files share a common string attribute *names*. Now, we
create a workflow connecting the two files. The *zoo.tab* data is
connected to **Data A** input of the **Merge Data** widget, and the
*zoo-with-images.tab* data to the **Data B** input. Outputs of the
**Merge Data** widget is then connected to the :doc:`Data Table <../data/datatable>` widget.
In the latter, the **Merged Data** channels are shown, where image attributes
are added to the original data.

.. figure:: images/MergeData-Example.png

The case where we want to include all instances in the output, even those
where no match by attribute *names* was found, is shown in the following workflow.

.. figure:: images/MergeData-Example2.png

Hint
----

If the two data sets consist of equally-named attributes (other than
the ones used to perform the merging), Orange will check by default for
consistency of the values of these attributes and report an error in
case of non-matching values. In order to avoid the consistency checking,
make sure that new attributes are created for each data set: you may use the
'*Columns with the same name in different files represent different
variables*' option in the :doc:`File <../data/file>` widget for loading the data.
