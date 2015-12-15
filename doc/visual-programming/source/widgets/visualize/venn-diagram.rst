Venn Diagram
============

.. figure:: icons/venn-diagram.png

Plots `Venn diagram <http://en.wikipedia.org/wiki/Venn_diagram>`__ for
two or more data subsets.

Signals
-------

**Inputs**:

-  **Data**

Input data set.

**Outputs**:

-  **Selected Data**

A subset of instances that the user has manually selected from the
diagram.

Description
-----------

**Venn Diagram** displays logical relations between data sets. This
projection shows two or more data sets represented by circles of
different colors. Intersections are subsets that belong to more than one
data set. To further analyse or visualize the subset click on an
intersection.

.. figure:: images/venn-workflow.png

.. figure:: images/venn-identifiers-stamped.png

1. Information on the input data.
2. Select identifiers by which to compare the data.
3. If *Auto commit* is on, changes are automatically communicated to
   other widgets. Alternatively, click *Commit*.
4. *Save graph* saves the graph to the computer in a .svg or .png
   format.

Examples
--------

The easiest way to use **Venn Diagram** is to select data subsets and
find matching instances in the visualization. We use the *breast-cancer*
data set to select two subsets with :doc:`Select Rows<../data/selectrows>` widget - the first
subset is that of breast cancer patients aged between 40 and 49 and the
second is that of patients with tumor size between 25 and 29. **Venn
Diagram** helps us find instances that correspond to both criteria,
which can be found in the intersection of the two circles.

.. figure:: images/VennDiagram-Example1.png

**Venn Diagram** widget can be also used for exploring different
prediction models. In the following example we analysed 3 prediction
methods, namely :doc:`Naive Bayes<../classify/naivebayes>`, :doc:`SVM Learner<../classify/svm>` and :doc:`Random Forest
Learner<../classify/randomforest>`, according to their misclassified instances. By selecting
misclassifications in the three :doc:`Confusion Matrix<../evaluation/confusionmatrix>` widgets and sending
them to Venn diagram, we can see all the misclassification instances
visualized per method used. Then we open **Venn Diagram** and select,
for example, the misclassified instances that were identified by all
three methods (in our case 2). This is represented as an intersection of
all three circles. Click on the intersection to see this two instanced
marked in the :doc:`Scatterplot<../visualize/scatterplot>` widget. Try selecting different diagram
sections to see how the scatterplot visualization changes.

.. figure:: images/VennDiagram-Example2.png
