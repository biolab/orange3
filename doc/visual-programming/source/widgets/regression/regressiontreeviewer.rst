Regression Tree Viewer
==========================

.. figure:: icons/regression-tree-viewer.png

Visualization of a regression tree.

Signals
-------

**Inputs**:

-  **Regression Tree**

   Regression tree

**Outputs**:

-  **Data**

   Data from a selected tree node

Description
-----------

This is a versatile widget with 2-D visualization of a `regression tree`_. The user can select a node, instructing the widget to output the data associated with the node, thus enabling explorative data analysis.

.. figure:: images/RegressionTreeViewer-stamped.png

1. Information on the input.

2. Set the zoom and define the tree width. The nodes display tooltips when hovering over them.

3. The edges between the nodes in the tree graph are drawn based on the selected
   edge width.

   -  All the edges will be of equal width if *Fixed* is chosen.
   -  When *Relative to root* is selected, the width of the edge will correspond to the proportion of instances in the corresponding node with respect to all the instances in the training data. Under this selection, the edge will get thinner and thinner when traversing toward the bottom of the tree.
   -  *Relative to parent* makes the edge width correspond to the proportion of instances in the nodes with respect to the instances in their parent node.

4. The nodes box defines the target class, which you can change based on
   the classes in the data. You can also use *Set Colors*  and color the tree according to the number of instances in a node or impurity. You can also choose to keep the default coloring. 

5. Press *Save Graph* to save the regression tree graph as a file to your computer in a .svg or .png format. 

6. Regression tree. 

Examples
--------

Below, is a simple schema, where we have read the data, constructed the regression tree and viewed it in our tree viewer. We loaded the *Housing* data set and limited the depth of the tree to only 4 levels because of the vastness of the data set. It is worth remembering that if both the viewer and :doc:`Regression Tree <../regression/regressiontree>` are open, any run of the tree induction algorithm will immediately affect the visualization. You can thus use this combination to explore how parameters of the induction algorithm influence the structure of the resulting tree.

.. figure:: images/Regression-Tree-Example1.png

Clicking on any node will output the related data instances. This is explored in the :doc:`Scatterplot <../visualize/scatterplot>`. Make sure that the tree data is passed as a data subset; this can be done by connecting the :doc:`Scatterplot <../visualize/scatterplot>` to the :doc:`File <../data/file>` widget first, and connecting it to the **Tree Viewer** widget next.

.. figure:: images/Regression-Tree-Example2.png


.. _regression tree: https://en.wikipedia.org/wiki/Decision_tree_learning
