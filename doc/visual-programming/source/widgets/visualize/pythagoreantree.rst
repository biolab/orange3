Pythagorean Tree
================

.. figure:: icons/pythagorean-tree.png

Pythagorean tree for visualising classification trees.

Signals
-------

**Inputs**:

-  **Tree**

A classification / regression tree model.

**Selected Data**:

-  A subset of instances that the user has manually selected from the
pythagorean tree.

Description
-----------

**Pythagorean Trees** are fractals that can be used to depict binary hierarchies as presented in an article by `Beck F. et al<http://publications.fbeck.com/ivapp14-pythagoras.pdf>`__. In our case, they're used for concisely visualizing tree models, such as :doc:`Classification Tree<../classify/classificationtree>` or :doc:`Regression Tree<../regression/regressiontree>`. 

.. figure:: images/Pythagorean-Tree1-stamped.png

1. Information on the input tree model.

2. Parameters for display settings:

    - _Depth_: set the depth to which the trees are grown.
    - _Target class_: set the target class for coloring the trees. Nodes where the selected class if present will be colored by the class color. If _None_ is selected, the tree will be white.
    - _Size_: set the size of the nodes. _Normal_ will keep nodes the size of the subset in the node. _Square root_ and _Logarithmic_ are the respective transformations of the node size.
    - _Log scale factor_ is only enabled when _logarithmic_ transformation is selected. You can set the log factor between 1 and 10.

3. Plot properties:

    - _Enable tooltips_: information on the node as displayed upon hovering.
    - _Show legend_: shows color legend for the plot.

4. _Save Image_: save the visualization to your computer as a *.svg* or *.png* file. 
   _Report_: produce a report.

Pythagorean Tree can visualize both classification and regression trees. Below is an example for regression tree. The only difference between the two is that regression tree doesn't enable coloring by class, but can color by class mean or standard deviation.


.. figure:: images/Pythagorean-Tree1-continuous.png

Example
-------

In this example we demonstrate the difference between :doc:`Classification Tree Viewer<../classify/classificationtreeviewer>` and **Pythagorean Tree**. They both visualize :doc:`Classification Tree<../classify/classificationtree>` algorithm, but Pythagorean tree is much more concise, even for a small data set such as _Iris.tab_.

.. figure:: images/Pythagorean-Tree-comparison.png

References
----------

Beck, F., Burch, M., Munz, T., Di Silvestro, L. and Weiskopf, D. (2014). Generalized Pythagoras Trees for Visualizing Hierarchies. In IVAPP '14 Proceedings of the 5th International Conference on Information Visualization Theory and Applications, 17-28.
