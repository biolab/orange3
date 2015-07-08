Hierarchical Clustering
=======================

![image]

Groups items using a hierarchical clustering algorithm.

Signals
-------

**Inputs**:

- **Distance Matrix**

  A matrix of distances between items being clustered

**Outputs**:

- **Selected Data**

  Data subset.

- **Other Data**

  Remaining data.

Description
-----------

The widget computes hierarchical clustering of arbitrary types of
objects from the matrix of distances between them and shows the
corresponding dendrogram.

![image][1]

1. The widget supports four kinds of linkages:
    - **Single linkage** is the distance between the closest elements of the two clusters
    - **Average linkage** computes the average distance between elements of the two clusters
    - **Weighted linkage** computes the weighted distance between elements of the two clusters
    - **Complete linkage** is the distance between clusters' most distant elements

2. Nodes of the dendrogram can be labeled. What the labels are depends upon
  the items being clustered. When clustering attributes, the
  labels are obviously the attribute names. When clustering instances, we
  can use the values of one of the attributes, typically the one that gives the
  name or the id of an instance, as labels. The label can be chosen in the box
  **Annotation**.

3. Huge dendrograms can be pruned in the *Pruning* box by
  selecting the maximum depth of the dendrogram. This only affects the displayed
  dendrogram, not the actual clustering.

4. Widget offers three different selection methods:
    - **Manual** (Clicking inside the dendrogram will select a cluster. Multiple clusters can be selected by
    holding Ctrl. Each selected cluster is shown in different color and is
    treated as a separate cluster in the output.)
    - **Height ratio** (Clicking on the bottom or top ruler of the dendrogram places a
    cutoff line in the graph. Items to the right of the line are selected.)
    - **Top N** (Selects the number of top nodes.)

5. If the items being clustered are instances, they can be added a cluster
  index (*Append cluster IDs*). The ID can appear as an ordinary **Attribute**,
  **Class attribute** or a **Meta attribute**. In the second
  case, if the data already has a class attribute, the original class is
  placed among meta attributes.

  The data can be automatically output on any change (*Auto send is on*) or, if the box
  isn't ticked, by pushing *Send Data*.


Examples
========

The schema below computes clustering of attributes and of examples.

![image]

We loaded the Zoo data set. The clustering of attributes is already
shown above. Below is the clustering of examples, that is, of animals,
and the nodes are annotated by the animals’ names. We connected the
Linear projection showing the freeviz-optimized projection of the data
so that it shows all examples read from the file, while the signal from
Hierarchical clustering is used as a subset. Linear projection thus
marks the examples selected in Hierarchical clustering. This way, we can
observe the position of the selected cluster(s) in the projection.

![image][1]

To (visually) test how well the clustering corresponds to the actual
classes in the data, we can tell the widget to show the class (“type”)
of the animal instead of its name (Annotate). Correspondence looks good.

![image][2]

A fancy way to verify the correspondence between the clustering and the
actual classes would be to compute the chi-square test between them. As
Orange does not have a dedicated widget for that, we can compute the
chi-square in Attribute Distance and observe it in Distance Map. The
only caveat is that Attribute Distance computes distances between
attributes and not the class and the attribute, so we have to use
Select attributes to put the class among the ordinary attributes and
replace it with another attribute, say “tail” (this is needed since
Attribute Distance requires data

  [image]: images/HierarchicalClustering-Schema.png
  [1]: images/HierarchicalClustering-Example.png
  [2]: images/HierarchicalClustering-Example2.png
