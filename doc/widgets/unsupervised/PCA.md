PCA
===

![PCA](icons/pca.png)

PCA linear transformation of input data.

Signals
-------

**Inputs**:

- **Data**

  Data set.

**Outputs**:

- **Transformed Data**

  PCA transformed input data.

- **Components**

  [Eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors).

Description
-----------

[Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) computes the PCA linear
transformation of the input data. It outputs either a transformed data set with weights of individual instances or
weights of principal components.

![PCA widget](images/PCA-stamped.png)

1. Select how many principal components you wish in your output. It is best to choose as few as possible with variance     
   covered as high as possible. You can also set how much variance you wish to cover with your principal components.
2. When *Auto send on change* is ticked, the widget will automatically communicate all the changes. Alternatively click
   *Send data*.
3. Principal components graph, where the red (lower) line is the variance covered per component and the green (upper)
   line is cumulative variance covered by components.

The number of components of the transformation can be selected
either in the *Components Selection* input box or by dragging the vertical
cutoff line in the graph.

Examples
--------

**PCA** can be used to simplify visualizations of large data sets. Below we
used *Iris* data set to show how we can improve the visualization of the data set with PCA. The transformed data in a **Scatter plot** show a much clearer distinction between classes than the default settings.

<img src="images/PCAExample.png" alt="image" width="600">

The widget provides two outputs: transformed data and principal components. Transformed 
data are weights for individual instances in the new coordinate system, while components are the system descriptors
(weights for princial components).
When fed into the **Data Table**, we can see both outputs in a numerical form.

<img src="images/PCAExample2.png" alt="image" width="600">
