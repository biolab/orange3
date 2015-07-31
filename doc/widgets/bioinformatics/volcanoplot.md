Volcano Plot
============

![image](icons/volcano-plot.png)

Plots significance versus fold-change for gene expression rates.

Signals
-------

**Inputs**:

- **Data**

  Input data set.

**Outputs**:

- **Selected data**

  Data subset.

Description
-----------

[**Volcano plot**](https://en.wikipedia.org/wiki/Volcano_plot_(statistics)) is a graphical method for 
visualizing changes in replicate data. The widget plots a binary logarithm of fold-change on x-axis versus
[statistical significance](https://en.wikipedia.org/wiki/Statistical_significance) 
(negative base 10 logarithm of p-value) on the y-axis. 

**Volcano Plot** is useful for a quick visual identification of statistically significant
data (genes). Genes that are highly dysregulated are
farther to the left and right, while highly significant fold changes appear higher on the plot.
A combination of the two are those genes that are statistically significant - the widget
automatically selects the top-ranking genes within the top right and left fields and outputs them.

![image](images/HeatMap-new2.png)

1. Information on the input data
2. Choose x attribute
3. Choose y attribute
4. Discrete attribute for color scheme
5. Color scheme legend. You can select which attribute instances you wish to see in the visualization.
6. Select the color scale strength (linear, square root or logarithmic)
7. To move the map use *Drag* and to select data subset use *Select*
8. Visualization

Example
-------

Below you can see a simple workflow for **Volcano Plot**. We use *Caffeine effect: time course and dose
response* data from **GEO Data Sets** widget and visualize them in a **Data Table**. We have
6378 gene in the input, so it is essential to prune the data and analyse only those genes
that are statistically significant. **Volcano Plot** helps us do exactly that. Once the
desired area is selected in the plot, we output the data and observe them in another **Data Table**.
Now we get only 4 instances, which were those genes that had a high normalized fold change under
high dose of caffeine and had a low p-value at the same time.

<img src="" alt="image" width="600">
