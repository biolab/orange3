Select Genes
============

![Select genes icon](icons/select-genes.png)

Manual selection of gene subset.

Signals
-------

**Inputs**:

- **Data**

  Data set.

- **Gene Subset**

  Data subset.

**Outputs**:

- **Selected Data**

  Data subset.

Description
-----------

**Select Genes** widget is used to manually create the gene subset. The user can decide which genes or gene sets 
will be used. The widget has two input channels: one is the standard *Data* channel that inputs a data sets from on of
the data widgets. The other channel is *Gene subset*, which feeds only a selected subset into the widget
(for example we can use **Differential Expression** widget to select only statistically significant genes). From the
input subset we can sort and select an even smaller subset that will allow for a detailed analysis.

On the other hand we might be interested in specific gene functions. We can go to *Select Specified Genes - 
Select Genes - Import names from gene sets*. Then we get a list of gene sets by category, name and the number
of genes in the set. This is how you can select genes by function - only those input genes that match the selected
function will be in the output (colored blue in the list).

You can also save a gene selection in *Saved selection* for the most frequently used genes.

![image]()


Example
-------

<img src="" alt="image" width="600">
