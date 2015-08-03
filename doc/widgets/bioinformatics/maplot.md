MA Plot
=======

![image](icons/ma-plot.png)

Visualization of intensity-dependent ratio of raw microarray data.

Signals
-------

**Inputs**:

- **Expression Array**

  DNA microarray.

**Outputs**:

- **Normalized Expression Array**

  Lowess-normalized microarray.

- **Filtered Expression Array**

  Selected instances (in the Z-score cutoff).

Description
-----------

[**MA Plot**](https://en.wikipedia.org/wiki/MA_plot) is a graphical method for visualizing intensity-dependent
ratio of raw mircoarray data. The A represents the average log intensity of the gene
expression (x-axis in the plot), while M stands for the binary log of intensity ratio (y-axis). The widget
outputs either normalized data (Lowess normalization method) or instances above the Z-score cutoff line (instances
with meaningful fold changes).

![image](https://github.com/ajdapretnar/orange3/blob/widget-documentation/doc/widgets/bioinformatics/images/MAplot5-stamped.png)

1. Information on the input data.
2. Select the attribute to split the plot by.
3. Center the plot using:
   - **average**
   - [**Lowess (fast-interpolated)**](https://en.wikipedia.org/wiki/Local_regression) normalization method
   - **Lowess** normalization method
4. Merge replicated by:
   - **average**
   - **median**
   - **geometric mean**
5. Set the **Z-score cutoff** threshold. Z-score is your confidence interval and it is set to
   95% by default. If the widget is set to output *filtered expression array*, instances above the
   [Z-score](https://en.wikipedia.org/wiki/Standard_score) threshold will be in the output (red dots in the plot).
6. Ticking the *Append Z-scores* will add an additional meta attribute with Z-scores to your output data.<br>
   Ticking the *Append Log ratio and Intensity values* will add two additional meta attributes with M and A values
   to your output data.
7. If *Auto commit is on*, the widget will automatically apply changes to the output. Alternatively click *Commit*.

Example
-------

Below you can see an example workflow for Heat Map widget. Heat map below displays attribute values
for *Zoo* data set (0 is white, 1 is light blue, >1 is dark blue). The first thing we see in the map is
'legs' attribute which is the only one colored in dark blue. In order to get a clearer heat map,
we then use **Select Columns** widget and remove 'legs' attribute from the data set. Then we again
feed the data to the **Heat Map**.

The new projections is much clearer. By removing 'legs' we get a neat visualization of attribute
values for each class. We see that mammals typically have hair, teeth, backbone and milk, while birds
have feathers, eggs and a tail.

Additionally we would like to see why 'legs' attribute was so pronounced in the first heat map.
We again use **Select Columns** widget to feed only this attribute into the **Data Table**. We already
see that this attribute has values different than 0 or 1 - animal either have 2 or 4 legs or none at all.
But as there were two classes represented by a very dark blue, namely invertebrates and insects, we wish
to inspect this further. We sort the table by type and look at invertebrates for example. We see that
this class has 0, 4, 5, 6 or even 8 legs, which is why it was a good idea to remove it from the
heat map visualization as an 'outlying' attribute.

![image](images/HeatMap-new1.png)

![image](images/HeatMap-new4.png)
