Heat Map
=============

![image](icons/heat-map.png)

Plots a heat map for a pair of attributes.

Signals
-------

**Inputs**:

- **Data**

  Input data set.

**Outputs**:

- None

Description
-----------

[**Heat map**](https://en.wikipedia.org/wiki/Heat_map) is a graphical method for visualizing attribute values
by class in a two-way matrix. Values are represented by color: the higher a certain value is,
the darker the represented color. By combining class and attributes on x and y axes we see where the attribute
values are the strongest and where the weakest, thus enabling us to find typical features (discrete) or value range 
(continuous) for each class.

![image]()

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

![image]()

![image]()
