Color
=====

Set color legend for variables.

**Inputs**

- Data: input data set

**Outputs**

- Data: data set with new colors

With the **Color** widget it is possible to change the colors and color palettes for visualizations.

![](images/Color-stamped.png)

1. A list of discrete variables. Set the color of each variable by double-clicking on it. The widget also enables renaming variables by clicking on their names.
2. A list of continuous variables. Click on the color strip to choose a different palette. To use the same palette for all variables, change it for one variable and click *Copy to all* that appears on the right. The widget also enables renaming variables by clicking on their names.
3. **Save**, **load** or **reset** the colors.
4. **Apply** changes by pressing the button or tick the checkbox to do so automatically.
5. Produce a report.

Palettes for numeric variables are grouped and tagged by their properties.

![](images/Color-Continuous_unindexed.png)

- **Linear** palettes are constructed so that human perception of the color change is linear with the change of the value.

- **Diverging** palettes have two colors on its ends and a central color (white or black) in the middle. Such palettes are particularly useful when the values can be positive or negative, as some widgets (for instance the Heat map) will put the 0 at the middle point in the palette.

- **Color-blind friendly** palettes cover different types of color blindness, and can also be linear or diverging.

- In the **Other** section you find and isoluminant palette, where all colors have equal brightness and the Rainbow palette which is particularly nice in visualizations that bin numeric values.

Example
-------

Let's load the _heart_disease_ data set. Then we  selected two new colors for the _diameter narrowing_ variable in the **Colors** widget. Finally, we can add the **[Scatter Plot](../visualize/scatterplot.md)** widget and inspect the color changes while coloring the data points by _diameter narrowing_.

![](images/Color-Example-Discrete.png)

To see the effect of color palettes for numeric variables, we can color the points in the scatter plot by _cholesterol_ and change the palette for this attribute in the **Color** widget to _Rainbow_.

![](images/Color-Example-Continuous.png)
