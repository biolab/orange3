Sieve Diagram
=============

![image](icons/sieve-diagram.png)

Plots a sieve diagram for a pair of attributes.

Signals
-------

**Inputs**:

- **Data**

  Input data set.

**Outputs**:

- None

Description
-----------

**Sieve diagram** is a graphical method for visualizing frequencies in
a two-way contingency table and comparing them to the [expected
frequencies](http://cnx.org/contents/d396c4ad-2fd7-47cd-be84-152b44880feb@2/What-is-an-expected-frequency)
under assumption of independence. The sieve diagram was
proposed by Riedwyl and Schüpbach in a technical report in 1983 and
later called a parquet diagram (Riedwyl and Schüpbach, 1994). In this display the
area of each rectangle is proportional to the expected frequency, while the
observed frequency is shown by the number of squares in each rectangle.
The difference between observed and expected frequency (proportional to
the standard Pearson residual) appears as the density of shading, using
color to indicate whether the deviation from independence is positive
(blue) or negative (red).

![image](images/SieveDiagram-stamped.png)

1. Select the attributes you want to display in the sieve plot.
2. Select an additional condition for the plot. This will show the selected two attributes only for the defined condition (e.g. attribute = survived, value = no).
3. Visual settings for your plot.
    - *Show squares (observed frequency)* displays horizontal and vertical lines representing observed frequency.
    - *Show data instances...* displayes dots in the plot that represent instances.
    - *...in color* adds the corresponding color to the dots.
4. *Save graph* saves the graph to your computer in a .svg or .png format.

The snapshot below shows a sieve diagram for *Titanic* data set and has
attributes *sex* and *survived* (the latter is a class attribute in
this data set). The plot shows that the two variables are highly
associated, as there are substantial differences between observed and
expected frequencies in all of the four quadrants. For example and as
highlighted in a balloon, the chance for surviving the accident was much higher
for female passengers than expected (0.06 vs. 0.15).

![image](images/SieveDiagram-Titanic.png)

Pairs of attributes with interesting associations are shown with shading 
the most interesting attribute pair in the *Titanic* data set, which is
indeed the one we show in the above snapshot. For contrast, a sieve
diagram of the least interesting pair (age vs. survival) is shown below.

![image](images/SieveDiagram-Titanic-age-survived.png)

Example
-------

Below we see a simple schema using a *Titanic* data set, where we use **Rank** widget to select the best attributes 
(the ones with the highest information gain, gain ratio or gini index) and feed them into **Sieve Diagram**. This
displays the sieve plot for the two best attributes, which in our case are sex and status. We see that the survival rate
on Titanic was very high for women of the first class and very low for female crew members. 

<img src="images/SieveDiagram-Example.png" alt="image" width="600">

References
----------

Riedwyl, H., and Schüpbach, M. (1994). Parquet diagram to plot contingency tables. In Softstat '93: Advances in Statistical Software, F. Faulbaum (Ed.). New York: Gustav Fischer, 293-299.
