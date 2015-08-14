Select Rows
===========

![image](icons/select-rows.png)

Selects data instances based on conditions over data features.

Signals
-------

**Inputs**:

- **Data**

  Data set.

**Outputs**:

- **Matching Data**

  Instances that match the conditions.

- **Non-Matching Data**

  Instances that do not match the conditions.

Description
-----------

This widget selects a subset from the input data based on user-defined conditions. 
Instances that match the selection rule are placed in the output *Matching Data* channel.

Criteria for data selection are presented as a collection of conjuncted terms (i.e. selected items are those
matching all the terms in '*Conditions*').

Condition terms are defined through selecting of an attribute, selecting
an operator from the list of operators,
and, if needed, defining the value to be used in the condition term.
Operators are different for discrete, continuous and string attributes.

![Select Data](images/SelectRows-stamped.png)

1. Conditions you want to apply, their operators and related values.
2. Add a new condition to the list of conditions.
3. Add all the possible variables at once.
4. Remove all the listed variables at once.
5. Information on the input data set.
6. Information on instances that match the condition(s).
7. Purge the output data.
8. When the 'Commit all change' box is ticked, all changes will be automatically communicated to other widgets.

Any change in composition of the condition will update the information pane (*Data Out*).

If *Commit on change* is selected, then the output is updated on any change 
in the composition of the condition or any of its terms.

Example
-------

In the workflow below we used *Zoo* data from the **File** widget and fed it
into the **Select Rows**. 
In the widget we chose to output only two animal types (namely fish and reptiles). 
We can inspect both the original data set and the data set with selected rows in the 
**Data Table** widget.

<img src="images/SelectRows-Example.png" alt="image" width="600">

In the next example we used the car data from *imports-85* data set and similarly
fed it into the **Box Plot** widget. We first observe the entire data set based on
the engine type. Then we selected only diesel cars in **Select Rows** widget and
feed it again into the **Box Plot**. There we can see all the imported diesel cars 
listed by brand.

<img src="images/SelectRows-Workflow.png" alt="image" width="600">
