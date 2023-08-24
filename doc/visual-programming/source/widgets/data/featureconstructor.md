Formula
=======

Add new features to your dataset.

**Inputs**

- Data: input dataset

**Outputs**

- Data: dataset with additional features

**Formula** allows computing new columns by combining the existing ones with a user-defined expression. The resulting column can be categorical, numerical or textual.

For numeric variables, it sufices to provide a name and an expression.

![](images/feature-constructor1-stamped.png)

1. List of constructed variables
2. Add or remove variables
3. New feature name
4. Expression in Python
5. Select a feature
6. Select a function
7. Produce a report
8. Press *Send* to communicate changes

The following example shows construction of a categorical variable: its value is "lower" is "sepal length" is below 6, "mid" if it is at least 6 but below 7, and "higher" otherwise. Note that spaces need to be replaced by underscores (`sepal_length`).

![](images/feature-constructor2-stamped.png)

1. List of variable definitions
2. Add or remove variables
3. New feature name
4. Expression in Python
5. If checked, the feature is put among meta attributes
6. Select a feature to use in expression
7. Select a function to use in expression
8. Optional list of values, used to define their order
9. Press *Send* to compute and output data

Hints
-----

If you are unfamiliar with Python math language, here's a quick introduction.

Expressions can use the following operators:
- `+`, `-`, `*`, `/`: addition, subtraction, multiplication, division
- `//`: integer division
- `%`: remainder after integer division
- `**`: exponentiation (for square root square by 0.5)
- `<`, `>`, `<=`, `>=` less than, greater than, less or equal, greater or equal
- `==` equal
- `!=` not equal
- if-else: *value* `if` *condition* else *other-value* (see the above example

See more [here](http://www.tutorialspoint.com/python/python_basic_operators.htm).
