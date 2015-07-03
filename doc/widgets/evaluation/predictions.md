Predictions
===========

![image](icons/predictions.png)

Shows the classifiers' predictions on data.

Signals
-------

**Inputs**:

- **Data**

  Data set.

- **Predictors**

  Predictors to be used on the data.

**Outputs**:

- **Predictions**

  Original data with added predictions.

Description
-----------

The widget receives a data set and one or more predictors (classifiers, not
learning algorithms - see the example below). It outputs the
data and the predictions.

Despite its simplicity, the widget allows for quite an interesting analysis
of decisions of predictive models; there is a simple demonstration at
the bottom of the page. **Confusion Matrix** is a related widget and
although many things can be done with any of them,
there are tasks for which one of them might be much more convenient than
the other.

![image][1]

The output of the widget is another data set, where predictions are
appended as new meta attributes. You can select which features you wish to output
(original data, predictions, probabilities).

Example
-------

Here is a demonstration.

![image][2]

First, compare the schema with the one for **Test Learners**. Widgets
representing learning algorithms, like **Naive Bayes** or
**Classification Tree** provide two kinds of signals, one with a learning
algorithm and one with a classifier. The learner is always available,
while for outputting a classifier, the classification widget needs to be given some data.

**Test Learners** tests learning algorithms, hence it expects learning
algorithms in the input. In the corresponding schema, we gave the Test
Learners some data from the **File** widget and a few "learner widgets".
Widget **Predictions** shows predictions of a classifier, hence it needs a
classifier and the data.

This is what we do: we randomly split the data into two subsets.
The larger subset, containing 70 % of data instances, is sent to *Naive Bayes*
and **Classification Tree**, so they can produce the corresponding
classifiers. Classifiers are then sent into **Predictions**, among with the
remaining 30 % of the data. Predictions shows how these examples are
classified.

The results of this procedure on the *heart disease* data are shown in the
**Data Table** snapshot. The last seven columns are (from right to left) the actual
class, and the predictions by the classification tree and naive Bayesian
classifier. For the latter two we see probabilities of class "1", class "0"
and the predicted class. Probabilities are shown
as separate attributes, so we can sort instances by these values.

To save the predictions, we simply attach the **Save** widget to
**Predictions**. The final file is a data table and can be saved as
a .tab or .csv format.

Finally, we can analyze the classifier’s predictions. For instance, we
want to observe the relations between probabilities predicted by the two
classifiers with respect to the class. For that, we first take
**Select Columns** with which we move the meta attributes with
probability predictions to features. The transformed data is
then given to the **Scatterplot**, which we set to use the attributes with
probabilities as the x and y axes, and the class is (already by
default) used to color the data points.

![image]

To get the above plot, we selected *Jitter continuous values**,
since the classification tree gives just a few distinct probabilities,
hence without jittering there would be too much overlap between the
points.

The blue points in the bottom left corner represent the people with no diameter
narrowing, which were correctly classified by both classifiers. The
upper right red points represent the patients with narrowed vessels,
which were correctly classified by both. At the top left there are a few
blue points: these are those without the narrowed vessels to whom the tree
gave a high probability of having the disease, while Bayesian classifier
was right by predicting a low probability of the disease. In the
opposite corner, the red points are the sick, to which the
tree gave a low probability, while the naive Bayesian classifier was
(again) right by assigning a high probability of having the disease.

Note that this analysis is done on a rather small sample, so these
conclusions may be ungrounded.

Here is the entire schema:

<img src="" alt="image" width="600">

Another example of using this widget is given in the documentation for
widget **Confusion Matrix**.

