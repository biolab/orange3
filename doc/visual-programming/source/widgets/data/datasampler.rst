Data Sampler
============

Selects a subset of data instances from an input dataset.

Inputs
    Data
        input dataset

Outputs
    Data Sample
        sampled data instances
    Remaining Data
        out-of-sample data


The **Data Sampler** widget implements several means of sampling data from
an input channel. It outputs a sampled and a complementary
dataset (with instances from the input set that are not included in the
sampled dataset). The output is processed after the input dataset is
provided and *Sample Data* is pressed.

.. figure:: images/DataSampler-stamped.png

1. Information on the input and output dataset
2. The desired sampling method:

   -  **Fixed proportion of data** returns a selected percentage of the
      entire data (e.g. 70% of all the data)
   -  **Fixed sample size** returns a selected number of data instances
      with a chance to set *Sample with replacement*, which always samples
      from the entire dataset (does not subtract instances already in
      the subset). With replacement, you can generate more instances than
      available in the input dataset.
   -  `Cross Validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_
      partitions data instances into complementary subsets, where you can
      select the number of folds (subsets) and which fold you want to
      use as a sample. 

3. *Replicable sampling* maintains sampling patterns that can be carried
   across users, while *stratification* mimics the composition of the
   input dataset.
4. Produce a report.
5. Press *Sample data* to output the data sample.
 

Examples
--------

First, let's see how the **Data Sampler** works. Let's look at the
information on the original dataset in the :doc:`Data Info <../data/datainfo>` widget. We see
there are 24 instances in the data (we used *lenses.tab*). We sampled
the data with the **Data Sampler** widget and we chose to go with a fixed
sample size of 5 instances for simplicity. We can observe the sampled
data in the :doc:`Data Table <../data/datatable>` widget. The second :doc:`Data Table <../data/datatable>` shows the
remaining 19 instances that weren't in the sample.

.. figure:: images/DataSampler-Example1.png 

In the workflow below, we have sampled 10 data instances from the *Iris*
dataset and sent the original data and the sample to :doc:`Scatter Plot <../visualize/scatterplot>`
widget for exploratory data analysis. The sampled data instances are plotted
with filled circles, while the original dataset is represented with
empty circles.

.. figure:: images/DataSampler-Example.png

**Data Sampler** can also be used to oversample a minority class in data. To do this, first separate the minority
class using a **Select Rows** widget. Connect Matching rows into a Data Samples widget, select Fixed sample size,
Sample with replacement and specify the number of instances you want to have in the resulting dataset.

.. figure:: images/DataSampler-Oversample1.png

Finally, concatenate the oversampled instances with the Unmatching Data output of the Select Rows widget.

.. figure:: images/DataSampler-Oversample2.png
