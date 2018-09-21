Predictions
===========

Shows models' predictions on the data.

Inputs
    Data
        input dataset
    Predictors
        predictors to be used on the data

Outputs
    Predictions
        data with added predictions
    Evaluation Results
        results of testing classification algorithms


The widget receives a dataset and one or more predictors (predictive models, not learning algorithms - see the example below). It outputs the data and the predictions.

.. figure:: images/Predictions-stamped.png

1. Information on the input, namely the number of instances to predict, the number of predictors and the task (classification or regression). If you have sorted the data table by attribute and you wish to see the original view, press *Restore Original Order*.
2. You can select the options for classification. If *Predicted class* is ticked, the view provides information on predicted class. If *Predicted probabilities for* is ticked, the view provides information on probabilities predicted by the classifier(s). You can also select the predicted class displayed in the view. The option *Draw distribution bars* provides a visualization of probabilities. 
3. By ticking the *Show full dataset*, you can view the entire data table (otherwise only class variable will be shown). 
4. Select the desired output.
5. Predictions.

The widget show the probabilites and final decisions of `predictive models <https://en.wikipedia.org/wiki/Predictive_modelling>`_. The output of the widget is another dataset, where predictions are appended as new meta attributes. You can select which features you wish to output (original data, predictions, probabilities). The result can be observed in a :doc:`Data Table <../data/datatable>`. If the predicted data includes true class values, the result of prediction can also be observed in a :doc:`Confusion Matrix <../evaluation/confusionmatrix>`.

Examples
--------

In the first example, we will use *Attrition - Train* data from the :doc:`Datasets <../data/datasets>` widget. This is a data on attrition of employees. In other words, we wish to know whether a certain employee will resign from the job or not. We will construct a predictive model with the :doc:`Tree <../model/tree>` widget and observe probabilites in **Predictions**.

For predictions we need both the training data, which we have loaded in the first **Datasets** widget and the data to predict, which we will load in another :doc:`Datasets <../data/datasets>` widget. We will use *Attrition - Predict* data this time. Connect the second data set to **Predictions**. Now we can see predictions for the three data instances from the second data set.

The :doc:`Tree <../model/tree>` model predicts none of the employees will leave the company. You can try other model and see if predictions change. Or test the predictive scores first in the :doc:`Test & Score <../evaluation/testandscore>` widget.

.. figure:: images/Predictions-Example1.png

In the second example, we will see how to properly use :doc:`Preprocess <../data/preprocess>` with **Predictions** or :doc:`Test & Score <../evaluation/testandscore>`.

This time we are using the *heart_disease.tab* data from the :doc:`File <../data/file>` widget. You can access the data through the dropdown menu. This is a dataset with 303 patients that came to the doctor suffering from a chest pain. After the tests were done, some patients were found to have diameter narrowing and others did not (this is our class variable).

The heart disease data have some missing values and we wish to account for that. First, we will split the data set into train and test data with :doc:`Data Sampler <../data/datasampler>`.

Then we will send the *Data Sample* into :doc:`Preprocess <../data/preprocess>`. We will use *Impute Missing Values*, but you can try any combination of preprocessors on your data. We will send preprocessed data to :doc:`Logistic Regression <../model/logisticregression>` and the constructed model to **Predictions**.

Finally, **Predictions** also needs the data to predict on. We will use the output of :doc:`Data Sampler <../data/datasampler>` for prediction, but this time not the *Data Sample*, but the *Remaining Data*, this is the data that wasn't used for training the model.

Notice how we send the remaning data directly to **Predictions** without applying any preprocessing. This is because Orange handles preprocessing on new data internally to prevent any errors in the model construction. The exact same preprocessor that was used on the training data will be used for predictions. The same process applies to :doc:`Test & Score <../evaluation/testandscore>`.

.. figure:: images/Predictions-Example2.png
