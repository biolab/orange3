Save Model
==========

Save a trained model to an output file.

**Inputs**

- Model: trained model

![](images/SaveModel-stamped.png)

1. Choose from previously saved models.
2. Save the created model with the *Browse* icon. Click on the icon and enter the name of the file. The model will be saved to a pickled file.
![](images/SaveModel-save.png)
3. Save the model.

Example
-------

When you want to save a custom-set model, feed the data to the model (e.g. [Logistic Regression](../model/logisticregression.md)) and connect it to **Save Model**. Name the model; load it later into workflows with [Load Model](../model/loadmodel.md). Datasets used with **Load Model** have to contain compatible attributes.

![](images/SaveModel-example.png)
