Curve Fit
=========

Fit a function to data.

**Inputs**

- Data: input dataset
- Preprocessor: preprocessing method(s)

**Outputs**

- Learner: curve fit learning algorithm
- Model: trained model
- Coefficients: fitted coefficients

The **Curve Fit** widget fits an arbitrary function to the input data. It only works for regression tasks.
The widget uses [scipy.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) to find the optimal values of the parameters.
