SVM
===

![image](icons/svm-classification.png)

Support vector machine learning algorithm

Signals
-------

**Inputs**:

- **Data**

  Data set.
  
- **Preprocessor**

  Preprocessed data.

**Outputs**:

- **Learner**

  Support vector machine learning algorithm with settings as specified in the dialog.

- **Classifier**

  Trained SVM classifier

- **Support Vectors**

  A subset of data instances from the training set that were used as support vectors in the trained classifier

Description
-----------

[**Support vector machine**](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM) is a classification technique that
separates the attribute space with a hyperplane, thus
maximizing the margin between the instances of different classes. The
technique often yields supreme predictive performance results. Orange
embeds a popular implementation of SVM from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) package. This
widget is its graphical user interface.

![Support vector machines widget](images/SVM-new-stamped.png)

1. Learner can be given a name under which it will appear in other widgets. The default name is
“SVM Learner”.
2. Classification type with test error settings. *C-SVM* and *v-SVM* are based
  on different minimization of the error function. On the right side you can set test error bounds,
  [*Cost*](http://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine) for C-SVM and *Complexity bound* for v-SVM.
3. The next block of options deals with kernel, a function that
  transforms attribute space to a new feature space to fit the
  maximum-margin hyperplane, thus allowing the algorithm to create
  non-linear classifiers with [*Polynomial*](https://en.wikipedia.org/wiki/Polynomial_kernel), [*RBF*](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) and [*Sigmoid*](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#sigmoid) kernels. Functions that specify the
  kernel are presented besides their names, and the constants involved are:
    - **g** for the gamma constant in kernel function (the recommended value is
    1/k, where k is the number of the attributes, but since there may be no
    training set given to the widget the default is 0 and the user has to
    set this option manually),
    - **c** for the constant c0 in the kernel function (default 0), and
    - **d** for the degree of the kernel (default 3).
4. Set permitted deviation from the expected value in *Numerical Tolerance*. Tick the box next to *Iteration Limit* to set the maximum number of iterations permitted.
5. Click *Apply* to commit changes.

Examples
--------

There are two typical uses for this widget, one where the widget is a
classifier and the other where it constructs an object for
learning. For the first one, we have split the data set into two data subsets
(*Sample* and *Remaining Examples*). The sample was sent to SVM which
produced a *Classifier*, that was then used in **Predictions** widget to
classify the data in *Remaning Examples*. A similar schema can be used if
the data would be already separated in two different files; in this
case, two **File** widgets would be used instead of the **File** - **Data Sampler**
combination.

<img src="images/SVM-Predictions.png" alt="image" width="600">

The second schema shows how to use the **SVM** widget to construct the
learner and compare it in cross-validation with **Majority** and
**k-Nearest Neighbours** learners.

![SVM and other learners compared by cross-validation](images/SVM-Evaluation.png)

The following schema observes a set of support vectors in a **Scatterplot**
visualization.

![Visualization of support vectors](images/SVM-with-support-vectors.png)

For the above schema to work correctly, the channel between **SVM** and
**Scatterplot** widget has to be set appropriately. Set the channel between
these two widgets by double-clinking on the connection between the
widgets and use the settings as displayed in the dialog below.

![Channel setting for communication of support vectors](images/SVM-support-vectors.png)

References
----------

[Introduction to SVM on StatSoft.](http://www.statsoft.com/Textbook/Support-Vector-Machines)
