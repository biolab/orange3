CN2 Rule Induction
==================

Induce rules from data using CN2 algorithm.

Inputs
    Data
        input dataset
    Preprocessor
        preprocessing method(s)

Outputs
    Learner
        CN2 learning algorithm 
    CN2 Rule Classifier
        trained model


The CN2 algorithm is a classification technique designed for the
efficient induction of simple, comprehensible rules of form "if *cond*
then predict *class*", even in domains where noise may be present.

**CN2 Rule Induction** works only for classification.

.. figure:: images/CN2-stamped.png
   :scale: 50%

1. Name under which the learner appears in other widgets. The default
   name is *CN2 Rule Induction*.

2. *Rule ordering*:

   - **Ordered**: induce ordered rules (decision list). Rule
     conditions are found and the majority class is assigned in the
     rule head.
   - **Unordered**: induce unordered rules (rule set). Learn rules for
     each class individually, in regard to the original learning
     data.

3. *Covering algorithm*:

   - **Exclusive**: after covering a learning instance, remove it from
     further consideration.
   - **Weighted**: after covering a learning instance, decrease its
     weight (multiplication by *gamma*) and in-turn decrease its
     impact on further iterations of the algorithm.


4. *Rule search*:

   - **Evaluation measure**: select a heuristic to evaluate found
     hypotheses:

     a. `Entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ (measure of unpredictability of content)
     b. `Laplace Accuracy <https://en.wikipedia.org/wiki/Laplace%27s_method>`_
     c. Weighted Relative Accuracy

   - **Beam width**; remember the best rule found thus far and monitor a
     fixed number of alternatives (the beam).

5. *Rule filtering*:

   - **Minimum rule coverage**: found rules must cover at least the
     minimum required number of covered examples. Unordered rules must
     cover this many target class examples.

   - **Maximum rule length**: found rules may combine at most the
     maximum allowed number of selectors (conditions).

   - **Default alpha**: significance testing to prune out most
     specialised (less frequently applicable) rules in regard to the
     initial distribution of classes.

   - **Parent alpha**: significance testing to prune out most
     specialised (less frequently applicable) rules in regard to the
     parent class distribution.

6. Tick 'Apply Automatically' to auto-communicate changes to other
   widgets and to immediately train the classifier if learning data is
   connected. Alternatively, press ‘Apply‘ after configuration.

Examples
--------

For the example below, we have used *zoo* dataset and passed it to **CN2 Rule Induction**. We can review and interpret the built model with :doc:`CN2 Rule Viewer <../visualize/cn2ruleviewer>` widget.

.. figure:: images/CN2-visualize.png

The second workflow tests evaluates **CN2 Rule Induction** and :doc:`Tree <../model/tree>` in :doc:`Test & Score <../evaluation/testandscore>`.

.. figure:: images/CN2-classification.png

References
----------

1. "Separate-and-Conquer Rule Learning", Johannes Fürnkranz, Artificial
   Intelligence Review 13, 3-54, 1999

2. "The CN2 Induction Algorithm", Peter Clark and Tim Niblett,
   Machine Learning Journal, 3 (4), pp261-283, (1989)

3. "Rule Induction with CN2: Some Recent Improvements", Peter Clark and
   Robin Boswell, Machine Learning - Proceedings of the 5th
   European Conference (EWSL-91), pp151-163, 1991

4. "Subgroup Discovery with CN2-SD", Nada Lavrač et al., Journal of
   Machine Learning Research 5 (2004), 153-188, 2004
