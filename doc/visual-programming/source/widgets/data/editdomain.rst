Edit Domain
===========

Rename features and their values.

Inputs
    Data
        input dataset

Outputs
    Data
        dataset with edited domain


This widget can be used to edit/change a dataset's domain. 

.. figure:: images/EditDomain-stamped.png

1. All features (including meta attributes) from the input dataset are listed in the *Variables* list. Selecting one feature displays an editor on the right.
2. Change the name of the feature.
3. Change the value names for discrete features in the *Values* list box. Double-click to edit the name. To reorder the values (for example to display them in :doc:`Distributions <../visualize/distributions>`), use the up and down keys at the bottom of the box. To add or remove a value, use + and - buttons.
4. Additional feature annotations can be added, removed or edited in the *Labels* box. Add a new label with the + button and add the *Key* and *Value* for the new entry. Key will be displayed in the top left corner of the :doc:`Data Table <../data/datatable>`, while values will appear below the specified column. Remove an existing label with the - button.
5. To revert the changes made to the selected feature, press the *Reset Selected* button while the feature is selected in the *Variables* list. Pressing *Reset All* will remove all the changes to the domain.
6. Press *Apply* to send the new domain to the output.

Example
-------

Below, we demonstrate how to simply edit an existing domain. We selected the *heart_disease.tab* dataset and edited the *gender* attribute. Where in the original we had the values *female* and *male*, we changed it into *F* for female and *M* for male. Then we used the down key to switch the order of the variables. Finally, we added a label to mark that the attribute is binary. We can observe the edited data in the :doc:`Data Table <../data/datatable>` widget.

.. figure:: images/EditDomain-Example.png
