import numpy as np
import sklearn.lda as skl_lda

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.preprocess import Impute, Continuize
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Default


class LDA(OWWidget):
    name = "LDA"
    description = "LDA optimization of linear projections."
    icon = "icons/XXX.svg"

    inputs = [("Data", Table, "set_data", Default)]
    outputs = [("Transformed data", Table, Default),
               ("Components", Table)]

    want_main_area = False
    resizing_enabled = False

    learner_name = settings.Setting("LDA")

    def __init__(self):
        super().__init__(self)

        self.data = None

        gui.lineEdit(gui.widgetBox(self.controlArea, self.tr("Name")),
                     self, "learner_name")
        gui.rubber(self.controlArea)
        gui.button(self.controlArea, self, self.tr("&Apply"),
                   callback=self.apply)

    def set_data(self, data):
        self.error(0)
        if data and not data.domain.has_discrete_class:
            self.error(0, 'Data with a discrete class variable expected.')
            data = None
        self.data = data

    def apply(self):
        transformed = components = None
        if self.data:
            self.data = Continuize(Impute(self.data))
            lda = skl_lda.LDA(solver='eigen', n_components=2)
            X = lda.fit_transform(self.data.X, self.data.Y)
            dom = Domain([ContinuousVariable('Component_1'),
                          ContinuousVariable('Component_2')],
                         self.data.domain.class_vars, self.data.domain.metas)
            transformed = Table(dom, X, self.data.Y, self.data.metas)
            transformed.name = self.data.name + ' (LDA)'
            dom = Domain(self.data.domain.attributes,
                         metas=[StringVariable(name='component')])
            metas = np.array([['Component_{}'.format(i + 1)
                                  for i in range(lda.scalings_.shape[1])]],
                                dtype=object).T
            components = Table(dom, lda.scalings_.T, metas=metas)
            components.name = 'components'

        self.send("Transformed data", transformed)
        self.send("Components", components)
