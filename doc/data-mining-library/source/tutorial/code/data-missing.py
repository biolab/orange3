import Orange
import numpy as np

data = Orange.data.Table("voting.tab")

print("Percent missing by column:")
print(data.isnull().mean(axis=0) * 100)
