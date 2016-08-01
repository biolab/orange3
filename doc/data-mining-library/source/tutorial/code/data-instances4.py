import Orange
from collections import Counter

data = Orange.data.Table("lenses")
print(Counter(str(r[r.domain.class_var]) for idx, r in data.iterrows()))
