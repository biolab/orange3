import Orange

data = Orange.data.Table("iris.tab")
new_domain = Orange.data.Domain(list(data.domain.attributes[:2]), data.domain.class_var)
new_data = Orange.data.Table(new_domain, data)

print(data.iloc[0])
print(new_data.iloc[0])
