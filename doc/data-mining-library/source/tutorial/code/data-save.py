import Orange
data = Orange.data.Table("lenses")
myope_subset = data[data["prescription"] == "myope"]
new_data = Orange.data.Table(data.domain, myope_subset)
new_data.save("lenses-subset.tab")
