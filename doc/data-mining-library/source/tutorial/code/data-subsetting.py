import Orange

data = Orange.data.Table("iris.tab")
print("Data set instances:", len(data))
subset = data[data["petal length"] > 3]
print("Subset size:", len(subset))
