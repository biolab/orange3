import Orange

data = Orange.data.Table("imports-85.tab")

print("First attribute:", data.domain[0].name)
name = "fuel-type"
print("Values of attribute '%s': %s" %
      (name, ", ".join(data.domain[name].values)))