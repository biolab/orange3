import Orange

average = lambda xs: sum(xs)/float(len(xs))

data = Orange.data.Table("iris")
targets = data.domain.class_var.values
print("%-15s %s" % ("Attribute", " ".join("%15s" % c for c in targets)))
for a in data.domain.attributes:
    dist = ["%15.2f" % average([d[a] for d in data if d.get_class() == c])
            for c in targets]
    print("%-15s" % a.name, " ".join(dist))
