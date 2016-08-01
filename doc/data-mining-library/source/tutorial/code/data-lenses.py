import Orange
data = Orange.data.Table("lenses")
print("Attributes:", ", ".join(x.name for x in data.domain.attributes))
print("Class:", data.domain.class_var.name)
print("Data instances", len(data))

target = "soft"
print("Data instances with %s prescriptions:" % target)
atts = data.domain.attributes
for idx, r in data.iterrows():
    if r[r.domain.class_var] == target:
        print(" ".join(["%14s" % str(r[a]) for a in atts]))
