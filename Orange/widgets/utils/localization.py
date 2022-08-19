__all__ = ["pl", "plsi"]


def pl(n, forms):
    if "|" in forms:
        return forms.split("|")[n != 1]
    else:
        return forms + "s" * (n != 1)


def plsi(n, forms):
    forms = forms.split("|")
    if n % 100 == 1:
        return forms[0]
    if n % 100 == 2:
        return forms[1]
    if n % 100 in (3, 4):
        return forms[2]
    return forms[3]
