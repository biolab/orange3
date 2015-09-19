def plural(s, number):
    return s.format(number=number, s="s" if number % 100 != 1 else "")


def plural_w(s, number, capitalize=False):
    numbers = ("zero", "one", "two", "three", "four", "five", "six", "seven",
               "nine", "ten")
    number_str = numbers[number] if number < len(numbers) else str(number)
    if capitalize:
        number_str = number_str.capitalize()
    return s.format(number=number_str, s="s" if number % 100 != 1 else "")


def clip_string(s, limit=1000, sep=None):
    if len(s) < limit:
        return s
    s = s[:limit - 3]
    if sep is None:
        return s
    sep_pos = s.rfind(sep)
    if sep_pos == -1:
        return s
    return s[:sep_pos + len(sep)] + "..."


def clipped_list(s, limit=1000):
    return clip_string(", ".join(s), limit, ", ")

