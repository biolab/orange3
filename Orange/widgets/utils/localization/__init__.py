def pl(n: int, forms: str) -> str:  # pylint: disable=invalid-name
    """
    Choose a singular/plural form for English - or create one, for regular nouns

    `forms` can be a string containing the singular and plural form, separated
    by "|", for instance `"dictionary|dictionaries".

    IF `forms` does not contain character |, plural is formed by appending
    an 's'.

    Args:
        n: number
        forms: plural forms, separated by "|", or a single (regular) noun

    Returns:
        form corresponding to the given number
    """
    if "|" in forms:
        return forms.split("|")[n != 1]
    else:
        return forms + "s" * (n != 1)
