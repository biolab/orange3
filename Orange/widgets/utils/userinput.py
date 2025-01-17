from typing import Type, TypeVar, Optional
import re

T = TypeVar("T", int, float)


def numbers_from_list(
        text: str,
        typ: Type[T],
        minimum: Optional[T] = None,
        maximum: Optional[T] = None,
        enforce_range: Optional[bool] = True) -> tuple[T, ...]:
    text = text.strip()
    if not text:
        return ()

    if re.search(r"(^|[^.])\.\.\.($|[^.])", text):
        return _numbers_from_dots(text, typ, minimum, maximum, enforce_range)
    else:
        if enforce_range:
            return _numbers_from_no_dots(text, typ, minimum, maximum)
        else:
            return _numbers_from_no_dots(text, typ)


def _get_points(numbers: list[str], typ: Type[T]) -> tuple[T, ...]:
    try:
        return tuple(map(typ, numbers))
    except ValueError as exc:
        msg = str(exc)
        raise ValueError(f"invalid value ({msg[msg.rindex(':') + 1:]})") from exc


def _numbers_from_no_dots(
        text: str,
        typ: Type[T],
        minimum: Optional[T] = None,
        maximum: Optional[T] = None) -> tuple[T, ...]:
    points = text.replace("...", " ... ").replace(", ", " ").split()
    steps = tuple(sorted(set(_get_points(points, typ))))
    under = minimum is not None and steps[0] < minimum
    over = maximum is not None and steps[-1] > maximum
    if under and over:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    if under:
        raise ValueError(f"value must be at least {minimum}")
    if over:
        raise ValueError(f"value must be at most {maximum}")
    return steps


def _numbers_from_dots(
        text: str,
        typ: Type[T],
        minimum: Optional[T] = None,
        maximum: Optional[T] = None,
        enforce_range: Optional[bool] = True) -> tuple[T, ...]:
    # many branches are results of many checks and don't degrade readability
    # pylint: disable=too-many-branches
    points = text.replace("...", " ... ").replace(",", " ").split()
    if points.count("...") > 1:
        raise ValueError("multiple '...'.")
    dotind = points.index("...")
    pre = _get_points(points[:dotind], typ)
    post = _get_points(points[dotind + 1:], typ)
    if pre and post and pre[-1] >= post[0]:
        raise ValueError("values before '...' must be smaller than values after.")

    diffs = {y - x for x, y in zip(pre, pre[1:])} \
            | {y - x for x, y in zip(post, post[1:])}
    if not diffs:
        raise ValueError("at least two values are required before or after '...'.")
    diff_of_diffs = max(diffs) - min(diffs)
    if diff_of_diffs > 1e-10:
        raise ValueError("points must be in uniform order.")
    diff = next(iter(diffs))
    if typ is float:
        diff = round(diff, 7)
    if diff <= 0:
        raise ValueError("points must be in increasing order.")

    minpoint = pre[0] if pre else minimum
    maxpoint = post[-1] if post else maximum
    if minpoint is None:
        raise ValueError("minimum value is missing.")
    if maxpoint is None:
        raise ValueError("maximum value is missing.")
    if enforce_range:
        if minimum is not None and minpoint < minimum:
            raise ValueError(f"minimum value is below the minimum {minimum}.")
        if maximum is not None and maxpoint > maximum:
            raise ValueError(f"maximum value is above the maximum {maximum}.")
    steps = (maxpoint - minpoint) // diff
    if (minpoint - maxpoint) % diff > 1e-10:
        if pre and post:
            raise ValueError(
                "the sequence before '...' does not end with the sequence after it.")
        if not pre:
            minpoint = maxpoint - steps * diff
        else:
            maxpoint = minpoint + steps * diff

    if typ is int:
        return tuple(range(minpoint, maxpoint + diff, diff))
    else:
        points = [minpoint + i * diff
                 for i in range(int((maxpoint - minpoint) / diff) + 1)]
        if maxpoint - points[-1] > 1e-10:
            points.append(maxpoint)
        return tuple(points)
