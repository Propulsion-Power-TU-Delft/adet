from itertools import zip_longest
from collections.abc import Iterable
from typing import Any, Iterator, Literal, TypeVar, overload

T = TypeVar('T')  # Iterable type
F = TypeVar('F')  # Fillvalue type


# fill mode — fillvalue is required
@overload
def grouper(
    iterable: Iterable[T],
    n: int,
    *,
    incomplete: Literal['fill'] = 'fill',
    fillvalue: F,
) -> Iterator[tuple[T | F, ...]]: ...


# strict or ignore modes — no fillvalue semantics
@overload
# Length 2 case
def grouper(
    iterable: Iterable[T],
    n: Literal[2],
    *,
    incomplete: Literal['strict', 'ignore'],
    fillvalue: Any = ...,
) -> Iterator[tuple[T, T]]: ...


# Length 3 case
@overload
def grouper(
    iterable: Iterable[T],
    n: Literal[3],
    *,
    incomplete: Literal['strict', 'ignore'],
    fillvalue: Any = ...,
) -> Iterator[tuple[T, T, T]]: ...


# Length 3 case
@overload
def grouper(
    iterable: Iterable[T],
    n: Literal[4],
    *,
    incomplete: Literal['strict', 'ignore'],
    fillvalue: Any = ...,
) -> Iterator[tuple[T, T, T, T]]: ...


def grouper(
    iterable: Iterable[T],
    n: int,
    *,
    incomplete: Literal['fill', 'strict', 'ignore'] = 'fill',
    fillvalue: F | None = None,
) -> Iterator[tuple[T | F, ...]]:
    """
    Collect data into non-overlapping fixed-length chunks or blocks.

    Examples
    --------
    grouper('ABCDEFG', 3, fillvalue='x') → ABC DEF Gxx
    grouper('ABCDEFG', 3, incomplete='strict') → ABC DEF ValueError
    grouper('ABCDEFG', 3, incomplete='ignore') → ABC DEF
    """
    iterators = [iter(iterable)] * n
    match incomplete:
        case 'fill':
            return zip_longest(*iterators, fillvalue=fillvalue)  # type: ignore
        case 'strict':
            return zip(*iterators, strict=True)
        case 'ignore':
            return zip(*iterators)
        case _:
            raise ValueError('Expected fill, strict, or ignore')


def ensure_tuple(x: int | Iterable[int]):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)
