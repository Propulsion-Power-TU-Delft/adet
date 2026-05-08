from itertools import zip_longest
from collections.abc import Iterable
import numpy as np
from typing import Any, Iterator, Literal, TypeVar, overload

T = TypeVar('T')  # Iterable type
F = TypeVar('F')  # Fillvalue type


def leaves(iter: Iterable[T | Iterable[T]]) -> tuple[T, ...]:
    return tuple(chain_mixed(*iter))


def chain_mixed(*args: T | Iterable[T]) -> Iterable[T]:
    for item in args:
        try:
            yield from item
        except Exception:
            yield item


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


def ensure_tuple(x: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(x, Iterable):
        return tuple(x)  # ty:ignore
    else:
        return (x,)


keys = np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
values = np.array(['A', 'B', 'C'])


def closest(query, keys, values, ord=2):
    dists = np.linalg.norm(keys - np.array(query), axis=1, ord=ord)
    idx = np.argmin(dists)
    return keys[idx], values[idx]
