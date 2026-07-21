"""Удаление подряд идущих дубликатов из массива чисел.

Пример:
    [1, 1, 2, 2, 3] -> [1, 2, 3]
    [0, 0, 1, 1, 0] -> [0, 1, 0]
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def compress_numbers(numbers: Sequence[T] | Iterable[T]) -> list[T]:
    """Вернуть новый список без подряд идущих дубликатов.

    Сохраняет исходный порядок и не мутирует входные данные.
    Сравнение элементов идёт по равенству (==), что корректно для int/float.

    :param numbers: последовательность чисел (любой итерируемый объект)
    :return: новый список, где каждый элемент не равен предыдущему
    """
    result: list[T] = []
    for value in numbers:
        if not result or value != result[-1]:
            result.append(value)
    return result


if __name__ == "__main__":
    print(compress_numbers([1, 1, 2, 2, 3]))  # [1, 2, 3]
    print(compress_numbers([0, 0, 1, 1, 0]))  # [0, 1, 0]
