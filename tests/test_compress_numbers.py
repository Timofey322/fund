"""Автотесты для compress_numbers."""

from __future__ import annotations

import math

import pytest

from common.utils.compress_numbers import compress_numbers


class TestExamplesFromTask:
    """Примеры из условия задачи."""

    def test_example_1(self):
        assert compress_numbers([1, 1, 2, 2, 3]) == [1, 2, 3]

    def test_example_2(self):
        assert compress_numbers([0, 0, 1, 1, 0]) == [0, 1, 0]


class TestEdgeCases:
    """Граничные случаи."""

    def test_empty(self):
        assert compress_numbers([]) == []

    def test_single_element(self):
        assert compress_numbers([7]) == [7]

    def test_all_same(self):
        assert compress_numbers([5, 5, 5, 5]) == [5]

    def test_no_duplicates(self):
        assert compress_numbers([1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_alternating(self):
        assert compress_numbers([1, 2, 1, 2, 1]) == [1, 2, 1, 2, 1]


class TestNonConsecutiveDuplicatesKept:
    """Дубликаты НЕ подряд должны сохраняться."""

    def test_separated_duplicates(self):
        # Повтор 1 не подряд → остаётся
        assert compress_numbers([1, 2, 2, 1]) == [1, 2, 1]

    def test_long_runs(self):
        assert compress_numbers([3, 3, 3, 1, 1, 3, 3]) == [3, 1, 3]


class TestNumberTypes:
    """Разные типы чисел."""

    def test_negative(self):
        assert compress_numbers([-1, -1, 0, 0, -1]) == [-1, 0, -1]

    def test_floats(self):
        assert compress_numbers([1.5, 1.5, 2.0, 2.0]) == [1.5, 2.0]

    def test_int_float_equal_collapse(self):
        # 1 == 1.0 → считаются одинаковыми (сравнение по ==)
        assert compress_numbers([1, 1.0, 2]) == [1, 2]


class TestPurity:
    """Функция не должна мутировать вход и должна возвращать новый список."""

    def test_input_not_mutated(self):
        src = [1, 1, 2, 3, 3]
        _ = compress_numbers(src)
        assert src == [1, 1, 2, 3, 3]

    def test_returns_new_list(self):
        src = [1, 2, 3]
        out = compress_numbers(src)
        assert out is not src
        assert out == [1, 2, 3]


class TestIterables:
    """Работа с произвольными итерируемыми объектами, не только списком."""

    def test_tuple_input(self):
        assert compress_numbers((1, 1, 2)) == [1, 2]

    def test_generator_input(self):
        gen = (x for x in [4, 4, 5, 5, 5, 6])
        assert compress_numbers(gen) == [4, 5, 6]


class TestLargeInput:
    """Производительность / корректность на большом входе."""

    def test_large_all_same(self):
        assert compress_numbers([9] * 10_000) == [9]

    def test_large_pairs(self):
        data = []
        for i in range(1000):
            data.extend([i, i])
        assert compress_numbers(data) == list(range(1000))


@pytest.mark.parametrize(
    "given, expected",
    [
        ([1, 1, 2, 2, 3], [1, 2, 3]),
        ([0, 0, 1, 1, 0], [0, 1, 0]),
        ([], []),
        ([2, 2, 2], [2]),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_parametrized(given, expected):
    assert compress_numbers(given) == expected
