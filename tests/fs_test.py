import pytest


def sum(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("I parametri devono essere numeri")
    return a + b


def test_sum():
    assert sum(1, 2) == 3
    assert sum(0, 0) == 0
    assert sum(-1, 1) == 0
    with pytest.raises(TypeError):
        sum("a", "b")
