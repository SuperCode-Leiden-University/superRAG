def add(x: int, y: int) -> int:
    """Add two numbers x and y
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
    return x + y

# Assertions to check the expected results
assert add(2, 3) == 5, "Test case 1 failed"
assert add(5, 7) == 12, "Test case 2 failed"