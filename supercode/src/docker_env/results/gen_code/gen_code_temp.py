def same_chars(s0: str, s1: str) -> bool:
    """
    Check if two words have the same characters.
    """
    return set(s0) == set(s1)

# Assertions to check the expected results
assert same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc') == True
assert same_chars('abcd', 'dddddddabc') == True
assert same_chars('dddddddabc', 'abcd') == True
assert same_chars('eabcd', 'dddddddabc') == False
assert same_chars('abcd', 'dddddddabce') == False
assert same_chars('eabcdzzzz', 'dddzzzzzzzddddabc') == False