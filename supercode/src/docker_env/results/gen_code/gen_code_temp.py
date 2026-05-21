def return_even(n):
    # Initialize an empty list to store even numbers
    even_numbers = []
    
    # Loop through the range from 0 to n
    for i in range(n + 1):
        # Check if the current number is even
        if i % 2 == 0:
            # Append the even number to the list
            even_numbers.append(i)
    
    # Return the list of even numbers
    return even_numbers

# Test the function with the provided test case
assert return_even(5) == [0, 2, 4]