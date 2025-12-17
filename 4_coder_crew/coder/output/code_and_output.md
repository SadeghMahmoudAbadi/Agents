# Calculation of Series Approximation of π

This program calculates the first 10,000 terms of the series: 1 - 1/3 + 1/5 - 1/7 + ... and multiplies the result by 4.

```python
total = 0
for i in range(10000):
    # Calculate the denominator (2*i + 1 gives odd numbers: 1, 3, 5, 7, ...)
    denominator = 2 * i + 1
    
    # Alternate signs: + for even i, - for odd i
    if i % 2 == 0:
        total += 1 / denominator
    else:
        total -= 1 / denominator

# Multiply by 4 as requested
result = total * 4
print(f"Sum of first 10,000 terms: {total}")
print(f"Result multiplied by 4: {result}")
```

## Output

Sum of first 10,000 terms: 0.7853731633975086
Result multiplied by 4: 3.1414926535900345

## Explanation

This series is known as the Leibniz formula for π:

π = 4 * (1 - 1/3 + 1/5 - 1/7 + 1/9 - ...)

The result we obtained (3.1414926535900345) is an approximation of π using the first 10,000 terms of this series.