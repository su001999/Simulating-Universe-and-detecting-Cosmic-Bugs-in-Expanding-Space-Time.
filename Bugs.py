import numpy as np
from sympy import symbols, Abs

# Define symbolic variables
S_t, P, P_0, M_A, M_E = symbols('S_t P P_0 M_A M_E')

# Constants for thresholds
epsilon = 1e-3  # Threshold for information loss
delta = 1e-3  # Threshold for emergence of new phenomena
gamma = 1e-3  # Threshold for matrix size discrepancy

# Matrix size values (actual and expected sizes)
matrix_sizes = [10, 15, 16, 17, 18, 19, 20, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196,
                215, 236, 259, 284]
expected_sizes = [10, 16, 17, 19, 21, 23, 25, 67, 74, 81, 89, 98, 108, 119, 131, 144, 158, 174, 191, 211, 232, 255, 281,
                  309, 340, 374, 411, 452]

# Sample numeric values for P and P_0
P_value = 5  # Replace with actual numeric value for P
P_0_value = 3  # Replace with actual numeric value for P_0


# Function to numerically calculate Information Loss (I_L)
def calculate_information_loss(T):
    # Replace this with an actual computation or test value, using symbolic summation for demonstration
    return -11  # Example value of information loss


# Function to calculate emergence of new phenomena (E_N)
def calculate_emergence_of_phenomena(P_value, P_0_value):
    return Abs(P_value - P_0_value)  # This will return a numeric value


# Function to calculate non-additive value (N_A)
def calculate_non_additive_value(matrix_sizes):
    return np.prod(matrix_sizes)  # Use numpy.prod() for numeric calculation


# Function to calculate matrix size discrepancy (D)
def calculate_matrix_discrepancy(matrix_sizes, expected_sizes):
    discrepancies = [Abs(M_A - M_E) for M_A, M_E in zip(matrix_sizes, expected_sizes)]
    return discrepancies


# Function to print formalism and values
def print_formalism_and_values(I_L, E_N, N_A, D, step_discrepancies):
    print("\nFormalism:")
    print(
        r"$\mathbb{B} = 1 \quad \text{if} \quad \left( \sum_{t=0}^{T} \left(S(t) - S(t+1)\right) > \epsilon \right) \lor \left(g(P) - g(P_0) > \delta\right) \lor \left( \prod_{i=1}^{n} M_i \leq 0 \right) \lor \left( |M_A - M_E| > \gamma \right) \quad \mathbb{B} = 0 \quad \text{otherwise}$")

    print("\nComputed Values:")
    print(f"Information Loss (I_L): {I_L}")
    print(f"Emergence of New Phenomena (E_N): {E_N}")
    print(f"Non-Additive Value (N_A): {N_A}")

    for step, D_value in enumerate(step_discrepancies):
        print(f"Step {step}: Matrix Size Discrepancy (D): {D_value}")


# Function to calculate and test bug conditions numerically
def test_B(T, matrix_sizes, expected_sizes, P_value, P_0_value):
    # Calculate numeric components
    I_L = calculate_information_loss(T)
    E_N = calculate_emergence_of_phenomena(P_value, P_0_value)
    N_A = calculate_non_additive_value(matrix_sizes)
    step_discrepancies = calculate_matrix_discrepancy(matrix_sizes, expected_sizes)

    # Print the formalism and values
    print_formalism_and_values(I_L, E_N, N_A, gamma, step_discrepancies)

    # Numerically check for bugs based on conditions
    B = 0
    if I_L > epsilon:
        print("Bug detected: Significant Information Loss")
        B = 1
    if E_N.evalf() > delta:  # Convert symbolic result to numeric using evalf()
        print("Bug detected: Emergence of New Physical Phenomena")
        B = 1
    if N_A <= 0:
        print("Bug detected: Non-Additive Value is Zero or Negative")
        B = 1
    if any(D.evalf() > gamma for D in step_discrepancies):
        print("Bug detected: Matrix Size Discrepancy")
        B = 1

    print(f"\nResult of Bug Test (B): {B}")
    return B


# Test the function
result = test_B(10, matrix_sizes, expected_sizes, P_value, P_0_value)
