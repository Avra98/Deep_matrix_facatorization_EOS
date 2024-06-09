from scipy.optimize import fsolve

def system_of_equations(x, c1, c2, c3, s):
  w1, w2, w3, w4 = x
  # Define the equations here
  eq1 = w1*w2*w3*w4 - s  # Product equation
  eq2 = w1**2 - w2**2 + w3**2 - w4**2 - c1  # Squared term equation 1
  eq3 = w1**2 + w2**2 - w3**2 - w4**2 - c2  # Squared term equation 2
  eq4 = -w1**2 + w2**2 + w3**2 - w4**2 - c3  # Squared term equation 3
  return [eq1, eq2, eq3, eq4]



# Define initial guess (replace with your own estimates)
initial_guess = [2, 2, 2, 2]

# Input your constants (replace with your desired values)
c1 = 1  # Replace with your value for c1
c2 = 2  # Replace with your value for c2
c3 = 3  # Replace with your value for c3
s = 5  # Replace with your value for s

# Solve the system of equations
solution = fsolve(system_of_equations, initial_guess, args=(c1, c2, c3, s))

# Print the solution
print("Solution:", solution)
