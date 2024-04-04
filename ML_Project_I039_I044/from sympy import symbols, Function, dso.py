from sympy import symbols, Function, dsolve, Eq
# Define symbols
t = symbols('t')
i = Function('i')
# Define the symbols for L, R, and C
L,R,C = symbols('L R C')
# Define the differential equation
eq = Eq(L*i(t).diff(t, t) + R*i(t).diff(t) + i(t)/C, 0)
# Solve the differential equation
sol = dsolve(eq, i(t))
print("Solution of the differential equation:")
print(sol)