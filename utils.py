import numpy_CAS as np
import re

def show_variable(name, value, decimals=5):
    print(name,'=')
    try: 
        print(np.round(value, decimals=decimals))
    except:
        print(value)
    print()

def equation(expression, variables, value):
    DECIMALS = 3
    for var in variables.keys():
        replacement = round(variables[var]) if isinstance(variables[var], float) else variables[var]
        expression = expression.replace(var, str(replacement))
    print(expression)
    print()
    match = re.search(r'([^=]+)=', expression)
    if match:
        result = match.group(1)
        print(result, " = ", str(value))
    print()
