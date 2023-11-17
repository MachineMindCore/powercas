import numpy_CAS as np

def show_variable(name, value, decimals=5):
    print(name,'=')
    try: 
        print(np.round(value, decimals=decimals))
    except:
        print(value)
    print()