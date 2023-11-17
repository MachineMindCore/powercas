import numpy_CAS as np

def extract(system, tag, dtype):
    dim = len(system.keys())
    result = np.zeros((dim, 1), dtype=dtype)    
    for i, key in enumerate(system.keys()):
        result[i,0] = system[key][tag]
    return result

def extract_E(system):
    return extract(system, "E", float)

def extract_d(system):
    return extract(system, "d", float)