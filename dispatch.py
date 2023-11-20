import numpy_CAS as np
from utils import show_variable, equation

# Expressions

LAMBDA = "λ = (PD + Σ_i^ngen (β_i/(2y_i))) / Σ_i^ngen (1/(2y_i))"
P_I = "P_i = (λ - β_i)/(2y_i)"
P_FIXED = "P_i -> p_fix"

# Functions

f_cost = lambda Pi, params: params[0] + params[1]*Pi + params[2]*(Pi**2)
df_cost = lambda Pi, params: params[1] + 2*params[2]*Pi

def dispatch_static(Pd, costs_params):
    print(LAMBDA)
    # lambda
    sum_1 = 0
    sum_2 = 0
    for params in costs_params:
        sum_1 += params[1]/(2*params[2])
        sum_2 += 1/(2*params[2])
    lambda_inc = (Pd + sum_1)/sum_2

    P = distribute_power(lambda_inc, costs_params)
    C = distribute_cost(P, costs_params)

    summary = {
        "lambda": lambda_inc,
        "P": P,
        "C": C,
    }
    return summary

def dispatch_fixed(Pd, costs_params, restrictions):
    print("------- k=0 ---------")
    initiation = dispatch_static(Pd, costs_params)
    P_k = initiation["P"]
    lambda_k = initiation["lambda"]

    violations = []
    off_load = True

    k = 1
    while off_load:
        print("-------- k={k} --------".format(k=k))
        off_load = False
        for i in range(P_k.shape[0]):
            p_i = P_k[i,0]
            if (restrictions[i][0] > p_i) or (restrictions[i][1] < p_i):
                P_k[i,0] = saturate(p_i, restrictions[i])
                violations.append([i, P_k[i,0]])
                off_load = True
            i += 1

        dP_k = Pd - np.sum(P_k)
        dlambda_k = 0
        ind_fix = [item[0] for item in violations]
        for i in range(P_k.shape[0]):
            if not (i in ind_fix):
                dlambda_k += 1/(2*costs_params[i][2])
        dlambda_k = dP_k/dlambda_k 
        
        lambda_k += dlambda_k
        P_k = distribute_power(lambda_k, costs_params, violations=violations)
        k += 1


    C = distribute_cost(P_k, costs_params)

    summary = {
        "lambda": lambda_k,
        "P": P_k,
        "C": C,
    }
    
    return summary

###

def saturate(value, limits):
    if value < limits[0]:
        value = limits[0]
    if value > limits[1]:
        value = limits[1]
    return value

def distribute_power(incremental, costs_params, violations=[]):
    P = np.zeros((len(costs_params), 1))
    print("-----------")
    print("Power computation")
    for i in range(len(P)):
        P[i,0] = (incremental - costs_params[i][1]) / (2*costs_params[i][2])
        equation(P_I, {"i": i, "λ": incremental}, P[i,0])
    print("Power fixed")
    for i, p_fixed in violations:
        P[i,0] = p_fixed
        equation(P_FIXED, {"_i": "_{i}".format(i=i),"p_fix": P[i,0]}, P[i,0])
    return P


def distribute_cost(P, costs_params):
    C = np.zeros((len(P), 1))
    for i in range(len(C)):
        C[i,0] = f_cost(P[i,0], costs_params[i])
    return C

# Dispatch
class Dispatch:
    def __init__(self, costs_params = [], restrictions = []) -> None:
        self.costs_params = costs_params
        self.restrictions = restrictions
        self.optimization = {}

    def optimize(self, Pd, restrictions = True):
        if restrictions:
            self.optimization = dispatch_fixed(Pd, self.costs_params, self.restrictions)
        else:
            self.optimization = dispatch_static(Pd, self.costs_params)

        return self
    