import numpy_CAS as np
from utils import show_variable
# Helper

def compose(M, M_down, M_right, m_ll):
    """
    Matriz composition:
    [
        [M, M_right],
        [M_down, m_ll]
    ]
    """
    M_a = np.append(M, M_down, axis=0)
    M_b = np.append(M_right, [[0]], axis=0)
    M_new = np.append(M_a, M_b, axis=1)
    M_new[M_new.shape[0]-1, M_new.shape[1]-1] = m_ll
    return M_new 

def reduce_kron(M, connection):
    """
    Kron reduction
    """
    p = connection[0]
    q = connection[1]
    last = len(M) - 1
    M_old = M[:last,:last]
    if q != 0:
        dZ = M[:,q] - M[:,p]
    else:
        dZ = -1*M[:,p]
    M_reduced = M_old - (1/M[last,last])*(dZ*dZ.transpose())
    return M_reduced

# Rules

def sature_dim(func):
    def wrapper(self, *args, **kwargs):
        object_result = func(self, *args, **kwargs)
        object_dim = len(object_result.Z)
        if object_dim > object_result.dim:
            return reduce_kron(object_result.Z, object_result.connection)
        else:
            
        print(f"After calling {func.__name__}. Result: {result}")
        return result
    return wrapper

def rule_0(z_q0, connection):
    """
    Start Z to reference
    """
    print("Inicio Z -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[z_q0]")
    Z_new = np.array([[z_q0]])
    print(Z_new)
    return Z_new


def rule_1(Z, z_q0, connection):
    """
    Node to reference (z_q0)
    """
    print("Nodo a referencia -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, 0], [0, z_q0]]")
    Z_down = np.zeros((1, len(Z)-1))
    Z_right = np.zeros((len(Z)-1, 1))
    Z_new = compose(Z, Z_down, Z_right, z_q0)

    show_variable("Z_k+1", Z_new)
    return Z_new

def rule_2(Z, z_pq, connection):
    """
    New node to old node (z_pq)
    """
    print("Nodo nuevo a existente-> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, Z_p:], [Z_:p, Z_pp+z_pq]]")
    p = connection[0]
    q = connection[1]
    Z_down = Z[p,:]
    Z_right = Z[:,p]
    z_ll = Z[p,p] + z_pq
    Z_new = compose(Z, Z_down, Z_right, z_ll)

    show_variable("Z_k+1", Z_new)
    return Z_new

def rule_3(Z, z_pq, connection):
    """
    old node to old node (z_pq)
    """
    print("Nodo existente a existente -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, Z_q:-Z_p:], [Z_:q-Z_:p, z_pq+Z_pp+Z_qq-2Z_pq]]")
    p = connection[0]
    q = connection[1]
    Z_down = Z[q,:] - Z[p,:]
    Z_right = Z[:,q] - Z[:,p]
    z_ll = z_pq + Z[p,p] + Z[q,q] - 2*Z[p,q]
    Z_new = compose(Z, Z_down, Z_right, z_ll)

    show_variable("Z_k+1", Z_new)
    return Z_new


# Models

class GraphZ:
    def __init__(self, dim) -> None:
        self.Z = None
        self.dim = dim
        self.connection = None
        self.nodes = set()

    def define_rule(self, connection):
        p = connection[0]
        q = connection[1]
        if self.nodes == []:
            rule = "0"
        elif q == 0:
            rule = "1"
        elif not(p in self.nodes) and (q in self.nodes):
            rule = "2"
        elif (p in self.nodes) and (q in self.nodes):
            rule = "3"
        else:
            raise ValueError("Not rule for this issue")
        
        self.nodes.add(p)
        self.nodes.add(q)
        return rule

    def add(self, z_pq, connection):
        self.connection = connection
        rule = self.define_rule(connection)
        if rule == "0":
            self.Z = rule_0(z_pq, connection)
        elif rule == "1":
            self.Z = rule_1(self.Z, z_pq, connection)
        elif rule == "2":
            self.Z = rule_2(self.Z, z_pq, connection)
        elif rule == "3":
            self.Z = rule_3(self.Z, z_pq, connection)
        return self

        