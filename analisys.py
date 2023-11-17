import numpy_CAS as np

def admittance_matrix(system):
    """
    Calcula la matriz de admitancia en base a los datos del diccionario del sistema.
    
    Args:
        system(Dict[int, Dict]): Diccionario que describe el sistema de potencia.
  
    Returns:
        Y(ndarray[complex]): Matriz de admitancia del sistema (referida a tierra).
    """

    dim = len(list(system.keys()))
    Y = np.zeros((dim, dim), dtype=complex)
    for from_id, data in system.items():
        # Diagonal
        Y[from_id-1, from_id-1] = data['conn_gnd'] + sum([y_ik for _, y_ik in data['conn']])
        # No Diagonal
        for to_id, y_ij in data['conn']:
            Y[from_id-1, to_id-1] = -y_ij
    return Y

def injected_current(Y_system, E_system):
    return np.dot(Y_system, E_system)

def injected_power(Y_system, E_system):

    n = len(Y_system)
    P_injected = np.zeros((n, 1), dtype=float)
    Q_injected = np.zeros((n, 1), dtype=float)

    # Matriz de angulos de Y
    theta_system = np.angle(Y_system)
    # Matriz de angulos de E
    delta_system = np.angle(E_system)
    for i in range(n): 
        for k in range(n):
            angle = theta_system[i,k] - delta_system[i,0] + delta_system[k,0]
            constant = np.abs(Y_system[i,k]*E_system[i,0]*E_system[k,0])
            # Calculo de potencia activa
            P_injected[i,0] += constant * np.cos(angle)
            # Calculo de potencia reactiva
            Q_injected[i,0] -= constant * np.sin(angle)
    return P_injected + Q_injected*1j

def system_current(Y_system, E_system):
    ones = np.ones(E_system.shape, dtype=float)
    I_system = Y_system * (np.dot(ones, E_system.transpose()) - np.dot(E_system, ones.transpose()))
    return I_system

def transfered_power(E_system, I_system):
    S_transfer = np.matmul(E_system, np.conjugate(I_system), dtype=complex)
    return S_transfer

def power_loss_m1(E_system, I_injected):
    loss = np.matmul(E_system.transpose(), np.conjugate(I_injected), dtype=complex)
    return loss

def power_loss_m2(S_injected):
    loss = 0
    n = S_injected.shape[0]
    for i in range(n):
        loss = S_injected[i,0] + loss
    return loss

def power_loss_m3(S_transfer):
    loss = 0
    n = S_transfer.shape[0]
    for i in range(n-1):
        for k in range(i+1,n):
            loss = S_transfer[i,k] + S_transfer[k,i] + loss
    return loss