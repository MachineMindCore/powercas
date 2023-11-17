import numpy_CAS as np
import math
import cmath
from utils import show_variable

def flat_profile(system):
    """
    Calcula el perfil plano como prerrequisito de cualquier método de flujo de potencia.


    Args:
        system (Dict[int, Dict]): Diccionario que describe el sistema de potencia.


    Returns:
        X0 (ndarray[float]): Vector de perfil plano de variables desconocidas.
        V0 (ndarray[complex]): Vector de perfil plano de voltaje complejo.
        IND (ndarry[str]): Vector de índice de variables desconocidas.
    """

    
    dim = len(system.keys())
    E = np.ones((dim, 1))
    d = np.zeros((dim, 1))
    
    miss_E = []
    miss_d = []
    for i, data in system.items():
        if data['E'] != None:
            E[i-1,0] = data['E']
        else:
            miss_E.append('E'+str(i))
        if data['d'] != None:
            d[i-1,0] = math.radians(data['d'])
        else:
            miss_d.append('d'+str(i))

    E_unknown = np.ones((len(miss_E), 1))
    d_unknown = np.zeros((len(miss_d), 1))

    V0 = E * np.exp(d*1j)
    if miss_E == []:
        X0 = d_unknown
    else:
        X0 = np.append(d_unknown, E_unknown, axis=0, dtype=float)
    IND = miss_d + miss_E
    return X0, V0, IND

def compute_C(system, IND):
    """
    Calcula los valores de potencia neta en los nodos con variables desconocidas.

    Args:
        system (Dict[int, Dict]): Diccionario que describe el sistema de potencia.
        IND (ndarry[str]): Vector de índice de variables desconocidas.

    Returns:
        C(ndarray[float]): Vector de potencias (P o Q) netas de los nodos de variables desconocidas.
    """
    C = np.zeros((len(IND), 1), dtype=float)
    for i, key in enumerate(IND):
        type = key[0]
        k = int(key[1:])
        if type == 'd':
            C[i,0] = system[k]['P_g'] - system[k]['P_d']
        if type == 'E':
            C[i,0] = system[k]['Q_g'] - system[k]['Q_d']
    return C

##########################################################################################################
# Newton Raphson

def jacobian_1(Y_mag, Y_ang, E, d, IND_d, IND_E):
    """
    Calcula la matriz parcial J1 de J.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        E(ndarray[float]): Magnitud de la matriz de voltajes.
        d(ndarray[float]): Ángulo de la matriz de voltajes.
        IND_d(List[str]): Índices de ángulos desconocidos.
        IND_E(List[str]): Índices de tensiones desconocidas.
    
    Returns:
        J1(ndarray[float]): Jacobiano 1
    """

    dim = len(Y_mag)
    E_dim = len(IND_E)
    d_dim = len(IND_d)
    J1 = np.zeros((d_dim, d_dim), dtype=float)
    for row, key_row in enumerate(IND_d):
        for column, key_column in enumerate(IND_d):
            i_key = int(key_row[1])-1
            j_key = int(key_column[1])-1
            if i_key == j_key:
                for j in range(dim):
                    if j != i_key:
                        J1[row][column] += E[i_key,0]*E[j,0]*Y_mag[i_key,j]*np.sin(Y_ang[i_key,j]-d[i_key,0]+d[j,0])
            else:
                J1[row][column] = -E[i_key,0]*E[j_key,0]*Y_mag[i_key,j_key]*np.sin(Y_ang[i_key,j_key]-d[i_key,0]+d[j_key,0]) 
    return J1

def jacobian_2(Y_mag, Y_ang, E, d, IND_d, IND_E):
    """
    Calcula la matriz parcial J2 de J.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        E(ndarray[float]): Magnitud de la matriz de voltajes.
        d(ndarray[float]): Ángulo de la matriz de voltajes.
        IND_d(List[str]): Índices de ángulos desconocidos.
        IND_E(List[str]): Índices de tensiones desconocidas.
    
    Returns:
        J2(ndarray[float]): Jacobiano 2
    """
    dim = len(Y_mag)
    E_dim = len(IND_E)
    d_dim = len(IND_d)
    J2 = np.zeros((d_dim, E_dim), dtype=float)
    for row, key_row in enumerate(IND_d):
        for column, key_column in enumerate(IND_E):
            i_key = int(key_row[1])-1
            j_key = int(key_column[1])-1
            if i_key == j_key:
                J2[row][column] = 2*E[i_key,0]*Y_mag[i_key,i_key]*np.cos(Y_ang[i_key,i_key])
                for j in range(dim):
                    if j != i_key:
                        J2[row][column] += E[j,0]*Y_mag[i_key,j]*np.cos(Y_ang[i_key,j]-d[i_key,0]+d[j,0])
            else:
                J2[row][column] = E[i_key,0]*Y_mag[i_key,j_key]*np.cos(Y_ang[i_key,j_key]-d[i_key,0]+d[j_key,0])
    return J2

def jacobian_3(Y_mag, Y_ang, E, d, IND_d, IND_E):
    """
    Calcula la matriz parcial J3 de J.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        E(ndarray[float]): Magnitud de la matriz de voltajes.
        d(ndarray[float]): Ángulo de la matriz de voltajes.
        IND_d(List[str]): Índices de ángulos desconocidos.
        IND_E(List[str]): Índices de tensiones desconocidas.
    
    Returns:
        J3(ndarray[float]): Jacobiano 3
    """
    dim = len(Y_mag)
    E_dim = len(IND_E)
    d_dim = len(IND_d)
    J3 = np.zeros((E_dim, d_dim), dtype=float)
    for row, key_row in enumerate(IND_E):
        for column, key_column in enumerate(IND_d):
            i_key = int(key_row[1])-1
            j_key = int(key_column[1])-1
            if i_key == j_key:
                for j in range(dim):
                    if j != i_key:
                        J3[row][column] += E[i_key,0]*E[j,0]*Y_mag[i_key,j]*np.cos(Y_ang[i_key,j]-d[i_key,0]+d[j,0])
            else:
                J3[row][column] = -E[i_key,0]*E[j_key,0]*Y_mag[i_key,j_key]*np.cos(Y_ang[i_key,j_key]-d[i_key,0]+d[j_key,0]) 
    return J3

def jacobian_4(Y_mag, Y_ang, E, d, IND_d, IND_E):
    """
    Calcula la matriz parcial J4 de J.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        E(ndarray[float]): Magnitud de la matriz de voltajes.
        d(ndarray[float]): Ángulo de la matriz de voltajes.
        IND_d(List[str]): Índices de ángulos desconocidos.
        IND_E(List[str]): Índices de tensiones desconocidas.
    
    Returns:
        J4(ndarray[float]): Jacobiano 4
    """
    dim = len(Y_mag)
    E_dim = len(IND_E)
    d_dim = len(IND_d)
    J4 = np.zeros((E_dim, E_dim), dtype=float)
    for row, key_row in enumerate(IND_E):
        for column, key_column in enumerate(IND_E):
            i_key = int(key_row[1])-1
            j_key = int(key_column[1])-1
            if i_key == j_key:
                J4[row][column] = -2*E[i_key,0]*Y_mag[i_key,i_key]*np.sin(Y_ang[i_key,i_key])
                for j in range(dim):
                    if j != i_key:
                        J4[row][column] -= E[j,0]*Y_mag[i_key,j]*np.sin(Y_ang[i_key,j]-d[i_key,0]+d[j,0])
            else:
                J4[row][column] = -E[i_key,0]*Y_mag[i_key,j_key]*np.sin(Y_ang[i_key,j_key]-d[i_key,0]+d[j_key,0])
    return J4

def jacobian(Y, E, d, IND):
    """
    Calcula la matriz jacobiana fragmentandola en 4 funciones por cada matriz parcial.


    Args:
        Y(ndarray[complex]): Matriz de admitancia del sistema.
        E(ndarray[float]): Magnitud de la matriz de voltajes.
        d(ndarray[float]): Ángulo de la matriz de voltajes.
        IND (ndarry[str]): Vector de índice de variables desconocidas.


    Returns:
        J(ndarray[float]): Matriz jacobiana completa.
    """

    Y_mag = np.abs(Y)
    Y_ang = np.angle(Y)
    IND_d = list(filter(lambda ind: ind[0] == 'd', IND))
    IND_E = list(filter(lambda ind: ind[0] == 'E', IND))
    J1 = jacobian_1(Y_mag, Y_ang, E, d, IND_d, IND_E)
    J2 = jacobian_2(Y_mag, Y_ang, E, d, IND_d, IND_E)
    J3 = jacobian_3(Y_mag, Y_ang, E, d, IND_d, IND_E)
    J4 = jacobian_4(Y_mag, Y_ang, E, d, IND_d, IND_E)
    J_up = np.concatenate((J1, J2), axis=1)
    J_dowm = np.concatenate((J3, J4), axis=1)
    J = np.concatenate(((J_up), (J_dowm)), axis=0)
    return J

def newton_raphson(Y, X0, V0, C, IND, tol = 0.001, max_steps = 100):
    """
    Calcula las tensiones y ángulos nodales de un sistema basado en una suposición inicial.
    Esta función imprime un resumen no retornable de variables por iteración.
    Cualquiera de las condiciones de parada detiene las iteraciones.


    Args:
        Y(ndarray[complex]): Matriz de admitancia del sistema.
        X0 (ndarray[float]): Vector de perfil plano de variables desconocidas.
        V0 (ndarray[complex]): Vector de perfil plano de voltaje complejo.
        C(ndarray[float]): Vector de potencias (P o Q) netas de los nodos de variables desconocidas.
        IND (ndarry[str]): Vector de índice de variables desconocidas.
        tol(float): Error de tolerancia (condición de parada).
        max_steps(int): Máximo número de iteraciones (condición de parada).

    Returns:
        E(ndarray[float]): Vector de magnitud de voltajes correctos.
        d(ndarray[float]): Vector de ángulo de voltajes correctos.
    """

    # pre-iteracion
    ## Indices
    IND_d = list(filter(lambda ind: ind[0] == 'd', IND))
    IND_E = list(filter(lambda ind: ind[0] == 'E', IND))
    ## Dimensiones
    d_dim = len(IND_d)
    E_dim = len(IND_E)
    dim = len(Y)
    # Variables
    Y_mag = np.abs(Y)
    Y_ang = np.angle(Y)
    P_k = np.zeros((d_dim, 1), dtype=float)
    Q_k = np.zeros((E_dim, 1), dtype=float)
    dC_k = np.concatenate(((P_k),(Q_k)), axis=0)
    C_k = np.zeros(C.shape, dtype=float)
    E_k = np.abs(V0)
    d_k = np.angle(V0)
    dd_k = np.zeros((d_dim, 1), dtype=float)
    dE_k = np.zeros((E_dim, 1), dtype=float)
    X_k = X0.copy()
    dX_k = np.zeros(X_k.shape, dtype=float)
    k = 0
    print('########## RESUMEN ##########')
    while True:
        k += 1
        # Proceso
        ## Actualizacion de P y Q
        for row, key_row in enumerate(IND_d):
            i = int(key_row[1])-1
            P_k[row,0] = 0
            for j in range(dim):
                P_k[row,0] += E_k[i,0]*E_k[j,0]*Y_mag[i,j]*np.cos(Y_ang[i,j]-d_k[i,0]+d_k[j,0])
        
        for row, key_row in enumerate(IND_E):
            i = int(key_row[1])-1
            Q_k[row,0] = 0
            for j in range(dim):
                Q_k[row,0] -= E_k[i,0]*E_k[j,0]*Y_mag[i,j]*np.sin(Y_ang[i,j]-d_k[i,0]+d_k[j,0])
        
        C_k = np.concatenate(((P_k), (Q_k)), axis=0)
        dC_k = C - C_k

        ## Actualizacion de X
        J_k = jacobian(Y, E_k, d_k, IND)
        dX_k = np.matmul(np.linalg.inv(J_k), dC_k)
        X_k = X_k + dX_k
        
        ## Actualizacion de d y E
        for row, key in enumerate(IND_d):
            i = int(key[1])-1
            d_k[i,0] = X_k[row,0]
        
        for row, key in enumerate(IND_E, start=d_dim):
            i = int(key[1])-1
            E_k[i,0] = X_k[row,0]
        error = max(dC_k).squeeze()

        ## Resumen
        print('----- Iteracion {k} -----'.format(k=k))
        show_variable('P_'+str(k), P_k)
        show_variable('Q_'+str(k), Q_k)
        # Condicion
        if error < tol or k > max_steps:
            break
    return E_k, d_k

###########################################################################################################
# Newton Raphson Fast Decoupled

def b_prime(Y_mag, Y_ang, IND_d):
    """
    Calcula la matriz B', aproximacion de J1.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        IND_d(List[str]): Índices de angulos desconocidos.
    Returns: 
        Bp(ndarray[float]): Matriz B'
    """
    dim_d = len(IND_d)
    Bp = np.zeros((dim_d, dim_d), dtype=float)
    for row, key_row in enumerate(IND_d):
        for column, key_column in enumerate(IND_d):
            i_key = key_row
            j_key = key_column
            Bp[row,column] = Y_mag[i_key,j_key] * np.sin(Y_ang[i_key,j_key])
    return Bp

def b_prime_prime(Y_mag, Y_ang, IND_E):
    """
    Calcula la matriz B'', aproximacion de J4.

    Args:
        Y_mag(ndarray[float]): Magnitud de la matriz de admitancia.
        Y_ang(ndarray[float]): Ángulo de la matriz de admitancia.
        IND_E(List[str]): Índices de tensiones desconocidas.
    Returns: 
        Bpp(ndarray[float]): Matriz B''
    """    
    dim_E = len(IND_E)
    Bpp = np.zeros((dim_E, dim_E), dtype=float)
    for row, key_row in enumerate(IND_E):
        for column, key_column in enumerate(IND_E):
            i_key = key_row
            j_key = key_column
            Bpp[row,column] = Y_mag[i_key,j_key] * np.sin(Y_ang[i_key,j_key])
    return Bpp

def newton_raphson_fast_decoupled(Y, X0, V0, C, IND, tol = 0.001, max_steps = 100):
    """
    Calcula las tensiones y ángulos nodales de un sistema basado en una suposición inicial.
    Esta función imprime un resumen no retornable de variables por iteración.
    Cualquiera de las condiciones de parada detiene las iteraciones.


    Args:
        Y(ndarray[complex]): Matriz de admitancia del sistema.
        X0 (ndarray[float]): Vector de perfil plano de variables desconocidas.
        V0 (ndarray[complex]): Vector de perfil plano de voltaje complejo.
        C(ndarray[float]): Vector de potencias (P o Q) netas de los nodos de variables desconocidas.
        IND (ndarry[str]): Vector de índice de variables desconocidas.
        tol(float): Error de tolerancia (condición de parada).
        max_steps(int): Máximo número de iteraciones (condición de parada).

    Returns:
        E(ndarray[float]): Vector de magnitud de voltajes correctos.
        d(ndarray[float]): Vector de ángulo de voltajes correctos.
    """

    # pre-iteracion
    print('######### RESUMEN ########')
    ## Indices
    IND_d_str = list(filter(lambda ind: ind[0] == 'd', IND))
    IND_E_str = list(filter(lambda ind: ind[0] == 'E', IND))
    
    IND_d = [int(key[1:])-1 for key in IND_d_str]
    IND_E = [int(key[1:])-1 for key in IND_E_str]
    ## Dimensiones
    d_dim = len(IND_d)
    E_dim = len(IND_E)
    dim = len(Y)
    # Variables
    Y_mag = np.abs(Y)
    Y_ang = np.angle(Y)
    P = C[:d_dim,:]
    Q = C[d_dim:,:]

    P_k = np.zeros((d_dim, 1), dtype=float)
    Q_k = np.zeros((E_dim, 1), dtype=float)
    E_k = np.abs(V0)
    d_k = np.angle(V0)
    dd_k = np.zeros((d_dim, 1), dtype=float)
    dE_k = np.zeros((E_dim, 1), dtype=float)
    dP_k = np.zeros((d_dim, 1), dtype=float)
    dQ_k = np.zeros((E_dim, 1), dtype=float)
    k = 0

    # Separacion de X0 en X_Ek y X_dk
    X_dk = X0[:d_dim,:]
    X_Ek = X0[d_dim:,:]

    # Calculo de Bp y Bpp
    Bp = b_prime(Y_mag, Y_ang, IND_d)
    Bp_inv = np.inv(Bp)
    Bpp = b_prime_prime(Y_mag, Y_ang, IND_E)
    Bpp_inv = np.inv(Bpp)

    show_variable('Bp', Bp)
    show_variable('Bpp', Bpp)

    while True:
        k += 1
        # Proceso
        ## Actualizacion de P y Q
        for row, key_row in enumerate(IND_d):
            i = key_row
            P_k[row,0] = 0
            for j in range(dim):
                P_k[row,0] += E_k[i,0]*E_k[j,0]*Y_mag[i,j]*np.cos(Y_ang[i,j]-d_k[i,0]+d_k[j,0])
        
        for row, key_row in enumerate(IND_E):
            i = key_row
            Q_k[row,0] = 0
            for j in range(dim):
                Q_k[row,0] -= E_k[i,0]*E_k[j,0]*Y_mag[i,j]*np.sin(Y_ang[i,j]-d_k[i,0]+d_k[j,0])
        
        # Actualizacion de dP y dQ
        dP_k = P - P_k
        dQ_k = Q - Q_k

        # Actualizacion de dd y dE
        E_k.dtype = float
        d_k.dtype = float
        dd_k = np.matmul(Bp_inv*-1, dP_k /E_k.from_index(IND_d))
        dE_k = np.matmul(Bpp_inv*-1, dQ_k/E_k.from_index(IND_E))
        # Actulizacion de X_dk y X_Ek
        X_dk = X_dk + dd_k
        X_Ek = X_Ek + dE_k

        # Actualizacion d_k y E_k
        d_k = d_k.to_index(IND_d, X_dk)
        E_k = E_k.to_index(IND_E, X_Ek) 

        # Actualizacion del error
        error_P = np.max(np.abs(dP_k))
        error_Q = np.max(np.abs(dQ_k))
        error = max(error_P, error_Q)
        
        ## Resumen
        print('----- Iteracion {k} -----')
        show_variable('P_'+str(k), P_k)
        show_variable('Q_'+str(k), Q_k)
        show_variable('dP_'+str(k), dP_k)
        show_variable('dQ_'+str(k), dQ_k)
        
        show_variable('dE_'+str(k), dE_k)
        show_variable('dd_'+str(k), dd_k)
        show_variable('E_'+str(k), E_k)
        show_variable('d_'+str(k), d_k)
        
        print('error_'+str(k) + '= ', error)
        print()
        # Condicion
        if error < tol or k > max_steps:
            break
    print("--------- FINAL (deg) --------")
    show_variable('E', E_k)
    show_variable('d', np.degrees(d_k))
    return E_k, d_k