import numpy_CAS as np
import powerflow as pf
import analisys as lis

class PowerSystem:
    def __init__(self, system = {}, Y = None):
        self.system = system
        self.Y = Y
        self.E = None
        self.d = None
        self.V = None
        self.IND = None

        self.I_system = None
        self.I_injected = None
        self.S_injected = None
        self.S_transfer = None
        self.loss = None

    def compute_Y(self):
        self.Y = lis.admittance_matrix(self.system)

    def add_node(self, id, type, E=None, d=None, P_g=None, Q_g=None, P_d=0, Q_d=0):
        info = {
            "type": type,
            "E": E,
            "d": d,
            "P_g": P_g,
            "Q_g": Q_g,
            "P_d": P_d,
            "Q_d": Q_d,
            "conn": [],
            "conn_gnd": 0,
        }
        new_node = {id: info}
        self.system ={key: value for d in (self.system, new_node) for key, value in d.items()}

    def add_line(self, from_id, to_id, admitance_line, admitance_gnd):
        self.system[from_id]["conn"].append((to_id, admitance_line))
        self.system[from_id]["conn_gnd"] += admitance_gnd/2

        self.system[to_id]["conn"].append((from_id, admitance_line))
        self.system[to_id]["conn_gnd"] += admitance_gnd/2

    def reset(self):
        self.system = {}


    def solve(self, fast = True, tol = 0.05):
        X0, V0, self.IND = pf.flat_profile(self.system)
        C = pf.compute_C(self.system, self.IND)
        if fast:
            self.E, self.d = pf.newton_raphson_fast_decoupled(self.Y, X0, V0, C, self.IND, tol)
        else:
            self.E, self.d = pf.newton_raphson(self.Y, X0, V0, C, self.IND, tol)
        self.V = self.E * np.exp(self.d*1j)
        self.V.dtype = complex
        return
    
    def analyze(self, loss="m2"):
        self.I_system = lis.system_current(self.Y, self.V)
        self.I_injected = lis.injected_current(self.Y, self.V)
        self.S_injected = lis.injected_power(self.Y, self.V)
        self.S_transfer = lis.transfered_power(self.V, self.I_system)
        
        if loss == "m1":
            self.loss = lis.power_loss_m1(self.V, self.I_injected)
        elif loss == "m2":
            self.loss = lis.power_loss_m2(self.S_injected)
        elif loss == "m3":
            self.loss = lis.power_loss_m3(self.S_transfer)
        return

