class Dispatch:
    def __init__(self, costs_funcs = [], restrictions = []) -> None:
        self.costs_funcs = costs_funcs
        self.restrictions = restrictions
        self.la = None
        self.cost_dist = None
        self.P_dist = None

    def _(Pd, costs):
        sum_1 = 0
        sum_2 = 0
        for params in costs:
            sum_1 += params[1]/(2*params[2])
            sum_2 += 1/(2*params[2])
        return (Pd + sum_1)/sum_2

    def plain_optimize(Pd):

        return
    
    def optimize(Pd):
        return