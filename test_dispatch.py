from dispatch import Dispatch

costs_params = [
    [260, 10.2, 0.02], 
    [320, 11.3, 0.021], 
    [280, 13.1, 0.024],
    [270, 12.5, 0.029]
]
restrictions = [
    [110, 440], 
    [161, 520], 
    [120, 750],
    [135, 670]
]

case = Dispatch(
    costs_params=costs_params,
    restrictions=restrictions
)


case.optimize(600)
print(case.optimization)