from system import PowerSystem

cc = PowerSystem()
#nodes
cc.add_node(1, "SLACK", E=1.01, d=0, P_d=3, Q_d=1)
cc.add_node(2, "PV", E=1.03, P_g=6.2, P_d=2, Q_d=0.5)
cc.add_node(3, "PV", E=1.02, P_g=3, P_d=6.1, Q_d=3)
#lines

cc.add_line(1, 2, 1/(0.009+0.035j), 0)
cc.add_line(1, 3, 1/(0.01+0.04j), 0)
cc.add_line(2, 3, 1/(0.008+0.03j), 0)

cc.compute_Y()
cc.solve()
cc.analyze(loss="m1")


    
