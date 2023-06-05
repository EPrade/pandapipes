from os.path import join
import pandas as pd
import numpy as np
import pandapipes as pps

import seaborn as sb
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import geopandas as gp
from pandapower.plotting import create_annotation_collection
from shapely.geometry import LineString

if __name__ == "__main__":
    fluid = "water"

    net = pps.create_empty_network(fluid=fluid)
    pps.create_junctions(net, 8, 1.05, 273.15+10)
    pps.create_flow_controls(net, [1,4], [2,5], 0.1, 0.05)
    pps.create_heat_exchangers(net, [2,5], [3,6], 0.05, 50000)
    pps.create_pipes_from_parameters(net, [0,1,6,3], [1,4,3,7], 5, 0.1, alpha_w_per_m2k=0)
    pps.create_circ_pump_const_pressure(net, 7, 0, 9, 2.6,
                                    t_flow_k=273.15+90)
    from pandapipes import topology as tp

    import networkx as nx
    from matplotlib import pyplot as plt

    ng = tp.create_nxgraph(net)
    pos = nx.planar_layout(ng)
    # pos = nx.spiral_layout(ng)
    # pos = nx.circular_layout(ng)
    # pos = nx.bipartite_layout(ng)
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    nx.draw_networkx(ng, pos=pos, ax=ax)

    plt.show()
    pps.pipeflow(net, mode='all', transient=False)
    # plot.simple_plot(net, junction_size=0.25, heat_exchanger_size=0.05, flow_control_size=0.00005,
    #                  flow_control_color='blue')