from os.path import join
import pandas as pd
import numpy as np
import pandapipes

import seaborn as sb
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import geopandas as gp
from pandapower.plotting import create_annotation_collection
from shapely.geometry import LineString

if __name__ == "__main__":
    fluid = "water"

    # create empty network
    net = pandapipes.create_empty_network("net")

    # create fluid
    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    # create junctions
    junction1 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=290, name="Junction 1")
    junction2 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=290, name="Junction 2")
    junction3 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=290, name="Junction 3")
    junction4 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=290, name="Junction 4")
    # create external grid
    pandapipes.create_ext_grid(net, junction=junction1, p_bar=6, t_k=363.15, name="External Grid", type="pt")
    # create sinks
    pandapipes.create_sink(net, junction=junction3, mdot_kg_per_s=1, name="Sink 1")
    pandapipes.create_sink(net, junction=junction4, mdot_kg_per_s=2, name="Sink 2")

    # create pipes
    pandapipes.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2, length_km=0.1,
                                           diameter_m=0.075, k_mm=0.025, sections=5, alpha_w_per_m2k=0,
                                           text_k=298.15, name="Pipe 1")

    pandapipes.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction3, length_km=2,
                                           diameter_m=0.05, k_mm=0.025, sections=4, alpha_w_per_m2k=0,
                                           text_k=298.15, name="Pipe 2")

    pandapipes.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction4, length_km=1,
                                           diameter_m=0.1, k_mm=0.025, sections=8, alpha_w_per_m2k=0,
                                           text_k=298.15, name="Pipe 3")
    pandapipes.pipeflow(net, mode="all")