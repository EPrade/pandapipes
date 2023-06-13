from os.path import join
import pandas as pd
import numpy as np
import pandapipes
from pandapipes.test.pipeflow_internals.test_transient import _output_writer
import seaborn as sb
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import geopandas as gp
from pandapower.plotting import create_annotation_collection
from shapely.geometry import LineString
from pandapipes.timeseries import run_timeseries
import tempfile

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
    pandapipes.create_ext_grid(net, junction=junction1, p_bar=12, t_k=363.15, name="External Grid", type="pt")
    # create sinks
    pandapipes.create_sink(net, junction=junction3, mdot_kg_per_s=200, name="Sink 1")
    pandapipes.create_sink(net, junction=junction4, mdot_kg_per_s=200, name="Sink 2")

    # create pipes
    pandapipes.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2, length_km=1,
                                           diameter_m=0.5, k_mm=0.01, sections=5, alpha_w_per_m2k=0,
                                           text_k=298.15, name="Pipe 1")

    pandapipes.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction3, length_km=1,
                                           diameter_m=0.35, k_mm=0.01, sections=4, alpha_w_per_m2k=0,
                                           text_k=298.15, name="Pipe 2")

    pandapipes.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction4, length_km=1,
                                           diameter_m=0.35, k_mm=0.01, sections=8, alpha_w_per_m2k=0,
                                           text_k=298.15, name="Pipe 3")

    dt = 60
    time_steps = range(15)
    ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
    run_timeseries(net, time_steps, transient=True, mode="all", iter=10, dt=dt)
    res_T = ow.np_results["res_internal.t_k"]
    res_junction = ow.np_results["res_junction.t_k"]
    res_pipe_to = ow.np_results["res_pipe.t_to_k"]
    res_pipe_from = ow.np_results["res_pipe.t_from_k"]

    pandapipes.pipeflow(net, mode="all")