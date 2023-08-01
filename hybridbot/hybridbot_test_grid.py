import pytest
from pandapower.control import ConstControl

import pandapipes as pp
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import tempfile
# create empty net
import pandas as pd
import os
import pandapower.control as control
from pandapipes.component_models import Pipe
from pandapipes.timeseries import run_timeseries, init_default_outputwriter
from pandapower.timeseries import OutputWriter, DFData
from pandapipes.test.pipeflow_internals import internals_data_path
from types import MethodType


class OutputWriterTransient(OutputWriter):
    def _save_single_xls_sheet(self, append):
        raise NotImplementedError("Sorry not implemented yet")

    def _init_log_variable(self, net, table, variable, index=None, eval_function=None,
                           eval_name=None):
        if table == "res_internal":
            #when pipes with only sections are in the pipe table, res_internal gets an indexing error
            pipe_sections_one = (net.pipe.sections-1 == 0).sum()
            index = np.arange(len(net.junction) + (net.pipe.sections-1).sum() - pipe_sections_one)
        return super()._init_log_variable(net, table, variable, index, eval_function, eval_name)


def _output_writer(net, time_steps, ow_path=None):
    """
    Creating an output writer.

    :param net: Prepared pandapipes net
    :type net: pandapipesNet
    :param time_steps: Time steps to calculate as a list or range
    :type time_steps: list, range
    :param ow_path: Path to a folder where the output is written to.
    :type ow_path: string, default None
    :return: Output writer
    :rtype: pandapower.timeseries.output_writer.OutputWriter
    """

    if transient_transfer:
        log_variables = [
            ('res_junction', 't_k'), ('res_junction', 'p_bar'), ('res_pipe', 't_to_k'), ('res_internal', 't_k')
        ]
    else:
        log_variables = [
            ('res_junction', 't_k'), ('res_junction', 'p_bar'), ('res_pipe', 't_to_k')
        ]
    ow = OutputWriterTransient(net, time_steps, output_path=ow_path, log_variables=log_variables)
    return ow


def format_res_internal(res_T, net):
    res_T_df = pd.DataFrame(res_T)
    res_junction = net.res_junction[~net.res_junction.isnull().any(axis=1)]
    res_pipe = net.res_pipe[~net.res_pipe.isnull().any(axis=1)]
    sections = net.pipe.loc[res_pipe.index, 'sections']
    column_dict = {}
    sections_processed = 0

    for junction in range(res_junction.shape[0]):
        # column_dict[junction] = {junction, 'junction_'+ str(junction)}
        name = res_junction.iloc[junction].name
        column_dict[junction] = 'junction_' + str(name)

    for pipe in range(res_pipe.shape[0]):
        name = res_pipe.iloc[pipe].name
        for section in range(sections.loc[name] - 1):
            # column_dict[sections_processed + section + res_junction.shape[0]] = {section + res_junction.shape[0], 'pipesection_'+ str(pipe) + '_' + str(section)}
            column_dict[sections_processed + section + res_junction.shape[0]] = 'pipesection_' + str(name) + '_' + str(
                section)
        sections_processed += sections.loc[name]

    res_T_df.columns = list(column_dict.values())

    return res_T_df

def get_pipe_x_results(res_T_df, net, pipe_index):

    from_j = net.pipe.from_junction.iloc[pipe_index]
    to_j = net.pipe.to_junction.iloc[pipe_index]
    from_column = 'junction_' + str(from_j)
    to_column = 'junction_' + str(to_j)
    first_node = res_T_df.loc[:,from_column]
    last_node = res_T_df.loc[:,to_column]

    section_columns = res_T_df.columns[res_T_df.columns.str.startswith('pipesection_' + str(pipe_index)+ '_')]
    assert section_columns.empty == False, 'no sections found, check if pipe_index is correct'
    section_nodes = res_T_df.loc[:,section_columns]
    pipe = np.zeros(((sections + 1), res_T.shape[0]))
    pipe[0, :] = copy.deepcopy(first_node)
    pipe[-1, :] = copy.deepcopy(last_node)
    pipe[1:-1, :] = np.transpose(copy.deepcopy(section_nodes))
    return pipe

ds = DFData(pd.DataFrame({"t_k": [400] * 250 + [310] * 400}))

transient_transfer = True
hybrid = False
if hybrid==True:
    net = pp.from_json(r"C:\Users\eprade\Documents\hybridbot\heating grid\net_v21_06_ng.json")
    #net.flow_control.in_service[5] = False
    net.heat_exchanger.qext_w = 0
    sections = 20
    net.pipe.loc[net.pipe.loc[net.pipe['in_service'] == True].index, 'sections'] = sections
    t_ctrl = ConstControl(net, "circ_pump_pressure", "t_flow_k", 0, profile_name="t_k", data_source=ds)

    nodes = 2
    length = 1
else:
    net = pp.create_empty_network(fluid="water")
    # create junctions
    j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 1")
    j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 2")
    j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 3")

    # create junction elements
    ext_grid = pp.create_ext_grid(net, junction=j1, p_bar=5, t_k=330, name="Grid Connection")
    sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=2, name="Sink")

    # create branch elements
    sections = 10
    nodes = 2
    length = 1
    pp.create_pipe_from_parameters(net, j1, j2, length, 75e-3, k_mm=.0472, sections=sections,
                                   alpha_w_per_m2k=5, text_k=293)
    pp.create_pipe_from_parameters(net, j2, j3, length, 75e-3, k_mm=.0472, sections=sections,
                                   alpha_w_per_m2k=5, text_k=293)
    t_ctrl = ConstControl(net, "ext_grid", "t_k", 0, profile_name="t_k", data_source=ds)
# read in csv files for control of sources/sinks



time_steps = range(300)
dt = 10
iterations = 60
ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
run_timeseries(net, time_steps, dynamic_sim=True, transient=transient_transfer, mode="all", dt=dt,
               reuse_internal_data=True, iter=iterations)

if transient_transfer:
    res_T = ow.np_results["res_internal.t_k"]
    res_J = ow.np_results["res_junction.t_k"]
else:
    res_T = ow.np_results["res_junction.t_k"]


res_T_df = format_res_internal(res_T, net)
pipe_number = 8
pipe1 = get_pipe_x_results(res_T_df, net, pipe_index=pipe_number)
# res_T_df.to_excel('res_T.xlsx')
#




# pipe1 = np.zeros(((sections + 1), res_T.shape[0]))
# #
# pipe1[0, :] = copy.deepcopy(res_T[:, 0])
# if hybrid == True:
#     pipe1[-1, :] = copy.deepcopy(res_T[:, 2+sections])
# pipe1[-1, :] = copy.deepcopy(res_T[:, 1])
# if transient_transfer:
#     pipe1[1:-1, :] = np.transpose(copy.deepcopy(res_T[:, nodes:nodes + (sections - 1)]))
# print(pipe1) # columns: timesteps, rows: pipe segments

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_title('pipe ' + str(pipe_number))
ax.set_ylabel("Temperature [K]")
ax.set_xlabel("Length coordinate [m]")

show_timesteps = [10, 150, 160, 170, 290]
line1, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[0]], color="black",
                 marker="+", label="Time step " + str(show_timesteps[0]), linestyle="dashed")
line11, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[1]], color="red",
                  linestyle="dotted", label="Time step " + str(show_timesteps[1]))
line12, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[2]], color="blue",
                  linestyle="dashdot", label="Time step " + str(show_timesteps[2]))
line13, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[3]], color="green",
                  linestyle="dashed", label="Time step " + str(show_timesteps[3]))
line14, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[4]], color="orange",
                  linestyle="solid", label="Time step " + str(show_timesteps[4]))

ax.set_ylim((280, 405))
ax.legend()
fig.canvas.draw()
plt.show()

#print(net.res_internal)
