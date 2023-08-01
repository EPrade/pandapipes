import pandapipes as pp
from pandapower.control import ConstControl
from pandapipes.timeseries import run_timeseries, init_default_outputwriter
from pandapower.timeseries import OutputWriter, DFData
import pandas as pd
import numpy as np
import tempfile


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


    log_variables = [
        ('res_junction', 't_k'), ('res_junction', 'p_bar'), ('res_pipe', 't_to_k'), ('res_internal', 't_k')
    ]


    ow = OutputWriterTransient(net, time_steps, output_path=ow_path, log_variables=log_variables)
    return ow



net = pp.create_empty_network(fluid="water")
# create junctions
j0 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 1", geodata=[1, 0])
#j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 2")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 3", geodata=[2, 0])
j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 4", geodata=[2, 1])
j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 5", geodata=[3, 2])
j5 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 6", geodata=[3, 0])
j6 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 7", geodata=[3, 1])
j7 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 8", geodata=[2, 2])
j8 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 9", geodata=[1, 2])

# create junction elements
circ_pump = pp.create_circ_pump_const_pressure(net, j8, j0, 1,0.5, t_flow_k=333)


# create branch elements
sections = 10
nodes = 2
length = 1
pp.create_pipe_from_parameters(net, j0, j2, length, 75e-3, k_mm=.0472, sections=sections,
                               alpha_w_per_m2k=5, text_k=293)
pp.create_pipe_from_parameters(net, j2, j5, length, 75e-3, k_mm=.0472, sections=sections,
                               alpha_w_per_m2k=5, text_k=293)
pp.create_pipe_from_parameters(net, j4, j7, length, 75e-3, k_mm=.0472, sections=sections,
                               alpha_w_per_m2k=5, text_k=293)
pp.create_pipe_from_parameters(net, j7, j8, length, 75e-3, k_mm=.0472, sections=sections,
                               alpha_w_per_m2k=5, text_k=293)

# pp.create_valve(net, from_junction=j1, to_junction=j2, diameter_m=0.2, opened=True)
# pp.create_valve(net, from_junction=j4, to_junction=j5, diameter_m=0.2, opened=True)
pp.create_flow_control(net, from_junction=j2, to_junction=j3, controlled_mdot_kg_per_s=0, diameter_m=0.2)
pp.create_flow_control(net, from_junction=j5, to_junction=j6, controlled_mdot_kg_per_s=0.05, diameter_m=0.2)
pp.create_heat_exchanger(net, from_junction=j3, to_junction=j7, diameter_m=0.2, qext_w=50)
pp.create_heat_exchanger(net, from_junction=j6, to_junction=j4, diameter_m=0.2, qext_w=50)


ds = DFData(pd.DataFrame({"t_k": [400] * 250 + [310] * 400}))
t_ctrl = ConstControl(net, "circ_pump_pressure", "t_flow_k", 0, profile_name="t_k", data_source=ds)
transient_transfer = True

#set flow controls with mdot==0 and corresponding heatexchangers OOS
# net.flow_control.in_service[net.flow_control.controlled_mdot_kg_per_s==0] = False
# fc_junction = net.flow_control.to_junction[net.flow_control.in_service == False].values
# net.heat_exchanger.in_service[net.heat_exchanger.from_junction==fc_junction] = False
# fc2_junction = net.flow_control.from_junction[net.flow_control.in_service == False].values
# net.valve.opened[net.valve.to_junction==fc2_junction] = False

time_steps = range(300)
dt = 100
iterations = 60
ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
run_timeseries(net, time_steps, dynamic_sim=True, transient=transient_transfer, mode="all", dt=dt,
               reuse_internal_data=True, iter=iterations)