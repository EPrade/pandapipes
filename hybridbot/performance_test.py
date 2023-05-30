from timeit import timeit
import time
import pandapipes as pp
from pandapipes.networks.simple_gas_networks import gas_3parallel, gas_meshed_delta, schutterwald
from pandapipes.plotting import simple_plot
import_module = "import pandapipes as pps"

net = pp.create_empty_network("net", "hgas")



# create junction
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 2")
j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3")
j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 4")


# create junction elements
ext_grid = pp.create_ext_grid(net,fluid= "hgas", junction=j1, p_bar=1.1, t_k=293.15, name="Grid Connection")
sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=0.045, name="Sink 1")
sink1 = pp.create_sink(net, junction=j4, mdot_kg_per_s=0.045, name="Sink 2")
# create branch element
pipe = pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=0.1, diameter_m=0.05, name="Pipe 1")
pipe1 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=0.1, diameter_m=0.05, name="Pipe 2")
pipe2 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j4, length_km=0.1, diameter_m=0.05, name="Pipe 3")



net_schutter = schutterwald()
#valve = pp.create_valve(net, from_junction=j2, to_junction=j3, diameter_m=0.05, opened=True, name="Valve 1")
# net = gas_3parallel()
# net1= gas_meshed_delta()

#simple_plot(net, plot_sinks=True, plot_sources=True)

#
# pps.create_fluid_from_lib(net, "hgas")
# pps.get_fluid(net, "hgas")

def time_pf(a_net):
    pp.pipeflow(a_net)
    return


# starttime = timeit()
# time_pf(oes)
# print("The time difference is :", timeit() - starttime)
#
# starttime = timeit()
# time_pf(net)
# print("The time difference is :", timeit() - starttime)


start_time = time.time()
time_pf(net_schutter)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
time_pf(net)
print("--- %s seconds ---" % (time.time() - start_time))


#method txt-files
#pipeflow()

#method external packages
#pipeflow()