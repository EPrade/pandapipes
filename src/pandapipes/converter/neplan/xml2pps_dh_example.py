import pandapipes.topology
from pandapipes.plotting import plot_pressure_profile
from xml_toolbox import *
#-----------------------------------------------

xml_path = r"C:\Users\eprade\Documents\SW Aalen\FW\NeplanFernwaerme.neplst360"
loads_path = r"C:\Users\eprade\Documents\SW Aalen\FW\2024-03-25_AW001_Verbraucher.xlsx"


elements = {'bus_name':'NodeList', 'line_name':'LineList', 'load_name':'DeviceList'}


node_columns = ['ElementType', 'Name', 'AliasName1', 'X', 'Y', 'Elevation']
line_columns = ['Name', 'AliasName1', 'LibraryType','Length','NodeName', 'NodeName','CoordinateList']
device_columns = ['ElementType', 'Name', 'AliasName1', 'LibraryType', 'NodeName', 'NodeName']

#specify element type for devices
element_type_device1 = 'HeatingLoad'
element_type_device2 = 'HeatingValve'
columns = [node_columns, line_columns, device_columns]




data = extract_data(xml_path, elements, columns)

fluid='water'
heat_loads = pd.read_excel(loads_path)
net = create_pps_net(data, elements,fluid, element_type_device1,element_type_device2, heat_loads)

#create sources
#wk2
flowj = 'HZ0006_VL'
returnj = 'HZ0006_RL'
flowj3 = 'HZ0010_VL'
returnj3= 'HZ0010_RL'
fj = net.junction[net.junction['name']=='HZ0006_VL'].index[0]
rj = net.junction[net.junction['name']=='HZ0006_RL'].index[0]
fj3 = net.junction[net.junction['name']=='HZ0010_VL'].index[0]
rj3 = net.junction[net.junction['name']=='HZ0010_RL'].index[0]
pandapipes.create_circ_pump_const_pressure(net, rj, fj,6,1, 283.15)
pandapipes.to_json(net, r"C:\Users\eprade\Documents\SW Aalen\FW\fwppnet2.json")

pandapipes.create_flow_control(net, rj3, fj3, 50, 0.1)
#pandapipes.create_circ_pump_const_mass_flow(net, rj, fj, 6, 0.5, 283.15)
#wk3
# pandapipes.create_junction(net, 6,283,100, 'wk2', 900)
#pandapipes.create_pressure_control(net, rj3, fj3, fj3, 6)
#pandapipes.create_heat_exchanger(net, 900, fj3, 0.5, -500)


#pipeflow
#net.heat_consumer.controlled_mdot_kg_per_s =0
net.junction.height_m = 0
net.pipe.diameter_m *= 10
pandapipes.pipeflow(net)
plot_pressure_profile(net, x0_junctions = [426])
net.heat_consumer.controlled_mdot_kg_per_s *=0.1
net.pipe.diameter_m *= 10
pandapipes.pipeflow(net)

#plotting
pandapipes.create_sink(net, fj, 0.5)
pandapipes.create_sink(net, fj3, 0.5)
#pandapipes.plotting.simple_plot(net, junction_size=0.01, heat_consumer_size=0.2, heat_consumer_color='b', plot_sinks=True)
unsupplied_junctions = pandapipes.topology.unsupplied_junctions(net)
#flow junctions
junction_list = [426,425,424,488,491]
plot_network(net, connotation_size=.5)
#return junctions
# junction_list = [427,534,533,863,588,866]
# plot_network(net, junction_list)








