# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from xml_toolbox import extract_data
from extraction_setup import columns, element_type_device1, element_type_device2, elements, fluid
import pandapipes
import pandas as pd
import numpy as np

def pps_dh_net_from_neplan(path, net_name, path_std_lib):
    """
    Creates pandapipes district heating net from neplan xml-file. Additionally takes a file for the pipe standard types which
    can exported from neplan as a .txt file as an input for the pipe creation.
    :param path: path to .xml file
    :type path: str
    :param net_name: name of pandapipes net to be created
    :type net_name: str
    :param path_std_lib: path to the .txt file for the pipe library
    :type path_std_lib: str
    :return: pandapipes dh net
    :rtype: pandapipes net
    """
    columns, element_type_device1, element_type_device2, elements, fluid = get_setup()
    data = extract_data(path, elements, columns)
    net = pandapipes.create_empty_network(net_name, fluid)
    try:

        junction_dic, junction_geodata = create_junction(data[elements['junction_name']])

        pandapipes.create_junctions(net, len(junction_dic['name']), pn_bar=6, tfluid_k=293.15, height_m=junction_dic['height_m'],
                                    name= junction_dic['name'], geodata=junction_geodata, name2=junction_dic['name2'])
        print(" \n --junctions created successfully--  \n")
    except Exception as e:
        print("Warning: No junctions created - ", e)

    try:
        # todo: implement distinguishing between pipes with std_type_lib and pipes without
        pipe_dic = create_pipes(data[elements['pipe_name']])
        pipe_parameters = read_std_type_pipe(pipe_dic['std_type'], path_std_lib)

        from_indices = []
        for name in pipe_dic['from_junction']:
            # Get indices where the name matches in the DataFrame
            matching_indices = np.where(net.junction['name'].values == name)[0]
            # Append the matching indices to the list
            from_indices.extend(matching_indices)
        to_indices = []
        for name in pipe_dic['to_junction']:
            # Get indices where the name matches in the DataFrame
            matching_indices = np.where(net.junction['name'].values == name)[0]
            # Append the matching indices to the list
            to_indices.extend(matching_indices)

        # values are a string (20,7) instead of a float (20.7)
        k_mm = pd.to_numeric(pipe_parameters['K'].str.replace(',', '.'))
        u_w_per_m2k = pd.to_numeric(pipe_parameters['HeatTransferCoefficient'].str.replace(',', '.'))
        diameter_m = pd.to_numeric(pipe_parameters['DI'].str.replace(',', '.'))/1000

        pandapipes.create_pipes_from_parameters(net, from_indices, to_indices, pipe_dic['length_km'], diameter_m, k_mm,
                                     loss_coefficient=0, sections=1, u_w_per_m2k=u_w_per_m2k , text_k=293,
                                     qext_w=0., name=pipe_dic['name'])
        print(" \n --pipes created successfully -- \n")
    except Exception as e:
        print("Warning: No pipes created - ", e)

    try:


        loads = create_empty_heat_load(data[elements['consumer_name']], element_type_device1)

        junction_index_dict = {name: index for index, name in enumerate(net.junction['name'])}
        # Create a list of indices by mapping the 'from_node' values to their corresponding indices in 'net.junction'
        from_indices = [junction_index_dict[node] for node in loads['from_node'] if node in junction_index_dict]
        to_indices = [junction_index_dict[node] for node in loads['to_node'] if node in junction_index_dict]
        x = net.junction_geodata.iloc[from_indices]['x']
        y = net.junction_geodata.iloc[from_indices]['y']

        pandapipes.create_heat_consumers(net, from_indices, to_indices, 0.05 , qext_w=50, controlled_mdot_kg_per_s=0.001,
                                         name=loads['name'],from_nodes=loads['from_node'], X=x, Y=y)
        print(" \n --heat consumers created successfully -- \n")
    except Exception as e:
        print("Warning: Heat consumers not created - ", e)

    try:

        valves = create_valve(data[elements['consumer_name']], element_type_device2, path_std_lib)
        # Create a dictionary mapping 'name' to its index in 'net.junction'
        junction_index_dict = {name: index for index, name in enumerate(net.junction['name'])}
        # Create a list of indices by mapping the 'from_node' values to their corresponding indices in 'net.junction'
        from_indices = [junction_index_dict[node] for node in valves['from_node'] if node in junction_index_dict]
        to_indices = [junction_index_dict[node] for node in valves['to_node'] if node in junction_index_dict]

        pandapipes.create_valves(net, from_indices, to_indices, valves['diameter'])

        print(" \n --valves created successfully -- \n")
    except Exception as e:
        print("Warning: valves not created - ", e)

    return net

def get_setup():
    return columns, element_type_device1, element_type_device2, elements, fluid

def create_junction(data):
    """
    Collects and prepares Parameters for pandapipes junction creation
    :param data:
    :type data:
    :return:
    :rtype:
    """
    junction_dic = {}
    junction_dic['height_m'] = data['Elevation']
    junction_dic['name'] = data['Name']
    junction_dic['name2'] = data['AliasName1']
    junction_geodata = list(zip(data['X'], data['Y']))
    junction_geodata_float = [(float(x), float(y)) for x, y in junction_geodata]
    return junction_dic, junction_geodata_float

def create_pipes(data):
    """
    Collects and prepares Parameters for pandapipes pipes creation
    :param data:
    :type data:
    :return:
    :rtype:
    """
    pipe_dic = {}
    pipe_dic['from_junction'] = data['NodeName1']
    pipe_dic['to_junction'] = data['NodeName2']
    pipe_dic['length_km'] = data['Length'].astype(float)/1000
    pipe_dic['std_type'] = data['LibraryType']
    pipe_dic['name'] = data['Name']
    return pipe_dic

def read_std_type_pipe(std_type, path_std_lib):
    std_type_library = pd.read_csv(path_std_lib, sep='\t', header=2)
    df_std_type = pd.DataFrame(std_type, columns=['LibraryType'])
    parameters = pd.merge(df_std_type, std_type_library, on='LibraryType', how='left')
    return parameters


def create_empty_heat_load(data, element_type_device):
    """
    Collects and prepares Parameters for pandapipes heat consumer creation
    :param data:
    :type data:
    :param element_type_device:
    :type element_type_device:
    :return:
    :rtype:
    """
    heat_load = data.loc[data['ElementType'] == element_type_device]
    heat_load_parameters = pd.DataFrame(index=range(len(heat_load)))
    heat_load.reset_index(drop=True, inplace=True)
    heat_load_parameters['name'] = heat_load['Name']
    heat_load_parameters['from_node'] = heat_load['NodeName1']
    heat_load_parameters['to_node'] = heat_load['NodeName2']
    return heat_load_parameters

def create_heat_load(data, element_type_device, heat_loads):
    """
    Collects and prepares Parameters for pandapipes heat consumer creation
    :param data:
    :type data:
    :param element_type_device:
    :type element_type_device:
    :param heat_loads:
    :type heat_loads:
    :return:
    :rtype:
    """
    heat_load = data.loc[data['ElementType'] == element_type_device]
    selected_columns = ['Name', 'T Rückspeise\n[°C]', 'Q Auswahl', 'Q (N)', 'Q (M)', 'Durchmesser\n[mm]', 'Vorlauftemperatur','ΔP [bar]']
    heat_loads_selected = heat_loads[selected_columns]
    name_condition_max = heat_loads_selected['Q Auswahl'] == 'Maximaler Verbrauch'
    name_condition_n = heat_loads_selected['Q Auswahl'] == 'Nominaler Verbrauch'
    # Use boolean indexing to select rows where the condition is True
    Qmax_index = heat_loads_selected[name_condition_max].index
    Qn_index = heat_loads_selected[name_condition_n].index
    heat_loads_selected['Q_selected'] = pd.Series([float('nan')] * len(heat_loads_selected))

    # Populate the new column based on the indices lists
    for index in Qmax_index:
        heat_loads_selected.loc[index, 'Q_selected'] = heat_loads_selected.loc[index, 'Q (M)']
    for index in Qn_index:
        heat_loads_selected.loc[index, 'Q_selected'] = heat_loads_selected.loc[index, 'Q (N)']

    heat_load_parameters = pd.DataFrame(index=range(len(heat_load)))
    heat_load.reset_index(drop=True, inplace=True)
    heat_load_parameters['name'] = heat_load['Name']
    heat_load_parameters['from_node'] = heat_load['NodeName0']
    heat_load_parameters['to_node'] = heat_load['NodeName1']

    heat_load_parameters['Q'] = \
        heat_loads_selected[heat_loads_selected['Name'].isin(heat_load['Name'])]['Q_selected']
    heat_load_parameters['di'] = \
        heat_loads_selected[heat_loads_selected['Name'].isin(heat_load['Name'])]['Durchmesser\n[mm]']
    heat_load_parameters['deltapbar'] = \
        heat_loads_selected[heat_loads_selected['Name'].isin(heat_load['Name'])]['ΔP [bar]']
    heat_load_parameters['Tflow'] = \
        heat_loads_selected[heat_loads_selected['Name'].isin(heat_load['Name'])]['Vorlauftemperatur']
    heat_load_parameters['Treturn'] = \
        heat_loads_selected[heat_loads_selected['Name'].isin(heat_load['Name'])]['T Rückspeise\n[°C]']

    return heat_load_parameters

def create_valve(data, element_type, path_std_lib):
    """
    Collects and prepares Parameters for pandapipes valve creation
    :param data:
    :type data:
    :param element_type:
    :type element_type:
    :param path_std_lib:
    :type path_std_lib:
    :return:
    :rtype:
    """
    data_valve = data.loc[data['ElementType'] == element_type]
    valve_std_type_param = read_std_type_pipe(data_valve['LibraryType'], path_std_lib)
    valve_parameters = {}
    valve_parameters['std_type'] = data_valve['LibraryType']
    valve_parameters['from_node'] = data_valve['NodeName1']
    valve_parameters['to_node'] = data_valve['NodeName2']
    valve_parameters['opened'] = True
    valve_parameters['diameter'] = 500e-3
    return valve_parameters
