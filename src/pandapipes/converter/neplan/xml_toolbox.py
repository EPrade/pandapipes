import pandas as pd
import sys
import xml.etree.ElementTree as ET
import pandapower
import pandapipes
import numpy as np
from pandapipes import plotting
import matplotlib.pyplot as plt
from pandapower.plotting import create_annotation_collection
def xml_to_dataframe(element):
    data = []
    dic_list = []
    columns = []
    i = 0

    for child in element:
        row = {}
        for child2 in child:
            if not columns:
                columns.append(child2.tag)

            row[child2.tag] = child2.text
            dic = {'Element_no': i, 'name': child2.tag}

            if child2.text is not None and child2.text.strip() == "":
                for child3 in child2:
                    dic[child3.tag] = child3.text
                    dic2 = {'Element_no': i, 'name': child3.tag}

                    if child3.text is not None and child3.text.strip() == "":
                        for child4 in child3:
                            dic2[child4.tag] = child4.text
                            dic3 = {'Element_no': i, 'name': child4.tag}

                            if child4.text is not None and child4.text.strip() == "":
                                ii = 0
                                for child5 in child4:
                                    dic3[child5.tag + str(ii)] = child5.text
                                    ii += 1
                                dic_list.append(dic3)
                        dic_list.append(dic2)
                dic_list.append(dic)
            dic[child2.tag] = child2.text
            dic_list.append(dic)
        data.append(row)
        i += 1

    return pd.DataFrame(data, columns=columns), dic_list

def get_df_fast_gpt(namen, dictionaries):
    df = pd.DataFrame(columns=namen)
    key_value_pairs = [(d.get(key), key) for d in dictionaries for key in d.keys() if key in namen]
    # Überprüfung auf doppelte Namen
    doppelte_namen = set([name for name in namen if namen.count(name) > 1])
    if doppelte_namen:
        positionen = {}
        for index, name in enumerate(namen):
            if namen.count(name) > 1:
                positionen.setdefault(name, []).append(index)
        angepasste_namen = namen.copy()
        for name in doppelte_namen:
            for counter, _ in enumerate(positionen[name]):
                neuer_name = f"{name}{counter}"
                index_to_insert = positionen[name][counter] + counter
                angepasste_namen.insert(index_to_insert, neuer_name)
                angepasste_namen.remove(name)
        df = pd.DataFrame(columns=angepasste_namen)
        namen = angepasste_namen
        doubling_names_list = positionen[name].copy()
    # Extrahieren der Element_no-Werte
    element_nos = [d.get("Element_no") for d in dictionaries]
    element_no = 0

    # Iteration über die Schlüssel-Wert-Paare
    for i, (value, key) in enumerate(key_value_pairs):
        # Fortschrittsanzeige
        sys.stdout.write('\r')
        sys.stdout.write(
            "[%-20s] %d%%" % ('=' * int(20 * (i + 1) / len(key_value_pairs)), 100 * (i + 1) / len(key_value_pairs)))
        sys.stdout.flush()
        if key == namen[0] and i != 0:
            element_no += 1
            try:
                doubling_names_list = positionen[name].copy()
            except:
                continue
        if key == 'VariableName':
            df.at[element_no, value] = key_value_pairs[i + 1][0] if i < len(key_value_pairs) - 1 else None
        if key == 'NodeName':
            try:
                idx = doubling_names_list[0]
                df.iloc[element_no, idx] = value
                del doubling_names_list[0]
            except:
                continue

        elif key != 'Value':
            df.at[element_no, key] = value
    # Ausgabe des Dataframes
    return df

def get_root_indices(root, element_list):
    list1 = [element.tag for element in root]
    return [index for index, string1 in enumerate(list1) if string1 in element_list.values()]

def extract_data(path, element_list, columns):
    tree, root = parse_data(path)
    root_indices = get_root_indices(root, element_list)
    dataframes = {}
    x = 0
    for i in element_list.values():
        print('\n' + 'extracting element :' +str(x+1) + ' / ' +str(len(element_list)))
        df, dic = xml_to_dataframe(root[root_indices[x]])
        element_columns = columns[x]
        dataframes[i] = get_df_fast_gpt(element_columns, dic)
        x+=1
    return dataframes

def parse_data(path):
    with open(path, encoding="ansi") as f:
        tree = ET.parse(f)
        root = tree.getroot()
    return tree,root


def create_pp_net(data, element_list):
    net = pandapower.create_empty_network()
    try:

        bus_dic, bus_geodata = create_bus(data[element_list['bus_name']])
        pandapower.create_buses(net, len(bus_dic['vn_kv']), bus_dic['vn_kv'], name=bus_dic['name'], geodata=bus_geodata,
                                name2=bus_dic['name2'])
        print(" \n --buses created successfully--  \n")
    except Exception as e:
        print("Warning: No buses created - ", e)

    try:

        line_dic = create_lines(data[element_list['line_name']])
        from_index = net.bus[net.bus['name'].isin(line_dic['from_bus'])].index
        to_index = net.bus[net.bus['name'].isin(line_dic['to_bus'])].index
        pandapower.create_lines(net, from_index, to_index, line_dic['length_km'],
                                line_dic['std_type'], name=line_dic['name'])
        print(" \n --lines created successfully -- \n")
    except Exception as e:
        print("Warning: No lines created - ", e)

    try:

        load_dic = create_load(data[element_list['load_name']])
    except Exception as e:
        print("Warning: No load data found - ", e)

    try:

        trafo_dic = create_trafo(data[element_list['trafo_name']])
    except Exception as e:
        print("Warning: No trafo data found - ", e)


    return net


def create_pps_net(data, element_list, fluid, element_type_device, element_type_device2, heat_loads):
    net = pandapipes.create_empty_network(fluid=fluid)
    try:

        junction_dic, junction_geodata = create_junction(data[element_list['bus_name']])

        pandapipes.create_junctions(net, len(junction_dic['name']), pn_bar=6, tfluid_k=293.15, height_m=junction_dic['height_m'],
                                    name= junction_dic['name'], geodata=junction_geodata, name2=junction_dic['name2'])
        print(" \n --junctions created successfully--  \n")
    except Exception as e:
        print("Warning: No junctions created - ", e)

    try:

        pipe_dic = create_pipes(data[element_list['line_name']])
        pipe_parameters = read_std_type_pipe(pipe_dic['std_type'])

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

        loads = create_heat_load(data[element_list['load_name']], element_type_device, heat_loads)
        cp = 4.195
        mdot = loads['Q'] / (loads['Tflow']-loads['Treturn']) / cp
        # Create a dictionary mapping 'name' to its index in 'net.junction'
        junction_index_dict = {name: index for index, name in enumerate(net.junction['name'])}
        # Create a list of indices by mapping the 'from_node' values to their corresponding indices in 'net.junction'
        from_indices = [junction_index_dict[node] for node in loads['from_node'] if node in junction_index_dict]
        to_indices = [junction_index_dict[node] for node in loads['to_node'] if node in junction_index_dict]
        x = net.junction_geodata.iloc[from_indices]['x']
        y = net.junction_geodata.iloc[from_indices]['y']
        pandapipes.create_heat_consumers(net, from_indices, to_indices,loads['di']/1000,qext_w=loads['Q'],
                                         controlled_mdot_kg_per_s=mdot, name=loads['name'],from_nodes=loads['from_node']
                                         , X=x, Y=y)
        print(" \n --heat consumers created successfully -- \n")
    except Exception as e:
        print("Warning: Heat consumers not created - ", e)

    try:
        #valves modeled as heat consumer cause they should not interrupt heat flow
        valves = create_valve(data[element_list['load_name']], element_type_device2)
        # Create a dictionary mapping 'name' to its index in 'net.junction'
        junction_index_dict = {name: index for index, name in enumerate(net.junction['name'])}
        # Create a list of indices by mapping the 'from_node' values to their corresponding indices in 'net.junction'
        from_indices = [junction_index_dict[node] for node in valves['from_node'] if node in junction_index_dict]
        to_indices = [junction_index_dict[node] for node in valves['to_node'] if node in junction_index_dict]
        pandapipes.create_valves(net, from_indices, to_indices, valves['diameter'])
        #pandapipes.create_pressure_controls(net, from_indices, to_indices, valves['diameter'], controlled_p_bar=6, control_active=False)
        print(" \n --valves created successfully -- \n")
    except Exception as e:
        print("Warning: valves not created - ", e)





    return net

def create_bus(data):
    bus_dic = {}
    bus_dic['vn_kv'] = data['Un']
    bus_dic['name'] = data['Name']
    bus_dic['name2'] = data['AliasName1']
    bus_geodata = list(zip(data['X'], data['Y']))
    bus_geodata_float = [(float(x), float(y)) for x, y in bus_geodata]
    return bus_dic, bus_geodata_float

def create_junction(data):
    bus_dic = {}
    bus_dic['height_m'] = data['Elevation']
    bus_dic['name'] = data['Name']
    bus_dic['name2'] = data['AliasName1']
    bus_geodata = list(zip(data['X'], data['Y']))
    bus_geodata_float = [(float(x), float(y)) for x, y in bus_geodata]
    return bus_dic, bus_geodata_float

def create_lines(data):
    line_dic = {}
    line_dic['from_junction'] = data['NodeName0']
    line_dic['to_junction'] = data['NodeName1']
    line_dic['length_km'] = data['Length']
    line_dic['std_type'] = data['LibraryType']
    r_ohm_per_km =0
    x_ohm_per_km =0
    c_nf_per_km=0
    r0_ohm_per_km=0
    x0_ohm_per_km=0
    c0_nf_per_km=0
    max_i_ka=0
    line_dic['name'] = data['Name']
    return line_dic

def create_pipes(data):
    line_dic = {}
    line_dic['from_junction'] = data['NodeName0']
    line_dic['to_junction'] = data['NodeName1']
    line_dic['length_km'] = data['Length'].astype(float)/1000
    line_dic['std_type'] = data['LibraryType']
    line_dic['name'] = data['Name']
    return line_dic

def read_std_type_pipe(std_type):
    std_type_library = pd.read_csv(r"C:\Users\eprade\Documents\SW Aalen\FW\Bibliothek SWA_Rohre.txt", sep='\t', header=2)
    df_std_type = pd.DataFrame(std_type, columns=['LibraryType'])
    parameters = pd.merge(df_std_type, std_type_library, on='LibraryType', how='left')
    return parameters

def create_load(data):
    bus = 0
    return

def create_heat_load(data, element_type_device, heat_loads):
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

def create_valve(data, element_type):
    data_valve = data.loc[data['ElementType'] == element_type]
    valve_std_type_param = read_std_type_pipe(data_valve['LibraryType'])
    valve_parameters = {}
    valve_parameters['std_type'] = data_valve['LibraryType']
    valve_parameters['from_node'] = data_valve['NodeName0']
    valve_parameters['to_node'] = data_valve['NodeName1']
    valve_parameters['opened'] = True
    valve_parameters['diameter'] = 500e-3
    return valve_parameters

def create_trafo(data):
    hv_bus=0
    lv_bus=0
    sn_mva=0
    vn_hv_kv=0
    vn_lv_kv=0
    vkr_percent=0
    vk_percent=0
    pfe_kw=0
    i0_percent=0
    vector_group=0
    vk0_percent=0
    vkr0_percent=0
    mag0_percent=0
    mag0_rx=0
    si0_hv_partial=0

    return

def plot_network(network, junction_to_show=None, connotation_size=5):
    junctions = network.junction.index

    collections = list()

    collections.append(plotting.create_junction_collection(network, junctions=junctions, size=0.1, color="grey", zorder=1))

    pipes = network.pipe.index
    collections.append(plotting.create_pipe_collection(network, pipes=pipes,
                                                       linewidth=2, color="#A6BBC8", linestyle="dashed", zorder=0))
    if junction_to_show==None:

        junct_coords = network.junction_geodata.values
        jic = create_annotation_collection(size=connotation_size, texts=np.char.mod('%.0f', network.junction.index),
                                          coords=junct_coords, zorder=5000, color='k')
    else:
        junct_coords = network.junction_geodata.loc[junction_to_show].values
        jic = create_annotation_collection(size=connotation_size, texts=np.char.mod('%.0f', network.junction.index),
                                           coords=junct_coords, zorder=5000, color='k')
    collections.append(jic)
    plotting.draw_collections(collections)
    plt.show()