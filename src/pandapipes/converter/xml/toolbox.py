import pandas as pd
import sys
import xml.etree.ElementTree as ET
import pandapipes
import numpy as np



def get_root_indices(root, element_list):
    """
    Get indices of specified elements in the XML root.

    :param root: Root element of the XML.
    :param element_list: Dictionary of element tags to find.
    :return: List of indices where the elements are located.
    """
    element_tags = [element.tag for element in root]
    return [index for index, tag in enumerate(element_tags) if tag in element_list.values()]

def parse_data(path):
    """
    Parse an XML file and return the tree and root.

    :param path: Path to the XML file.
    :return: Parsed XML tree and root element.
    """
    with open(path, encoding="ansi") as f:
        tree = ET.parse(f)
        root = tree.getroot()
    return tree, root

def extract_data(path, element_list, columns):
    """
    Extract specified elements from the XML and convert them to dataframes.

    :param path: Path to the XML file.
    :param element_list: Dictionary of elements to extract.
    :param columns: List of column names for each element.
    :return: Dictionary of dataframes for each element.
    """
    tree, root = parse_data(path)
    root_indices = get_root_indices(root, element_list)
    dataframes = {}

    for i, (element_key, element_tag) in enumerate(element_list.items()):
        print(f'\nExtracting element: {i + 1}/{len(element_list)}')
        df, dic = xml_to_dataframe(root[root_indices[i]])
        dataframes[element_tag] = get_df_from_dic(columns[i], dic, element_tag)

    return dataframes

def xml_to_dataframe(element):
    """
    Convert an XML element into a Pandas DataFrame and a detailed dictionary list.

    Args:
        element (xml.etree.ElementTree.Element): The XML element to process.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the XML element.
        list: A list of dictionaries with detailed information about nested elements.
    """
    data = []
    dic_list = []
    columns = set()

    def parse_element(element, parent_data=None, depth=0):
        """Recursively parse XML elements and their children."""
        if parent_data is None:
            parent_data = {}

        current_row = parent_data.copy()

        # Process current element's attributes and text
        if element.tag not in current_row:
            current_row[element.tag] = element.text

        # Add the element to the columns set
        columns.add(element.tag)

        # Record the current element data in the dictionary list
        dic = {'Element_no': len(data), 'name': element.tag, 'depth': depth, **element.attrib}
        if element.text is not None:
            dic[element.tag] = element.text.strip()
        dic_list.append(dic)

        # Recursively process child elements
        for child in element:
            parse_element(child, current_row, depth + 1)

        # Only append completed rows (leaves or fully parsed paths)
        if depth == 0 or not list(element):
            data.append(current_row)

    # Start parsing from the root element
    for i, child in enumerate(element):
        parse_element(child)

    # Convert the collected data to a DataFrame
    return pd.DataFrame(data, columns=sorted(columns)), dic_list


def get_df_from_dic(namen, dictionaries, element):
    # Make column names unique by appending an index to duplicates
    seen = {}
    unique_namen = []
    for name in namen:
        if name in seen:
            seen[name] += 1
            unique_namen.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_namen.append(name)

    # Initialize an empty DataFrame with unique column names
    df = pd.DataFrame(columns=unique_namen)
    row_data = {col: None for col in unique_namen}  # Initialize row data
    key_usage = {name: 0 for name in namen}  # Track how many times each key is used

    for i, d in enumerate(dictionaries):
        # Check if a new element begins
        if 'ElementType' in d and d['ElementType'] is not None:
            # Append the completed row to the DataFrame if not the first row
            if any(row_data.values()):  # Check if there's any data to append
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                # Reset row data and key usage for the new element
                row_data = {col: None for col in unique_namen}
                key_usage = {name: 0 for name in namen}

        # Fill the row with values from the dictionary
        for key, value in d.items():
            if key in namen:
                # Use the appropriate occurrence of the key in unique_namen
                column_name = f"{key}_{key_usage[key]}" if key_usage[key] > 0 else key
                if column_name in unique_namen:
                    row_data[column_name] = value
                key_usage[key] += 1

        # Special handling for 'DeviceList' and 'RegulatedPressure'
        if element == 'DeviceList' and d.get('VariableName') == 'RegulatedPressure':
            if 'RegulatedPressure' in unique_namen:
                row_data['RegulatedPressure'] = d.get('RegulatedPressure', None)

        # Progress display
        sys.stdout.write(
            f'\r[{"=" * int(20 * (i + 1) / len(dictionaries)):<20}] {100 * (i + 1) / len(dictionaries):.1f}%')
        sys.stdout.flush()

    # Append the last row if any data remains
    if any(row_data.values()):
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    return df



def create_pps_net(data, element_list, fluid, element_type_device, element_type_device2, heat_loads, lib_path_pipes,
                   consumer_setup):
    """
    Create a pandapipes network based on extracted data.

    :param data: Extracted data from the XML.
    :param element_list: List of elements to process.
    :param fluid: Fluid type for the network.
    :param element_type_device: Element type for heat consumers.
    :param element_type_device2: Element type for valves.
    :param heat_loads: Heat load data.
    :param consumer_setup: Consumer setup type (0 or 1).
    :param lib_path_pipes: Path to the standard pipe library.
    :return: Pandapipes network.
    """
    net = pandapipes.create_empty_network(fluid=fluid)

    #create junctions
    try:
        junction_data, junction_geodata = create_junction_data(data[element_list['bus_name']])
        pandapipes.create_junctions(net, len(junction_data['name']),
                                    pn_bar=6, tfluid_k=293.15,
                                    height_m=junction_data['height_m'],
                                    name=junction_data['name'],
                                    geodata=junction_geodata,
                                    name2=junction_data['name2'],
                                    zone=junction_data['LibraryType'])
        print("\n--Junctions created successfully--\n")
    except Exception as e:
        print("Warning: No junctions created -", e)
    #create pipes
    try:
        pipe_data = create_pipe_data(data[element_list['line_name']])
        pipe_params = read_std_type_pipe(pipe_data['std_type'], lib_path_pipes)

        junction_name_to_index = net.junction.reset_index().set_index('name')['index'].to_dict()

        from_indices = pipe_data['from_junction'].map(junction_name_to_index).dropna().tolist()
        to_indices = pipe_data['to_junction'].map(junction_name_to_index).dropna().tolist()

        pandapipes.create_pipes_from_parameters(net, from_indices, to_indices,
                                                pipe_data['length_km'],
                                                pd.to_numeric(pipe_params['DI'].str.replace(',', '.')) / 1000,
                                                pd.to_numeric(pipe_params['K'].str.replace(',', '.')),
                                                u_w_per_m2k=pd.to_numeric(pipe_params['HeatTransferCoefficient'].str.replace(',', '.')),
                                                name=pipe_data['name'],
                                                geodata=pipe_data['geo'])
        print("\n--Pipes created successfully--\n")
    except Exception as e:
        print("Warning: No pipes created -", e)
    #create consumers
    try:
        loads = create_heat_load_data(data[element_list['load_name']], element_type_device, heat_loads)

        # Adjust attributes based on consumer setup
        if consumer_setup == 0:
            loads['t_return'] = None
        else:
            loads['mdot'] = None

        # Map junction names and zones to their indices
        junction_info = {
            name: (index, zone)
            for index, (name, zone) in enumerate(zip(net.junction['name'], net.junction['zone']))
        }

        # Determine from and to indices for heat consumers
        from_indices = []
        to_indices = []

        for node0, node1 in zip(loads['node0'], loads['node1']):
            if node0 in junction_info:
                index0, zone0 = junction_info[node0]
                (from_indices if zone0 == 'VL_Knoten-FW' else to_indices).append(index0)

            if node1 in junction_info:
                index1, zone1 = junction_info[node1]
                (from_indices if zone1 == 'VL_Knoten-FW' else to_indices).append(index1)

        # Retrieve x and y coordinates for geodata
        x = net.junction_geodata.iloc[from_indices]['x']
        y = net.junction_geodata.iloc[from_indices]['y']

        # Create heat consumers
        pandapipes.create_heat_consumers(
            net, from_indices, to_indices,
            qext_w=loads['q'], controlled_mdot_kg_per_s=loads['mdot'],
            treturn_k=loads['t_return'], name=loads['name'], X=x, Y=y
        )
        print("\n-- Heat consumers created successfully --\n")
    except Exception as e:
        print("Warning: Heat consumers not created -", e)

    # Create valves
    try:
        valves = create_valve_data(data[element_list['load_name']], element_type_device2, lib_path_pipes)

        # Map junction names to indices
        junction_index_dict = {name: index for index, name in enumerate(net.junction['name'])}

        # Map valve nodes to junction indices
        from_indices = [
            junction_index_dict[node] for node in valves['from_node'] if node in junction_index_dict
        ]
        to_indices = [
            junction_index_dict[node] for node in valves['to_node'] if node in junction_index_dict
        ]

        # Create valves
        pandapipes.create_valves(net, from_indices, to_indices, valves['diameter'])
        print("\n-- Valves created successfully --\n")
    except Exception as e:
        print("Warning: Valves not created -", e)

    return net

def create_pps_net_gas(data, element_list, fluid, element_type_device, lib_path):
    """
    Creates a gas network using pandapipes based on the provided data and elements.

    Parameters:
    - data: Dictionary containing DataFrames for network elements.
    - element_list: Dictionary mapping element types to their DataFrame keys in `data`.
    - fluid: The type of fluid used in the network, e.g., 'gas'.
    - element_type_device: The type of device elements to consider, e.g., 'GasValve'.

    Returns:
    - A pandapipes network object.
    """
    net = pandapipes.create_empty_network(fluid=fluid)

    # Create junctions
    try:
        junction_dic, junction_geodata = create_junction_data(data[element_list['bus_name']])
        pandapipes.create_junctions(
            net,
            len(junction_dic['name']),
            pn_bar=6,
            tfluid_k=293.15,
            height_m=junction_dic['height_m'],
            name=junction_dic['name'],
            geodata=junction_geodata,
            name2=junction_dic['name2']
        )
        print("\n --junctions created successfully-- \n")
    except Exception as e:
        print("Warning: No junctions created -", e)

    # Create pipes
    try:
        pipe_dic = create_pipe_data(data[element_list['line_name']])
        pipe_parameters = read_std_type_pipe(pipe_dic['std_type'], lib_path, xlsx=True)

        # Mapping junction names to indices
        name_to_index = {name: idx for idx, name in enumerate(net.junction['name'])}
        from_indices = pipe_dic['from_junction'].map(name_to_index).dropna().astype(int).tolist()
        to_indices = pipe_dic['to_junction'].map(name_to_index).dropna().astype(int).tolist()

        # Convert pipe parameters
        k_mm = pd.to_numeric(pipe_parameters['K'].str.replace(',', '.'))
        diameter_m = pd.to_numeric(pipe_parameters['DI'], errors='coerce') / 1000

        pandapipes.create_pipes_from_parameters(
            net, from_indices, to_indices, pipe_dic['length_km'], diameter_m, k_mm,
            loss_coefficient=0, sections=1, name=pipe_dic['name'], geodata=pipe_dic['geo'],
             PN=pipe_parameters['PN'], stdtype=pipe_parameters['LibraryType']
        )
        print("\n --pipes created successfully -- \n")
    except Exception as e:
        print("Warning: No pipes created -", e)

    # Create loads
    try:
        loads = data[element_list['line_name']]
        loads_ha = loads[loads['AliasName1'].str.contains('Hausanschluss')]

        # Mapping junction names to indices
        junction_index_dict = {name: idx for idx, name in enumerate(net.junction['name'])}
        indices1 = [junction_index_dict[node] for node in loads_ha['NodeName'] if node in junction_index_dict]
        indices2 = [junction_index_dict[node] for node in loads_ha['NodeName_1'] if node in junction_index_dict]

        house_junctions = []
        for idx1, idx2 in zip(indices1, indices2):
            name1 = net.junction.loc[idx1, 'name2']
            name2 = net.junction.loc[idx2, 'name2']

            if not name1.startswith('HA Abgang'):
                house_junctions.append(net.junction.loc[idx1])
                house_junctions[-1]['index'] = idx1
            elif not name2.startswith('HA Abgang'):
                house_junctions.append(net.junction.loc[idx2])
                house_junctions[-1]['index'] = idx2

        house_junctions_df = pd.DataFrame(house_junctions)
        pandapipes.create_sinks(
            net, house_junctions_df['index'], np.nan, name=house_junctions_df['name'],
            name2=house_junctions_df['name2']
        )

        print("\n --Loads created successfully -- \n")
    except Exception as e:
        print("Warning: Loads not created -", e)

    # Create valves
    try:
        valves = data[element_list['device_name']]
        valves = valves[valves['ElementType'] == element_type_device]

        # Mapping junction names to indices
        junction_index_dict = {name: idx for idx, name in enumerate(net.junction['name'])}
        from_indices = [junction_index_dict[node] for node in valves['NodeName'] if node in junction_index_dict]
        to_indices = [junction_index_dict[node] for node in valves['NodeName_1'] if node in junction_index_dict]

        pandapipes.create_pressure_controls(
            net, from_indices, to_indices, to_indices,
            valves['Value'], name=valves['Name'], name2=valves['AliasName1']
        )

        print("\n --valves created successfully -- \n")
    except Exception as e:
        print("Warning: valves not created -", e)

    return net

def create_junction_data(data):
    """
    Extract junction data from the input.

    :param data: Input data for junctions.
    :return: Dictionary of junction properties and geodata.
    """
    bus_data = {
        'height_m': data['Elevation'],
        'name': data['Name'],
        'name2': data['AliasName1'],
        'LibraryType': data['LibraryType']
    }
    geodata = list(zip(data['X'], data['Y']))
    geodata_float = [(float(x), float(y)) for x, y in geodata]
    return bus_data, geodata_float

def create_pipe_data(data):
    """
    Extract pipe data from the input.

    :param data: Input DataFrame for pipes.
    :return: Dictionary of pipe properties.
    """

    max_index = len(data.columns) - 1

    # Extract pipe data
    pipe_data = {
        'from_junction': data['NodeName'],
        'to_junction': data['NodeName_1'],
        'length_km': data['Length'].astype(float) / 1000,
        'std_type': data['LibraryType'],
        'name': data['Name'],
        'geo': [
            [(row.iloc[col], row.iloc[col + 1]) for col in range(0, max_index, 2) if col + 1 <= max_index and not pd.isna(row.iloc[col])]
            for _, row in data.iterrows()
        ]
    }
    return pipe_data

def read_std_type_pipe(std_type, lib_path, xlsx=False):
    """
    Read standard pipe types from a library.

    :param std_type: List of pipe types.
    :param lib_path: Path to the library file.
    :return: DataFrame of pipe parameters.
    """
    if not xlsx:
        library = pd.read_csv(lib_path, sep='\t', header=2)
    else:
        library = pd.read_excel(lib_path, header=2)
    return pd.merge(pd.DataFrame({'LibraryType': std_type}), library, on='LibraryType', how='left')

def create_heat_load_data(data, element_type_device, heat_loads):
    """
    Creates a DataFrame of heat load parameters for a specified device type.

    Parameters:
    - data: DataFrame containing element data.
    - element_type_device: The type of device for which heat loads are created.
    - heat_loads: DataFrame containing heat load details.
    - consumer_modelling: Boolean indicating whether consumer modeling is used.

    Returns:
    - A DataFrame with merged heat load data and parameters.
    """
    # Filter data for the specified element type
    heat_load = data.loc[data['ElementType'] == element_type_device]

    # Ensure required columns are present in heat_loads
    required_cols = ['name', 't_return', 'q', 'diameter', 't_flow', 'mdot']
    for col in required_cols:
        if col not in heat_loads.columns:
            heat_loads[col] = 0.1 if col == 'diameter' else pd.NA

    # Reorder columns
    all_columns = required_cols + [col for col in heat_loads.columns if col not in required_cols]
    heat_loads_selected = heat_loads.reindex(columns=all_columns)

    # Prepare heat load parameters
    heat_load_parameters = pd.DataFrame({
        'name': heat_load['Name'].reset_index(drop=True),
        'node0': heat_load['NodeName'].reset_index(drop=True),
        'node1': heat_load['NodeName_1'].reset_index(drop=True)
    })

    # Merge heat load parameters with the selected columns
    merged_df = pd.merge(heat_load_parameters, heat_loads_selected, on='name', how='left')
    return merged_df

def create_valve_data(data, element_type, lib_path):
    """
    Creates a dictionary of valve parameters for a specified element type.

    Parameters:
    - data: DataFrame containing element data.
    - element_type: The type of element for which valves are created.

    Returns:
    - A dictionary containing valve parameters.
    """
    # Filter data for the specified element type
    data_valve = data.loc[data['ElementType'] == element_type]

    # Read and process standard type parameters
    valve_std_type_param = read_std_type_pipe(data_valve['LibraryType'], lib_path)
    valve_std_type_param['d'] = valve_std_type_param['LibraryType'].str.extract('(\d+)').astype(int) / 1000

    # Prepare valve parameters
    valve_parameters = {
        'std_type': data_valve['LibraryType'],
        'from_node': data_valve['NodeName'],
        'to_node': data_valve['NodeName_1'],
        'opened': True,
        'diameter': valve_std_type_param['d']
    }

    return valve_parameters

def process_network_data(xml_path, loads_path, lib_path_pipes, elements, node_columns, line_columns, device_columns,
                         fluid, element_type_device1, element_type_device2=None, consumer_setup=1, gas=False):
    """
    Processes network data from various sources and creates a network model.

    Parameters:
    - xml_path: Path to the XML file containing network element data.
    - loads_path: Path to the Excel file containing load data.
    - lib_path_pipes: Path to the Excel file containing library data for pipes.
    - elements: Dictionary specifying element names for extraction.
    - node_columns: List of column names for node elements.
    - line_columns: List of column names for line elements.
    - device_columns: List of column names for device elements.
    - fluid: Type of fluid used in the network (e.g., 'water').
    - element_type_device1: Element type for the first set of devices (e.g., 'HeatingLoad').
    - element_type_device2: Element type for the second set of devices (e.g., 'HeatingValve').
    - consumer_setup: An optional parameter for consumer setup, default is 1.

    Returns:
    - A network model object created using the provided data.
    """

    # Define column groups for extraction
    columns = [node_columns, line_columns, device_columns]

    # Extract data from XML
    data = extract_data(xml_path, elements, columns)
    if not gas:
        # Load heat loads from Excel
        heat_loads = pd.read_excel(loads_path)

        # Create the network model
        net = create_pps_net(
            data,
            elements,
            fluid,
            element_type_device1,
            element_type_device2,
            heat_loads,
            lib_path_pipes,
            consumer_setup=consumer_setup
        )

    else:
        net = create_pps_net_gas(data, elements, fluid, element_type_device1, lib_path_pipes)
    return net