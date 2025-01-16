from toolbox import process_network_data

#setup
#define path to the xml file
xml_path = r"xml.neplst360"

#define path to the heat loads
loads_path = r"loads.xlsx"
#define path to the pipe type library
lib_path_pipes = r"pipes.txt"

#define which elements to be extracted from the xml file; the elements in this example xml-file are called "NodeList",
# "LineList", etc.
elements = {'bus_name':'NodeList', 'line_name':'LineList', 'load_name':'DeviceList'}

#define what parameters you want to extract for the different elements
node_columns = ['ElementType', 'Name', 'AliasName1', 'X', 'Y', 'Elevation', 'LibraryType']
line_columns = ['Name', 'AliasName1', 'LibraryType','Length','NodeName', 'NodeName','CoordinateList']
#define how long the list of geocoordinates per pipe should be
desired_keys_geo = [f'double' for i in range(100)]
line_columns = line_columns + desired_keys_geo
load_columns = ['Name', 'AtLineName', 'AliasName1', 'NodeName']
device_columns = ['ElementType', 'Name', 'AliasName1', 'LibraryType', 'NodeName', 'NodeName']

#specify element type for devices
#heat
element_type_device1 = 'HeatingLoad'
element_type_device2 = 'HeatingValve'

#define the fluid of the net
fluid='water'

#execute the conversion function with the parameters defined above and specify if it is a gas or heating network
net = process_network_data(xml_path, loads_path, lib_path_pipes, elements, node_columns, line_columns,
                           device_columns, fluid, element_type_device1, element_type_device2, gas=False)

# gas
#define path to the xml file
xml_path = r"xml.neplst360"

#define path to the heat loads
loads_path = r"loads.xlsx"
#define path to the pipe type library
lib_path_pipes = r"pipes.txt"

# gas
elements = {'bus_name': 'NodeList', 'line_name': 'LineList', 'device_name': 'DeviceList'}


node_columns = ['ElementType', 'Name', 'AliasName1', 'X', 'Y', 'Elevation', 'LibraryType']
line_columns = ['Name', 'AliasName1', 'LibraryType', 'Length', 'NodeName', 'NodeName', 'CoordinateList']
desired_keys_geo = [f'double' for i in range(200)]
line_columns = line_columns + desired_keys_geo
load_columns = ['Name', 'AtLineName', 'AliasName1', 'NodeName']
device_columns = ['ElementType', 'Name', 'AliasName1', 'LibraryType', 'NodeName', 'NodeName', 'OnElementName',
                  'VariableName', 'Value']

# specify element type for devices

# gas
element_type_device1 = 'GasValve'
fluid = 'hgas'
net_gas = process_network_data(xml_path, loads_path, lib_path_pipes, elements, node_columns, line_columns,
                           device_columns, fluid, element_type_device1, gas=True)