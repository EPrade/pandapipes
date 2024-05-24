# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

elements = {'junction_name':'NodeList', 'pipe_name':'LineList', 'consumer_name':'DeviceList'}

#Here should be distinguished between necessary columns and data the user wants to extract
node_columns = ['ElementType', 'Name', 'AliasName1', 'X', 'Y', 'Elevation']
line_columns = ['Name', 'AliasName1', 'LibraryType','Length','NodeName', 'NodeName','CoordinateList']
device_columns = ['ElementType', 'Name', 'AliasName1', 'LibraryType', 'NodeName', 'NodeName']

columns = [node_columns, line_columns, device_columns]

#specify element type for devices
element_type_device1 = 'HeatingLoad'
element_type_device2 = 'HeatingValve'

fluid = 'water'