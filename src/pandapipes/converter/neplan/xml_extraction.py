import re
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from time import sleep
import sys
from xml_toolbox import *
#-----------------------------------------------

path = r"C:\Users\eprade\Documents\SW Aalen\NeplanStrom.neplst360"



elements = {'bus_name':'NodeList', 'line_name':'LineList', 'trafo_name':'DeviceList'}


node_columns = ['ElementType', 'Name', 'AliasName1', 'X', 'Y', 'Un']
line_columns = ['Name', 'LibraryType','Length','NodeName', 'NodeName','CoordinateList']
device_columns = ['ElementType', 'Name', 'AliasName1', 'VariableName', 'Value',  'NodeName', 'LayerName', 'X', 'Y']
columns = [node_columns, line_columns, device_columns]




data = extract_data(path, elements, columns)

create_pp_net(data, elements)







