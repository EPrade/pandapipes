# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandas as pd
import sys
import xml.etree.ElementTree as ET
import pandapower
import pandapipes
import numpy as np
from pandapipes import plotting
import matplotlib.pyplot as plt
from pandapower.plotting import create_annotation_collection

def dict_to_data_list(element, parent_no=-1, level=0, dic_list=None, element_no_list=None, group_id_list=None):
    """
    Transforms the dictionary data to a list
    :param element:
    :type element:
    :param parent_no:
    :type parent_no:
    :param level:
    :type level:
    :param dic_list:
    :type dic_list:
    :param element_no_list:
    :type element_no_list:
    :param group_id_list:
    :type group_id_list:
    :return:
    :rtype:
    """
    if dic_list is None:
        dic_list = []

    if element_no_list is None:
        element_no_list = [0]  # Use a list to keep track of element_no across recursive calls

    if group_id_list is None:
        group_id_list = [0]

    element_no = element_no_list[0]
    current_element = {
        'Element_no': element_no,
        'Parent_no': parent_no,
        'Level': level,
        'Group': group_id_list[0],
        'name': element.get('name', f'Element_{element_no}')
    }

    if 'text' in element:
        current_element['text'] = element['text']

    dic_list.append(current_element)

    # Increment the element_no for the next element
    element_no_list[0] += 1

    # Process children if present
    if 'children' in element:
        for child in element['children']:
            dict_to_data_list(child, element_no, level + 1, dic_list, element_no_list, group_id_list)
    # Increment the group ID if this is a root level element
    if parent_no == 0:
        group_id_list[0] += 1
    return dic_list

def convert_dict_to_list(data_dict):
    # Wrapper function to manage element_no using a list
    return dict_to_data_list(data_dict)


def xml_to_data_dic(element):
    """
    Recursively extracts data from the xml-file and writes it into a dictionary
    :param element:
    :type element:
    :return:
    :rtype:
    """
    dic_list = []
    columns = []
    i = 0

    node = {
        'name': element.tag
    }
    if element.text and element.text.strip():
        node['text'] = element.text.strip()

    children = list(element)
    if children:
        node['children'] = [xml_to_data_dic(child) for child in children]

    return node


def extract_values_recursive(data_list, names_of_interest, data_dict):
    """
    Extracts data from the dictionary recursively iterating through the children levels
    :param data_list:
    :type data_list:
    :param names_of_interest:
    :type names_of_interest:
    :param data_dict:
    :type data_dict:
    :return:
    :rtype:
    """
    for entry in data_list:
        name = entry['name']
        if name in names_of_interest:
            data_dict[name].append(entry.get('text', None))

        # Recursively handle children if they exist
        if 'children' in entry:
            extract_values_recursive(entry['children'], names_of_interest, data_dict)

def get_df_from_xml_dic(columns, dictionaries):
    """
    Creates df with respective columns from the given data dictionaries
    :param columns:
    :type columns:
    :param dictionaries:
    :type dictionaries:
    :return:
    :rtype:
    """
    data_dict = {name: [] for name in columns}
    data = dictionaries['children']
    extract_values_recursive(data, columns, data_dict)

    if dictionaries['name'] in ["LineList", "DeviceList"]: #NodeName occurs twice
        data_dict["NodeName1"] = data_dict['NodeName'][0::2]
        data_dict["NodeName2"] = data_dict['NodeName'][1::2]
        del data_dict["NodeName"]
    df= pd.DataFrame(data_dict)
    return df

def get_root_indices(root, element_list):
    list1 = [element.tag for element in root]
    return [index for index, string1 in enumerate(list1) if string1 in element_list.values()]

def extract_data(path, element_list, columns):
    """
    Extracts raw xml-data and returns dictionary with pandas-Dataframes for the given elements with given columns.
    :param path: path to xml-file
    :type path: str
    :param element_list: list of elements in the xml-file to extract
    :type element_list: list of str
    :param columns: parameters of the elements to save into dataframes columns
    :type columns: list of str
    :return: dictionary with dataframes
    :rtype: dict
    """
    tree, root = parse_data(path)
    root_indices = get_root_indices(root, element_list)
    dataframes = {}
    x = 0
    for i in element_list.values():
        print('\n' + 'extracting element :' +str(x+1) + ' / ' +str(len(element_list)))
        dic = xml_to_data_dic(root[root_indices[x]])
        dic_list = convert_dict_to_list(dic)
        element_columns = columns[x]
        dataframes[i] = get_df_from_xml_dic(element_columns, dic)
        x+=1
    return dataframes

def parse_data(path):
    with open(path, encoding="ansi") as f:
        tree = ET.parse(f)
        root = tree.getroot()
    return tree,root




