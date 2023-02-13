from os.path import join
import pandas as pd
import numpy as np
import pandapipes as pps

import seaborn as sb
import pandapipes.plotting as plot
import matplotlib.pyplot as plt

from pandapower.plotting import create_annotation_collection




def create_heat_consumers(net):
    path = r"C:\Users\eprade\Documents\hybridbot\heckenweg_houses.csv"
    house_data = pd.read_csv(path)
    house_pipes = house_data['Trasse']
    #iterate over pipe_routes
    for i in house_data['Trasse'].unique():
        number_of_houses_per_route = len(house_data[house_data['Trasse']==i])
        number_of_pipes_per_route = number_of_houses_per_route+1
        from_j = net.pipe[net.pipe['name']==i]['from_junction'].iloc[0]
        to_j = net.pipe[net.pipe['name']==i]['to_junction'].iloc[0]
        length = net.pipe[net.pipe['name']==i]['length_km'].iloc[0]
        new_length = length / (number_of_houses_per_route+2)
        old_junction_index = net.junction.shape[0]
        #create new junctions for pipes , flow control, heatexchanger
        pps.create_junctions(net, number_of_houses_per_route, pn_bar=1, tfluid_k=283.15, name=('flow_control_from_route'+str(i)))
        pps.create_junctions(net, number_of_houses_per_route, pn_bar=1, tfluid_k=283.15, name=('flow_control_to_route' + str(i)))
        pps.create_junctions(net, number_of_houses_per_route, pn_bar=1, tfluid_k=283.15, name=('heat_exchanger_from_route' + str(i)))
        pps.create_junctions(net, number_of_houses_per_route, pn_bar=1, tfluid_k=283.15, name=('heat_exchanger_to' + str(i)))
        #create new pipes
        pipe_junctions = [None] * (number_of_pipes_per_route*2)
        pipe_junctions[0] = from_j
        pipe_junctions[number_of_pipes_per_route*2-1] = to_j
        old_junction_index_copy = old_junction_index_copy
        for ii in range(1,number_of_pipes_per_route*2-2,2):
            junction_index = old_junction_index + ii
            pipe_junctions[ii] = junction_index
            junction_index += 3
            pipe_junctions[ii+1] = junction_index
            old_junction_index += 2

        new_from = pipe_junctions[::2]
        new_to = pipe_junctions[1::2]
        net.pipe = net.pipe.drop(net.pipe[net.pipe['name']==i].index)
        pps.create_pipes_from_parameters(net, new_from, new_to,length_km=new_length, diameter_m=0.05, k_mm=0.2, name=('route'+str(i)))
        #create flow controls
        flow_junctions = [None] * (number_of_houses_per_route * 2)
        old_junction_index = old_junction_index_copy
        for x in range(1,number_of_houses_per_route,2):
            junction_index = old_junction_index + x
            flow_junctions[x] = junction_index
            junction_index += 1
            flow_junctions[x+1] = junction_index
            old_junction_index += 1
        new_from_fc = pipe_junctions[::2]
        new_to_fc = pipe_junctions[1::2]
        pps.create_flow_controls(net, new_from_fc, new_to_fc)
        #create heat exchangers
        heat_junctions = [None] * (number_of_houses_per_route * 2)
        old_junction_index = old_junction_index_copy
        for x in range(1, number_of_houses_per_route, 2):
            junction_index = old_junction_index + x
            heat_junctions[x] = junction_index
            junction_index += 1
            heat_junctions[x + 1] = junction_index
            old_junction_index += 1
        new_from_he = pipe_junctions[::2]
        new_to_he = pipe_junctions[1::2]
        pps.create_heat_exchangers(net, new_from_he, new_to_he)

    return


if __name__ == "__main__":
    fluid = "water"

    net = pps.create_empty_network(fluid=fluid)
    in_junctions = pd.read_csv(r"C:\Users\eprade\Documents\hybridbot\junctions.csv")
    in_pipes = pd.read_csv(r"C:\Users\eprade\Documents\hybridbot\pipes.csv")
    geodata = in_junctions[['long', 'lat']].values
    pps.create_junctions(net, nr_junctions=12, name=in_junctions['junction'],pn_bar=1, tfluid_k=283.15, geodata=geodata)
    pps.create_pipes_from_parameters(net, in_pipes['from_junction']-1, in_pipes['to_junction']-1,
                                     length_km=in_pipes['length_m']/1000,
                                     diameter_m=0.05, k_mm=0.2, name=in_pipes['pipe'])

    create_heat_consumers(net)
    colors = sb.color_palette('colorblind')

    jc = plot.create_junction_collection(net, color=colors[0], size= 0.0001)
    pc = plot.create_pipe_collection(net, color=colors[1])
    coords = net.junction_geodata[['x', 'y']].values
    jic = create_annotation_collection(size=0.0003, texts=np.char.mod('%.0f', net.junction.index),
                                      coords=coords, zorder=0.01, color='k')

    collections = [jc, pc, jic]

    plot.draw_collections(collections)
    plt.show()
    a= 0