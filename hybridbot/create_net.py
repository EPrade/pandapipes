from os.path import join
import pandas as pd
import numpy as np
import pandapipes as pps

import seaborn as sb
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import geopandas as gp
from pandapower.plotting import create_annotation_collection
from shapely.geometry import LineString

def rf_coord(ref_point1, ref_point2, point, dist, x=True):
    #erzeugt einen punkt um 90° verschoben zu einer ref_Linie
    m = (ref_point2['y'] - ref_point1['y'])/(ref_point2['x']-ref_point1['x'])
    if x==True:
        x1 = point[0]
        y1 = point[1]
        x2 = x1 + dist
        y2 = m*x2 - m*x1 +y1
        point = (x2, y2)
    if x==False:
        x1 = point[0]
        y1 = point[1]
        y2 = y1 + dist
        x2 = (y2-y1)/m + x1
        point = (x2, y2)
    return point

def create_front_flow(net, first_route=1, return_flow=True):

    #distinguish cross junctions and end junctions
    a = net.pipe[['from_junction']].values.tolist()
    a.extend(net.pipe[['to_junction']].values.tolist())
    # get junctions that only appear once
    end_junctions = [x for x in a if a.count(x) == 1]
    end_junctions = [item for sublist in end_junctions for item in sublist]
    cross_junctions = [x for x in a if a.count(x) > 1]
    cross_junctions = np.unique(np.array(cross_junctions)).tolist()
    pump_junction = None
    #get Trasse
    path = r"C:\Users\eprade\Documents\hybridbot\straßen.csv"
    house_data = pd.read_csv(path)
    house_pipes = house_data['Trasse']
    main_junction_n = net.junction_geodata.shape[0]
    main_pipes = range(net.pipe.shape[0])
    if return_flow == True:
        # create cross junctions for return flow with geodata
        b = cross_junctions

        b_s = ['rf_cross_' + str(s) for s in b]
        names = [None] * net.junction_geodata.shape[0]
        for j_name in range(net.junction_geodata.shape[0]):
            if j_name in end_junctions:
                names [j_name] = 'rf_end_' + str(j_name)
            else:
                names [j_name] = 'rf_cross_' + str(j_name)

        rf_main_junctions_geodata = net.junction_geodata.iloc[b, :]
        r_f_coords = np.stack((rf_main_junctions_geodata['x'], rf_main_junctions_geodata['y']), axis=1)
        r_f_coords = tuple(map(tuple, r_f_coords))
        r_f_points = []
        j_list = net.junction.index
        for xy in range(net.junction_geodata.shape[0]):
            dist = 0.0005
            point = net.junction_geodata.iloc[xy]
            try:
                fj = net.pipe[net.pipe['from_junction']==xy]['from_junction'].iloc[0]
                tj = net.pipe[net.pipe['from_junction']==xy]['to_junction'].iloc[0]
            except:
                fj = net.pipe[net.pipe['to_junction'] == xy]['from_junction'].iloc[0]
                tj = net.pipe[net.pipe['to_junction'] == xy]['to_junction'].iloc[0]

            ref_point1 = net.junction_geodata.iloc[fj]
            ref_point2 = net.junction_geodata.iloc[tj]
            new_point = rf_coord(ref_point1, ref_point2, point, dist, x=False)
            r_f_points.append(new_point)
        r_f_points = tuple(map(tuple, r_f_points))
        pps.create_junctions(net, net.junction_geodata.shape[0], pn_bar=1, tfluid_k=283.15, name=names,
                             geodata=r_f_points)
    # iterate over pipe_routes
    for i in house_data['Trasse'].unique():
        number_of_houses_per_route = len(house_data[house_data['Trasse'] == i])
        number_of_pipes_per_route = number_of_houses_per_route
        from_j = net.pipe[net.pipe['name'] == i]['from_junction'].iloc[0]
        to_j = net.pipe[net.pipe['name'] == i]['to_junction'].iloc[0]
        length = net.pipe[net.pipe['name'] == i]['length_km'].iloc[0]
        new_length = length / (number_of_houses_per_route + 2)

        #check if route ends in ending or crossing and create junctions and pipe lists accordingly
        house_list = [None] * number_of_houses_per_route
        junction_index = net.junction.shape[0]
        pipe_variable = 0
        if to_j in end_junctions:
            #end_junctions = True
            pipe_list = [None] * (number_of_houses_per_route*2+1)
            junction_list_fc = [None] * (number_of_houses_per_route)
            junction_list_he = [None] * (number_of_houses_per_route)
            junction_list_rf = [None] * (number_of_houses_per_route)
            # 1 junction from pipe to fc; 1 junction fc to he; 1 junction end of he ;
            # -1 because at the last house pipe to fc is ending junction
            rf_lauf=junction_index + number_of_houses_per_route*2
            for ii in range(number_of_houses_per_route):
                placeholder = True
                junction_list_fc[ii] = junction_index
                junction_list_he[ii] = junction_index+1
                junction_list_rf[ii] = rf_lauf
                rf_lauf+=1
                junction_index+=2
                pipe_list[pipe_variable + 1] = junction_list_fc[ii]
                pipe_list[pipe_variable + 2] = junction_list_fc[ii]
                pipe_variable+=2
            junction_list_fc[-1] = to_j
            pipe_list = pipe_list[0: -1]


        else:
            #end_junctions = False
            pipe_list = [None] * (number_of_houses_per_route*2+2)
            junction_list_fc = [None] * (number_of_houses_per_route)
            junction_list_he = [None] * (number_of_houses_per_route)
            junction_list_rf = [None] * (number_of_houses_per_route)
            rf_lauf = junction_index + number_of_houses_per_route * 2
            # +1 pipe because connection to cross is needed instead of installing last house at end section
            for ii in range(number_of_houses_per_route):
                placeholder = True
                junction_list_fc[ii] = junction_index
                junction_list_he[ii] = junction_index+1
                junction_list_rf[ii] = rf_lauf
                rf_lauf += 1
                junction_index+=2
                pipe_list[pipe_variable+1] = junction_list_fc[ii]
                pipe_list[pipe_variable + 2] = junction_list_fc[ii]
                pipe_variable += 2
        pipe_list[0] = from_j
        pipe_list[-1] = to_j
        new_from = pipe_list[::2]
        new_to = pipe_list[1::2]
        #junction names
        name_list1 = ['pipe/fc_' + str(i), 'fc/he_' + str(i)] * number_of_houses_per_route

        from_coord = net.junction_geodata.loc[from_j, :]
        to_coord = net.junction_geodata.loc[to_j, :]

        if from_coord[0] < to_coord[0]:
            x_dist = (to_coord[0] - from_coord[0]) / number_of_houses_per_route
            x_coords = np.arange(from_coord[0], to_coord[0], x_dist)
        else:
            x_dist = (to_coord[0] - from_coord[0]) / number_of_houses_per_route
            x_coords = np.arange(from_coord[0], to_coord[0], x_dist)
        if from_coord[1] < to_coord[1]:
            y_dist = (to_coord[1] - from_coord[1]) / number_of_houses_per_route
            y_coords = np.arange(from_coord[1], to_coord[1], y_dist)
        else:
            y_dist = (to_coord[1] - from_coord[1]) / number_of_houses_per_route
            y_coords = np.arange(from_coord[1], to_coord[1], y_dist)

        x_coords = np.repeat(x_coords, 2)
        y_coords = np.repeat(y_coords, 2)
        new_coords1 = np.stack((x_coords, y_coords), axis=1)
        pps.create_junctions(net, number_of_houses_per_route * 2, pn_bar=1, tfluid_k=283.15, name=name_list1,
                             geodata=new_coords1)

        #create geodata between main junctions rf
        from_coord = net.junction_geodata.loc[from_j+main_junction_n, :]
        to_coord = net.junction_geodata.loc[to_j+main_junction_n, :]

        if from_coord[0] < to_coord[0]:
            x_dist = (to_coord[0] - from_coord[0]) / number_of_houses_per_route
            x_coords = np.arange(from_coord[0], to_coord[0], x_dist)
        else:
            x_dist = (to_coord[0] - from_coord[0]) / number_of_houses_per_route
            x_coords = np.arange(from_coord[0], to_coord[0], x_dist)
        if from_coord[1] < to_coord[1]:
            y_dist = (to_coord[1] - from_coord[1]) / number_of_houses_per_route
            y_coords = np.arange(from_coord[1], to_coord[1], y_dist)
        else:
            y_dist = (to_coord[1] - from_coord[1]) / number_of_houses_per_route
            y_coords = np.arange(from_coord[1], to_coord[1], y_dist)


        newlst_x = [0] * len(x_coords)
        #newlst_x[::3] = x_coords
        newlst_y = [0] * len(y_coords)
        #newlst_y[::3] = y_coords
        new_coords = np.stack((x_coords, y_coords), axis=1)
        new_coords = tuple(map(tuple, new_coords))
        # linestring = LineString([from_coord,to_coord], axis=1)

        # create components from lists
        name_list = ['he_rf_' + str(i)] * number_of_houses_per_route
        pps.create_junctions(net, number_of_houses_per_route, pn_bar=1, tfluid_k=283.15, name=name_list,
                             geodata=new_coords)

        pps.create_flow_controls(net, junction_list_fc, junction_list_he, 0.05, 0.2, name=('fc_'+str(i)))
        pps.create_heat_exchangers(net, junction_list_he, junction_list_rf, 0.01, 5000, name=('he_'+str(i)))
        pps.create_pipes_from_parameters(net, new_from, new_to,length_km=new_length, diameter_m=0.1, k_mm=0.02, name=('route'+str(i)))

        #create return flow components
        if return_flow == True:
            #abfangen wenn "erste" Trasse
            if i !=first_route:
                to_j_rf = net.junction[net.junction['name'].astype(str).str.match('rf_cross_' + str(from_j))].index[0]

                if to_j in end_junctions: # ändern auf end_junctions

                    from_j_rf_end = net.junction[net.junction['name'].astype(str).str.contains('he_rf_' + str(i))].iloc[
                        -1].name
                    from_junction = from_j_rf_end

                    pipe_list = [None] * (number_of_houses_per_route*2)
                    rf_junctions = net.junction[net.junction['name'].astype(str).str.contains('he_rf_' + str(i))]
                    # 1 junction from pipe to fc; 1 junction fc to he; 1 junction end of he ;
                    # -1 because at the last house pipe to fc is ending junction
                    rf_j_variable = 0
                    for ii in range(0,number_of_houses_per_route*2-2,2):

                        pipe_list[ii+1] = rf_junctions.iloc[-rf_j_variable-2].name
                        pipe_list[ii+2] = rf_junctions.iloc[-rf_j_variable-2].name
                        rf_j_variable+=1



                else:
                    #check if 'to_junction' is end junction

                    from_j_rf_cross = \
                    net.junction[net.junction['name'].astype(str).str.match('rf_cross_' + str(to_j))].iloc[
                        -1].name
                    from_junction = from_j_rf_cross

                    pipe_list = [None] * (number_of_houses_per_route*2 + 2)
                    rf_junctions = net.junction[net.junction['name'].astype(str).str.contains('he_rf_' + str(i))]
                    # +1 pipe because connection to cross is needed instead of installing last house at end section
                    rf_j_variable = 0
                    for ii in range(0,number_of_houses_per_route*2-1,2):

                        pipe_list[ii + 1] = rf_junctions.iloc[-rf_j_variable - 1].name
                        pipe_list[ii + 2] = rf_junctions.iloc[-rf_j_variable - 1].name
                        rf_j_variable += 1
                pipe_list[0] = from_junction
                pipe_list[-1] = to_j_rf
                new_from = pipe_list[::2]
                new_to = pipe_list[1::2]
                pps.create_pipes_from_parameters(net, new_from, new_to, length_km=new_length, diameter_m=0.1, k_mm=0.02,
                                                 name=('route_rf_' + str(i)))
            else:
                pps.create_junction(net, pn_bar=1, tfluid_k=283.15, name='junction_to_pump', geodata=net.junction_geodata.iloc[0])
                to_junction = net.junction[net.junction['name'].astype(str).str.contains('pump')].index[0]
                from_j_rf_end = net.junction[net.junction['name'].astype(str).str.contains('he_rf_' + str(i))].iloc[
                    -1].name
                from_junction = from_j_rf_end

                pipe_list = [None] * (number_of_houses_per_route * 2)
                rf_junctions = net.junction[net.junction['name'].astype(str).str.contains('he_rf_' + str(i))]
                # 1 junction from pipe to fc; 1 junction fc to he; 1 junction end of he ;
                # -1 because at the last house pipe to fc is ending junction
                rf_j_variable = 0
                for ii in range(0, number_of_houses_per_route * 2 - 2, 2):
                    pipe_list[ii + 1] = rf_junctions.iloc[-rf_j_variable - 2].name
                    pipe_list[ii + 2] = rf_junctions.iloc[-rf_j_variable - 2].name
                    rf_j_variable += 1
                pipe_list[0] = from_junction
                pipe_list[-1] = to_junction
                new_from = pipe_list[::2]
                new_to = pipe_list[1::2]
                pps.create_pipes_from_parameters(net, new_from, new_to, length_km=new_length, diameter_m=0.1, k_mm=0.02,
                                                 name=('route_rf_' + str(i)))
                pump_junction = to_junction
        net.pipe.iloc[main_pipes,-2] = False
    return pump_junction

def create_heat_consumers(net, end_junctions):
    path = r"C:\Users\eprade\Documents\hybridbot\straßen.csv"
    house_data = pd.read_csv(path)
    house_pipes = house_data['Trasse']
    #iterate over pipe_routes
    for i in house_data['Trasse'].unique():
        number_of_houses_per_route = len(house_data[house_data['Trasse']==i])
        number_of_pipes_per_route = number_of_houses_per_route
        from_j = net.pipe[net.pipe['name']==i]['from_junction'].iloc[0]
        to_j = net.pipe[net.pipe['name']==i]['to_junction'].iloc[0]
        length = net.pipe[net.pipe['name']==i]['length_km'].iloc[0]
        new_length = length / (number_of_houses_per_route+2)
        old_junction_index = net.junction.shape[0]
        #create new junctions for pipes , flow control, heatexchanger
        name_list = ['pipe/fc_'+str(i), 'fc/he_'+str(i), 'he_rf_'+str(i)] * number_of_houses_per_route
        name_list = name_list[: -1]
        name_list.append('start_rf_' + str(i) + '_j_' + str(to_j))
        #geodate new junctions
        from_coord= net.junction_geodata.loc[from_j,:]
        to_coord = net.junction_geodata.loc[to_j,:]


        if from_coord[0]<to_coord[0]:
            x_dist = (to_coord[0] - from_coord[0]) / number_of_houses_per_route
            x_coords = np.arange(from_coord[0], to_coord[0], x_dist)
        else:
            x_dist = (to_coord[0] - from_coord[0]) / number_of_houses_per_route
            x_coords = np.arange(from_coord[0],to_coord[0], x_dist)
        if from_coord[1]<to_coord[1]:
            y_dist = (to_coord[1] - from_coord[1]) / number_of_houses_per_route
            y_coords = np.arange(from_coord[1], to_coord[1], y_dist)
        else:
            y_dist = (to_coord[1] - from_coord[1]) / number_of_houses_per_route
            y_coords = np.arange(from_coord[1], to_coord[1], y_dist)
        x_coords = np.repeat(x_coords, 3)
        y_coords = np.repeat(y_coords, 3)
        new_coords = np.stack((x_coords, y_coords), axis=1)
        new_coords = tuple(map(tuple, new_coords))
        #linestring = LineString([from_coord,to_coord], axis=1)
        pps.create_junctions(net, number_of_houses_per_route*3, pn_bar=1, tfluid_k=283.15, name=name_list, geodata=new_coords)

        #pps.create_junction(net, pn_bar=1, tfluid_k=283.15, name=('last_pipe_junction '+str(i)))
        #create new pipes
        if number_of_pipes_per_route == 1:
            pipe_junctions = [None] * 2
            pipe_junctions[0] = from_j
            pipe_junctions[1] = to_j
        else:
            pipe_junctions = [None] * (number_of_pipes_per_route*2+2)
            pipe_junctions[0] = from_j
            pipe_junctions[1] = old_junction_index
            pipe_junctions[-1] = to_j
            old_junction_index_copy = old_junction_index


            for ii in range(2,number_of_pipes_per_route*2-1,2):
                junction_index = old_junction_index + ii-2
                pipe_junctions[ii] = junction_index
                junction_index += 3
                pipe_junctions[ii+1] = junction_index
                old_junction_index += 1
            pipe_junctions[-2] = junction_index
        new_from = pipe_junctions[::2]
        new_to = pipe_junctions[1::2]
        inactive_index = net.pipe[net.pipe['name'] == i].index
        net.pipe.iloc[inactive_index, -2] = False
        #net.pipe = net.pipe.drop(net.pipe[net.pipe['name']==i].index)

        pps.create_pipes_from_parameters(net, new_from, new_to,length_km=new_length, diameter_m=0.5, k_mm=0.02, name=('route'+str(i)))
        #create flow controls
        flow_junctions = [None] * (number_of_houses_per_route * 2)
        old_junction_index = old_junction_index_copy
        #flow_junctions[0] = old_junction_index +1
        for x in range(0,number_of_houses_per_route*2,2):
            junction_index = old_junction_index + x
            flow_junctions[x] = junction_index
            junction_index += 1
            flow_junctions[x+1] = junction_index
            old_junction_index += 1
        new_from_fc = flow_junctions[::2]
        new_to_fc = flow_junctions[1::2]
        pps.create_flow_controls(net, new_from_fc, new_to_fc, 0.01, 0.2)
        #create heat exchangers
        heat_junctions = [None] * (number_of_houses_per_route * 2)
        old_junction_index = old_junction_index_copy+1

        for x in range(0, number_of_houses_per_route*2, 2):
            junction_index = old_junction_index + x
            heat_junctions[x] = junction_index
            junction_index += 1
            heat_junctions[x + 1] = junction_index
            old_junction_index += 1
        new_from_he = heat_junctions[::2]
        new_to_he = heat_junctions[1::2]
        new_to_he[-1] = to_j
        heat_exchange = house_data[house_data['Trasse'] == i]['kW']
        pps.create_heat_exchangers(net, new_from_he, new_to_he, 0.01, 1)

    return

def create_return_flow(net, plot_shift):
    #creates return flow pipes and junctions fpr district heating grid
    #get junctions for return flow
    # return_pipe_junctions = net.junction[net.junction['name'].astype(str).str.contains('he_rf_')]
    # start_junctions_return = net.junction[net.junction['name'].astype(str).str.contains('start_rf_')]
    int_check = net.junction.name.apply(type) == int
    int_indices = int_check[int_check].index
    return_junctions = net.junction.iloc[int_indices, :]
    return_junction_geodata = net.junction_geodata.iloc[int_indices, :]

    get_max_x = net.junction_geodata.x.max()
    get_max_y = net.junction_geodata.y.max()
    get_min_x = net.junction_geodata.x.min()
    get_min_y = net.junction_geodata.y.min()
    x_dist = np.abs(get_max_x-get_min_x)
    y_dist = np.abs(get_max_y - get_min_y)
    plot_diff_x = x_dist * plot_shift
    plot_diff_y = y_dist * plot_shift
    x_coords = return_junction_geodata.x + plot_diff_x
    y_coords = return_junction_geodata.y + plot_diff_y
    new_coords = np.stack((x_coords, y_coords), axis=1)
    new_coords = tuple(map(tuple, new_coords))

    return_name = return_junctions.name
    return_name = 'return_' + return_name.astype(str)

    pps.create_junctions(net, return_junctions.shape[0], pn_bar=1, tfluid_k=283.15, name=return_name, geodata=new_coords)

    path = r"C:\Users\eprade\Documents\hybridbot\straßen.csv"
    house_data = pd.read_csv(path)
    house_pipes = house_data['Trasse']
    # iterate over pipe_routes
    for i in house_data['Trasse'].unique():
        number_of_houses_per_route = len(house_data[house_data['Trasse'] == i])
        number_of_pipes_per_route = number_of_houses_per_route
        last_j_str = net.pipe[net.pipe['name'] == i].from_junction.iloc[0]
        check_last_j = net.junction['name'].str.contains('return_' + str(last_j_str+1))
        check_last_j = check_last_j.dropna()
        last_j = check_last_j[check_last_j].index.values[0]
        he_j_check = net.junction['name'].str.contains('he_rf_' + str(i))
        he_j_check = he_j_check.dropna()
        he_j = he_j_check[he_j_check].index.values
        last_he_j = net.pipe[net.pipe['name']==i].to_junction.iloc[0]
        #pipe_junctions = [None] * (number_of_pipes_per_route*2)
        length = net.pipe[net.pipe['name'] == i]['length_km'].iloc[0]
        new_length = length / (number_of_houses_per_route + 2)
        old_junction_index = net.junction.shape[0]
        if number_of_pipes_per_route == 1:
            pipe_junctions = [None] * 2
            pipe_junctions[0] = last_he_j
            pipe_junctions[-1] = last_j
        else:
            pipe_junctions = [None] * (number_of_pipes_per_route*2)
            pipe_junctions[0] = last_he_j
            pipe_junctions[1] = he_j[-1]
            pipe_junctions[-1] = last_j

            pipe_junctions[-2] = he_j[0]
            xi = 1
            for ii in range(0,number_of_pipes_per_route-2,1):

                pipe_junctions[xi+1] = he_j[-ii-1]

                pipe_junctions[xi+2] = he_j[-ii-2]
                xi +=2

        new_from = pipe_junctions[::2]
        new_to = pipe_junctions[1::2]
        pps.create_pipes_from_parameters(net, new_from, new_to, length_km=new_length, diameter_m=0.5, k_mm=0.02, name=('route_rf_'+str(i)))




    #
    # from_j = net.heat_exchanger['to_junction']
    # to_j = net.pipe['from_junction']
    # lengths = net.pipe['length_km']
    # names = net.pipe['name']
    # names = [str(i) for i in names]
    # names = [s + '_return_' for s in names]
    #
    # pps.create_pipes_from_parameters(net, from_j, to_j,length_km=lengths, diameter_m=0.05, k_mm=0.2, name=names)
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
                                     diameter_m=0.1, k_mm=0.02, name=in_pipes['pipe'])

    pump_junction = create_front_flow(net)


    # a = in_pipes[['from_junction']].values.tolist()
    # a.extend(in_pipes[['to_junction']].values.tolist())
    # #get junctions that only appear once
    # end_junctions = [x for x in a if a.count(x) == 1]
    #
    # create_heat_consumers(net, end_junctions)
    # create_return_flow(net, 0.1)

    pps.create_circ_pump_const_pressure(net, pump_junction, 0, 9, 2.6,
                                    t_flow_k=273.15+90)
    from pandapipes import topology as tp

    import networkx as nx
    from matplotlib import pyplot as plt
    ng = tp.create_nxgraph(net)
    pos = nx.planar_layout(ng)
    #pos = nx.spiral_layout(ng)
    #pos = nx.circular_layout(ng)
    #pos = nx.bipartite_layout(ng)
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    nx.draw_networkx(ng, pos=pos, ax=ax)

    plt.show()
    plot.simple_plot(net, junction_size=0.25, heat_exchanger_size=0.05, flow_control_size=0.00005, flow_control_color='blue')
    #pps.pipeflow(net, mode='hydraulics')
    #pps.pipeflow(net, mode='all', transient=False)
    colors = sb.color_palette('colorblind')

    jc = plot.create_junction_collection(net, color=colors[0], size= 0.000015)
    pc = plot.create_pipe_collection(net, color=colors[1])
    coords = net.junction_geodata[['x', 'y']].values
    jic = create_annotation_collection(size=0.00009, texts=np.char.mod('%.0f', net.junction.index),
                                      coords=coords, zorder=0.0001, color='k')

    collections = [jc, pc, jic]

    plot.draw_collections(collections)
    plt.show()
    a= 0