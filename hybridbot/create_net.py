import math
from os.path import join
import pandas as pd
import numpy as np
import pandapipes as pps
import tempfile
import seaborn as sb
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import geopandas as gp
from pandapower.plotting import create_annotation_collection
from shapely.geometry import LineString
from pandapipes.test.pipeflow_internals.test_transient import _output_writer
from pandapipes.timeseries import run_timeseries
import pandapower.control as control
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.plotting.plotting_toolbox import get_color_list, _rotate_dim2, get_angle_list, \
    get_list
import re

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    def norm(self):
        return self.dot(self)**0.5
    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)
    def perp(self):
        return Vector(1, -self.x / self.y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    def __str__(self):
        return f'({self.x}, {self.y})'

    def perptoy(self):
        return Vector(1, 0)

    def perptox(self):
        return Vector(1, -10000)

def rf_coord(ref_point1, ref_point2, point, dist, x=False):
    A = ref_point1
    B = ref_point2
    #erzeugt einen punkt um 90° verschoben zu einer ref_Linie
    m = (B['y'] - A['y'])/(B['x']-A['x'])
    #dist = 10
    # A = Vector(ref_point1[0], ref_point1[1])
    # B = Vector(ref_point2[0], ref_point2[1])
    # AB = B - A
    # if (A.y == B.y):
    #     AB_perp_normed = AB.perptox().normalized()
    # elif (A.x == B.x):
    #     AB_perp_normed = AB.perptoy().normalized()
    # else:
    #     AB_perp_normed = AB.perp().normalized()
    # dist = 2000
    # P1 = B + AB_perp_normed * dist
    # P2 = B - AB_perp_normed * dist
    #
    # P1_int = [int(s) for s in re.findall(r'\b\d+\b', f'Point{P1}')]
    # P1_1 = P1_int[0] , P1_int[1]
    # tempStr = (str(P1_1[0])+'.'+str(P1_1[1]))
    # P1_float = float(tempStr)
    # P1_2 = P1_int[2], P1_int[3]
    # tempStr = (str(P1_2[0]) + '.' + str(P1_2[1]))
    # P12_float = float(tempStr)
    # point1 = P1_float , P12_float
    #
    # P2_int = [int(s) for s in re.findall(r'\b\d+\b', f'Point{P2}')]
    # P2_1 = P2_int[0], P2_int[1]
    # tempStr = (str(P2_1[0]) + '.' + str(P2_1[1]))
    # P2_float = float(tempStr)
    # P2_2 = P2_int[2], P2_int[3]
    # tempStr = (str(P2_2[0]) + '.' + str(P2_2[1]))
    # P22_float = float(tempStr)
    # point2 = P2_float, P22_float


    # dy = math.sqrt(dist**2/(m**2+1))
    # dx = -m*dy
    # A_new = [None] * 2
    # B_new = [None] * 2
    # A_new[0],A_new[1] = A[0] + dx, A[1] + dy
    # B_new[0], B_new[1] = B[0] + dx, B[1] + dy
    cd_length = dist*100000

    ab = LineString([A, B])
    left2 = ab.offset_curve(cd_length)

    left2 = list(left2.coords)



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

def create_front_and_return_flow(net, first_route=1, return_flow=True):
    """
    Creates pipes, flow controls, heat exchangers, respective junctions and the return flow pipes for the given main
    routes and their respective number of heat sinks
    :param net: pandapipes net
    :type net: pandapipes net
    :param first_route: first route from/to circulation pump
    :type first_route:  int
    :param return_flow:
    :type return_flow:
    :return:
    :rtype:
    """

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
    heat_values = pd.DataFrame(data=None)
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
        fj = 10000
        for xy in range(net.junction_geodata.shape[0]):
            dist = 0
            #dist = 0
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
        heat_kw = house_data[house_data['Trasse']==i]['kW']
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
            rf_lauf=junction_index + number_of_houses_per_route*2
            #iterate over number of houses and create junctions for pipes, flow controls and heat exchangers
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

        #get coordinate of new junctions created
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

        new_coords = np.stack((x_coords, y_coords), axis=1)
        new_coords = tuple(map(tuple, new_coords))

        # create components from lists
        name_list = ['he_rf_' + str(i)] * number_of_houses_per_route
        pps.create_junctions(net, number_of_houses_per_route, pn_bar=1, tfluid_k=283.15, name=name_list,
                             geodata=new_coords)

        pps.create_flow_controls(net, junction_list_fc, junction_list_he, 0.2, 0.2, name=('fc_'+str(i)))
        pps.create_heat_exchangers(net, junction_list_he, junction_list_rf, 0.01, heat_kw*1000, name=('he_'+str(i)))
        pps.create_pipes_from_parameters(net, new_from, new_to,length_km=new_length, diameter_m=0.1, k_mm=0.02, name=('route'+str(i)),alpha_w_per_m2k=2)

        #create return flow components
        if return_flow == True:
            #check if first/starting route
            if i !=first_route:
                to_j_rf = net.junction[net.junction['name'].astype(str).str.match('rf_cross_' + str(from_j))].index[0]
                if to_j in end_junctions:
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
                                                 name=('route_rf_' + str(i)),alpha_w_per_m2k=5)
            else:
                #if not first route
                pps.create_junction(net, pn_bar=1, tfluid_k=283.15, name='junction_to_pump', geodata=net.junction_geodata.iloc[0])
                to_junction = net.junction[net.junction['name'].astype(str).str.contains('pump')].index[0]
                from_j_rf_end = net.junction[net.junction['name'].astype(str).str.endswith('rf_cross_' + str(to_j))].iloc[-1].name
                from_junction = from_j_rf_end
                pipe_list = [None] * (number_of_houses_per_route * 2+2)
                rf_junctions = net.junction[net.junction['name'].astype(str).str.contains('he_rf_' + str(i))]
                rf_j_variable = 0
                for ii in range(0, number_of_houses_per_route *2-1, 2):
                    pipe_list[ii + 1] = rf_junctions.iloc[-rf_j_variable - 1].name
                    pipe_list[ii + 2] = rf_junctions.iloc[-rf_j_variable - 1].name
                    rf_j_variable += 1
                pipe_list[0] = from_junction
                pipe_list[-1] = to_junction
                new_from = pipe_list[::2]
                new_to = pipe_list[1::2]
                pps.create_pipes_from_parameters(net, new_from, new_to, length_km=new_length, diameter_m=0.1, k_mm=0.02,
                                                 name=('route_rf_' + str(i)), alpha_w_per_m2k=2)
                pump_junction = to_junction
        net.pipe.iloc[main_pipes,-2] = False
    return pump_junction


if __name__ == "__main__":

    fluid = "water"

    net = pps.create_empty_network(fluid=fluid)

    #read data
    in_junctions = pd.read_csv(r"C:\Users\eprade\Documents\hybridbot\junctions.csv")
    in_pipes = pd.read_csv(r"C:\Users\eprade\Documents\hybridbot\pipes.csv")

    #prepare net
    geodata = in_junctions[['long', 'lat']].values

    pps.create_junctions(net, nr_junctions=12, name=in_junctions['junction'],pn_bar=1, tfluid_k=283.15, geodata=geodata)
    pps.create_pipes_from_parameters(net, in_pipes['from_junction']-1, in_pipes['to_junction']-1,
                                     length_km=in_pipes['length_m']/1000,
                                     diameter_m=0.1, k_mm=0.02, name=in_pipes['pipe'])

    pump_junction = create_front_and_return_flow(net)

    pps.create_circ_pump_const_pressure(net, pump_junction, 0, 9, 2.6,
                                      t_flow_k=273.15+70)
    net.junction_geodata[["x", "y"]] = net.junction_geodata[["y", "x"]]

    # profiles_heat = pd.read_csv()
    # ds_heat = DFData(profiles_heat)
    #
    # const_flow_m = control.ConstControl(net, element='flow_control', variable='mdot_kg_per_s',
    #                                     element_index=net.source.index.values,
    #                                     data_source=ds_heat,
    #                                     profile_name=net.source.index.values.astype(str))
    # const_heat_to = control.ConstControl(net, element='heat_exchanger', variable='t_k_to',
    #                                     element_index=net.source.index.values,
    #                                     data_source=ds_heat,
    #                                     profile_name=net.source.index.values.astype(str))
    # const_heat_from = control.ConstControl(net, element='heat_exchanger', variable='t_k_from',
    #                                      element_index=net.source.index.values,
    #                                      data_source=ds_heat,
    #                                      profile_name=net.source.index.values.astype(str))

    #calculations
    #get temperature and mass flow values
    t_vor = 330
    t_r = 310
    m_dot = 0.1
    heat_values = m_dot*(t_vor-t_r)
    #pps.pipeflow(net, mode='hydraulics')

    pps.pipeflow(net, mode='all', transient=False)
    # not_connected_j = [12, 14, 16, 19, 20, 21, 23, 56, 83, 110, 193, 239, 280]
    # off_pipes = [0,1,2,3,4,5,6,7,8,9,10]
    # net.junction = net.junction.drop(not_connected_j)
    # net.pipe = net.pipe.drop(off_pipes)
    # dt = 60
    # time_steps = range(100)
    # ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
    # run_timeseries(net, time_steps, transient=True, mode="all", iter=50, dt=dt)
    # res_T = ow.np_results["res_internal.t_k"]


    #plotting
    from pandapipes import topology as tp

    import networkx as nx
    from matplotlib import pyplot as plt

    ng = tp.create_nxgraph(net)
    pos = nx.planar_layout(ng)
    # pos = nx.spiral_layout(ng)
    # pos = nx.circular_layout(ng)
    # pos = nx.bipartite_layout(ng)
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    nx.draw_networkx(ng, pos=pos, ax=ax)

    plt.show()

    plot.simple_plot(net, junction_size=0.25, heat_exchanger_size=0.05, flow_control_size=0.00005, flow_control_color='blue')


    colors = sb.color_palette('colorblind')

    jc = plot.create_junction_collection(net, color=colors[0], size= 0.000015)
    pc = plot.create_pipe_collection(net, color=colors[1])
    coords = net.junction_geodata[['x', 'y']].values

    jic = create_annotation_collection(size=0.00009, texts=np.char.mod('%.0f', net.junction.index),
                                      coords=coords, zorder=0.0001, color='k')

    collections = [jc, pc, jic]

    plot.draw_collections(collections, axes_visible=(False, True))
    plt.show()
    a= 0
    #plot heat results
    from pandapower.plotting import cmap_continuous

    cmap_list_v = [(0.0, "green"), (1.25, "yellow"), (2.5, "red")]
    cmap_v, norm_v = cmap_continuous(cmap_list_v)

    vc = plot.create_pipe_collection(net, linewidths=1, cmap=cmap_v, norm=norm_v,
                                     z=net.res_pipe.v_mean_m_per_s.abs(),
                                     cbar_title="mean water velocity [m/s]")

    collections += [vc]

    plot.draw_collections(collections)
    plt.show()

    max_p = net.res_junction.p_bar.max()
    cmap_list_p = [(4, 'red'), (max_p / 2, 'yellow'), (max_p, 'green')]
    cmap_p, norm_p = cmap_continuous(cmap_list_p)
    jc = plot.create_junction_collection(net, size=0.000025, cmap=cmap_p, norm=norm_p,
                                         z=net.res_junction.p_bar, cbar_title="junction pressure [bar]")
    collections += [jc]
    plot.draw_collections(collections)
    plt.show()

    cmap_list_t = [(300, 'blue'), (320, 'yellow'), (340, 'green')]
    cmap_t, norm_t = cmap_continuous(cmap_list_t)
    tc = plot.create_junction_collection(net, size=0.000025, cmap=cmap_t, norm=norm_t,
                                         z=net.res_junction.t_k, cbar_title="junction temperature [t_k]", alpha=0.2)
    collections = collections[:-1]
    collections += [tc]
    plot.draw_collections(collections)
    plt.show()

    rf_index = net.junction[net.junction['name'].astype(str).str.contains('rf')].index
    net.junction_geodata = net.junction_geodata.drop(rf_index)