import pandas as pd

import pandapipes as pps


def calculate_old_buildings(m_cols, timestep, net, t_flow_data, t_return_data, mdot):
    t_flow = t_flow_data.loc[timestep, :]
    t_return = t_return_data.loc[timestep, :]
    mdot_value = mdot.loc[timestep, :]  # get mdot
    heat_value = (t_flow - t_return) * mdot_value
    control_list = []
    for i in range(old_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1

        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('_' + str(m_cols[i]))].index[control_variable]
        if heat_value.iloc[i] == 0:
            net.heat_exchanger.loc[idx, 'qext_w'] = 1
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = 0.00001
        else:
            net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = mdot_value.iloc[i]
    return net

def calculate_new_buildings(OpSimData, m_cols, net):
    t_flow = OpSimData.loc['t_flow', :]
    t_return = OpSimData.loc['t_return', :]
    mdot = OpSimData.loc['mdot', :]
    heat_value = (t_flow - t_return) * mdot
    #timestep = OpSimData.loc['timestep', :]
    control_list = []
    for i in range(new_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1

        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('_' + str(m_cols[i]))].index[control_variable]
        if heat_value.iloc[i] == 0:
            net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = mdot.iloc[i]
        else:
            net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = mdot.iloc[i]
    return net

if __name__ == "__main__":
    #1. prepare net
    #1.1 import net
    net = pps.from_json(r"C:\Users\eprade\Documents\hybridbot\heating grid\net_v01.json")
    #1.2 get indices of heat exchangers and flow controls that need Opsim input or predefined data accordingly to them beeing "Neubaugebiet" or not
    new_building_routes = [1,2,31,4,32]
    #TODO:route 8 taucht nicht auf in den Zeitreihen
    old_building_routes = [5,10,6,7,9]


    # new_routes_he = ['he_' +str(s) for s in new_building_routes]
    # old_routes_he = ['he_' + str(s) for s in old_building_routes]
    # new_he = net.heat_exchanger.query("name == @new_routes_he").index
    # old_he = net.heat_exchanger.query("name == @old_routes_he").index
    #
    # new_routes_fc = ['fc_' + str(s) for s in new_building_routes]
    # old_routes_fc = ['fc_' + str(s) for s in old_building_routes]
    # new_fc = net.flow_control.query("name == @new_routes_fc").index
    # old_fc = net.flow_control.query("name == @old_routes_fc").index


    #1.2.2 import timeseries for old houses
    houses_routes = pd.read_csv(r"C:\Users\eprade\Documents\hybridbot\straßen.csv")
    new_houses = houses_routes[houses_routes['Trasse'].isin(new_building_routes)]
    old_houses = houses_routes[houses_routes['Trasse'].isin(old_building_routes)]

    cols = \
        pd.read_excel(r"C:\Users\eprade\Documents\hybridbot\heating grid\mdot_T_Speichersimulation_HAST_trassen_kurz.xlsx",
                      'mdot_HAST', header=None, nrows=1).values[0]
    mdot = pd.read_excel(
        r"C:\Users\eprade\Documents\hybridbot\heating grid\mdot_T_Speichersimulation_HAST_trassen_kurz.xlsx", 'mdot_HAST',
        header=None, skiprows=1)
    t_flow_data = pd.read_excel(
        r"C:\Users\eprade\Documents\hybridbot\heating grid\mdot_T_Speichersimulation_HAST_trassen_kurz.xlsx", 'TVL_HAST',
        header=None, skiprows=1)
    t_return_data = pd.read_excel(
        r"C:\Users\eprade\Documents\hybridbot\heating grid\mdot_T_Speichersimulation_HAST_trassen_kurz.xlsx", 'TRL_HAST',
        header=None, skiprows=1)


    #2. calculations

    mdot.columns = cols
    t_flow_data.columns = cols
    t_return_data.columns = cols
    mdot_new = mdot.loc[:, new_building_routes]
    mdot = mdot.loc[:, old_building_routes]
    t_flow_data = t_flow_data.loc[:, old_building_routes]
    t_return_data = t_return_data.loc[:, old_building_routes]
    m_cols = mdot.columns
    m_new_cols = mdot_new.columns


    #2.1 Execute every timestep : get timestep, temperature and mass flow values from opsim / itera
    OpSimData = pd.DataFrame(1, index=['timestep', 't_flow', 't_return', 'mdot'], columns=range(25))

    net = calculate_new_buildings(OpSimData, m_new_cols, net)

    #2.2 get temperature and mass flow values of old buildings for each timestep
    timestep = 0  #get from OpSim
     # (tvor-trück)*mdot

    net = calculate_old_buildings(m_cols, timestep, net, t_flow_data, t_return_data, mdot)


    #2.3 run calculation
    # #pps.pipeflow(net, mode='hydraulics')
    pps.pipeflow(net, mode='all', transient=False)

    #2.3 run transient calculation
    # not_connected_j = [12, 14, 16, 19, 20, 21, 23, 56, 83, 110, 193, 239, 280]
    # off_pipes = [0,1,2,3,4,5,6,7,8,9,10]
    # net.junction = net.junction.drop(not_connected_j)
    # net.pipe = net.pipe.drop(off_pipes)
    # dt = 60
    # time_steps = range(100)
    # ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
    # run_timeseries(net, time_steps, transient=True, mode="all", iter=50, dt=dt)
    # res_T = ow.np_results["res_internal.t_k"]