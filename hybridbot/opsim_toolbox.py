import pandas as pd
import numpy as np
import pandapipes as pps


def calculate_old_buildings(m_cols, timestep, net, t_flow_data, t_return_data, mdot, old_houses):
    t_flow = t_flow_data.loc[timestep, :]
    t_return = t_return_data.loc[timestep, :]
    mdot_value = mdot.loc[timestep, :]  # get mdot
    heat_value = (t_flow - t_return) * mdot_value * 4180
    control_list = []
    for i in range(old_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1

        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('_' + str(m_cols[i]))].index[control_variable]
        if heat_value.iloc[i] == 0:
            net.heat_exchanger.loc[idx, 'qext_w'] = 0
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = 0.00001
        else:
            net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = mdot_value.iloc[i]
    return net


def calculate_new_buildings(OpSimData, callNumber, m_cols, net, new_houses):
    t_flow = OpSimData.loc['TEMPERATURE_T_1_THERMAL_STORAGE', :]
    t_return = OpSimData.loc['TEMPERATURE_T_3_THERMAL_STORAGE', :]
    mdot = OpSimData.loc['MASS_FLOW', :]

    heat_value = (t_flow - t_return) * mdot * 4180
    print("mdot: " + str(mdot))
    print("heat_sum_opsim: " + str(heat_value.sum()))
    # print('heat value :  ' + str(heat_value))
    # timestep = OpSimData.loc['timestep', :]
    control_list = []
    route_list = set()
    input_data = pd.DataFrame()
    all_houses_off = pd.DataFrame(np.nan, index=range(new_houses.shape[0]), columns=['index', 'active'])
    # pps.to_json(net, r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\net_before.json")
    for i in range(new_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1
        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('he_' + str(m_cols[i]))].index[
            control_variable]
        mdot_value2 = mdot[i]
        mdot_value = mdot.iloc[i]
        all_houses_off.loc[i, 'index'] = m_cols[i]
        route_list.add(m_cols[i])
        if mdot_value == 0:
            net.heat_exchanger.loc[idx, 'qext_w'] = 0
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = 0
            all_houses_off.loc[i, 'active'] = False

        else:
            net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = mdot.iloc[i]
            all_houses_off.loc[i, 'active'] = True

        input_data.loc[i, 'heat_value'] = heat_value.iloc[i]
        input_data.loc[i, 'mdot'] = net.flow_control.loc[idx, 'controlled_mdot_kg_per_s']
        input_data.loc[i, 'deltaT'] = t_flow.iloc[i] - t_return.iloc[i]
    # print(all_houses_off)
    route_list = list(route_list)
    # print(route_list)
    # print("net_fc" + str(net.flow_control.iloc[:,3]))
    # =============================================================================
    #     for ii in route_list:
    #         end_fcs = net.flow_control[net.flow_control['name'].str.contains('end_route_' + str(ii))].index
    #         if all(all_houses_off[all_houses_off['index'] == ii]['active']) == False:
    #             net.flow_control.iloc[end_fcs, 3] = 0.15
    #             mdot_v = 0.15
    #             heat = 4180 * mdot_v * 20
    #             net.heat_exchanger.iloc[end_fcs, 4] = heat
    #             print("following_routes_needed_bypass: "+ str(end_fcs))
    #         else:
    #             net.flow_control.iloc[end_fcs, 3] = 0.00001
    #             mdot_v = 0.15
    #             heat = 4180 * mdot_v * 20
    #             net.heat_exchanger.iloc[end_fcs, 4] = 0
    #
    #     print("net_fc" + str(net.flow_control.iloc[:,3]))
    # =============================================================================

    input_data.to_pickle(
        r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_input_" + str(
            callNumber) + '.p')
    # print(input_data)
    # pps.to_json(net, r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\net_after.json")
    return net


def calculate_new_buildings_transient(OpSimData, callNumber, m_cols, net, new_houses):
    t_flow = OpSimData.loc['TEMPERATURE_T_1_THERMAL_STORAGE', :]
    t_return = OpSimData.loc['TEMPERATURE_T_3_THERMAL_STORAGE', :]
    mdot = OpSimData.loc['MASS_FLOW', :]
    heat_value = (t_flow - t_return) * mdot * 4180
    # print('heat value :  ' + str(heat_value))
    # timestep = OpSimData.loc['timestep', :]
    control_list = []

    # input_data = pd.DataFrame()

    for i in range(new_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1
        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('he_' + str(m_cols[i]))].index[
            control_variable]
        mdot_value = mdot.iloc[i]

        if mdot_value < 0.01:
            net.heat_exchanger.loc[idx, 'qext_w'] = 0
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = 0.00001


        else:
            net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = mdot.iloc[i]

        # input_data.loc[i,'heat_value'] = heat_value.iloc[i]
        # input_data.loc[i,'mdot'] = net.flow_control.loc[idx, 'controlled_mdot_kg_per_s']
        # input_data.loc[i,'deltaT'] = t_flow.iloc[i] - t_return.iloc[i]

    heat_supply_sum = net.heat_exchanger['qext_w'].sum()
    # input_data.to_pickle(
    #    r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_input_" + str(callNumber)+'.p')
    # print(input_data)
    # print (heat_supply_sum)
    return net, heat_supply_sum


def calculate_new_buildings_transient_m_controlled(OpSimData, callNumber, m_cols, net, new_houses, t_flow_last_ts,
                                                   t_return_set):
    t_flow = OpSimData.loc['TEMPERATURE_T_1_THERMAL_STORAGE', :]
    t_return = OpSimData.loc['TEMPERATURE_T_3_THERMAL_STORAGE', :]
    mdot = OpSimData.loc['MASS_FLOW', :]
    heat_value = (t_flow - t_return) * mdot * 4180

    m_controlled = (t_flow_last_ts - t_return_set) * 4180 / heat_value
    # print('heat value :  ' + str(heat_value))
    # timestep = OpSimData.loc['timestep', :]
    control_list = []

    # input_data = pd.DataFrame()

    for i in range(new_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1
        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('he_' + str(m_cols[i]))].index[
            control_variable]
        mdot_value = mdot.iloc[i]

        net.heat_exchanger.loc[idx, 'qext_w'] = heat_value.iloc[i]
        net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = m_controlled.iloc[i]

        # input_data.loc[i,'heat_value'] = heat_value.iloc[i]
        # input_data.loc[i,'mdot'] = net.flow_control.loc[idx, 'controlled_mdot_kg_per_s']
        # input_data.loc[i,'deltaT'] = t_flow.iloc[i] - t_return.iloc[i]

    heat_supply_sum = net.heat_exchanger['qext_w'].sum()
    # input_data.to_pickle(
    #    r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_input_" + str(callNumber)+'.p')
    # print(input_data)
    # print (heat_supply_sum)
    return net, heat_supply_sum


def create_results(net):
    he_results = pd.DataFrame()
    he_results['name'] = net.junction.name[net.junction['name'].str.contains('/he') == True].reset_index(drop=True)
    he_results['t_from_k'], he_results['t_to_k'] = net.res_heat_exchanger.iloc[:, 3], net.res_heat_exchanger.iloc[:, 4]
    he_results['v_mean_m_s'], he_results['p_from_bar'], he_results['p_to_bar'] = 0, 0, 0
    he_results['type'] = 'house/he'
    pipe_results = pd.DataFrame()
    pipe_results['name'], pipe_results['t_from_k'], pipe_results['t_to_k'], pipe_results['v_mean_m_s'], \
    pipe_results['p_from_bar'], pipe_results['p_to_bar'], pipe_results['type'] = net.pipe['name'], \
                                                                                 net.res_pipe['t_from_k'], net.res_pipe[
                                                                                     't_to_k'], net.res_pipe[
                                                                                     'v_mean_m_per_s'], net.res_pipe[
                                                                                     'p_from_bar'], \
                                                                                 net.res_pipe['p_to_bar'], 'pipe'
    pipe_results['type'].loc[net.pipe.name[net.pipe['name'].str.contains('rf')].index] = 'return_pipe'
    losses = pd.DataFrame()
    t_h = net.res_junction.loc[net.circ_pump_pressure.flow_junction.iloc[0], 't_k']
    t_l = net.res_junction.loc[net.circ_pump_pressure.return_junction.iloc[0], 't_k']
    pump_energy = (t_h - t_l) * net.res_circ_pump_pressure['mdot_flow_kg_per_s'] * 4200
    house_demands = net.heat_exchanger['qext_w'].sum()
    loss = pump_energy - house_demands
    losses['losses'] = loss
    return pd.concat([he_results, pipe_results, losses])


def create_results_transient(res_junction, net):
    results = pd.DataFrame(res_junction)

    results = results.rename(net.junction['name'].reset_index(drop=True), axis=1)

    return results


def get_losses(net, results, heat_sum):
    return_junction = net.circ_pump_pressure['return_junction'].iloc[0]
    flow_junction = net.circ_pump_pressure['flow_junction'].iloc[0]
    m_flow = net.res_circ_pump_pressure['mdot_flow_kg_per_s'].iloc[0]
    return_T = results.iloc[-1:, return_junction]
    flow_T = results.iloc[-1:, flow_junction]
    print('T-return : ' + str(return_T))
    print('m_flow: ' + str(m_flow))
    print('heat_sum_taken: ' + str(heat_sum))
    return (m_flow * (flow_T - return_T) * 4180) - heat_sum


def heat_loss_pipe_transient(net, res_pipe_to, res_pipe_from, time_steps):
    dT = res_pipe_from[time_steps[-1]] - res_pipe_to[time_steps[-1]]
    mdot = net.res_pipe['mdot_from_kg_per_s']
    Q_loss = dT * mdot * 4200
    Q_loss_sum = Q_loss.sum()
    return Q_loss_sum


def heat_loss_pipe(net):
    idx = net.pipe[net.pipe['name'].astype(str).str.contains('rf')].index
    print("index dropped: " + str(idx))

    net.res_pipe = net.res_pipe.drop(idx)
    print(net.res_pipe)
    dT = net.res_pipe['t_from_k'] - net.res_pipe['t_to_k']
    dT.mask(dT < 0, 0, inplace=True)
    mdot = net.res_pipe['mdot_from_kg_per_s']
    Q_loss = dT * mdot * 4200
    Q_loss_sum = Q_loss.sum()
    return Q_loss_sum


def save_last_timestep(net):
    save_t_vor = pd.DataFrame(net.res_heat_exchanger.t_from_k)
    return save_t_vor.to_csv(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Python Proxy\temp.t_vor.csv")