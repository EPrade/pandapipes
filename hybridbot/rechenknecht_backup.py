'''
Created on Mo Nov 14 17:08:54 2022
Code template to create pandapipes proxy for HybridBot
@author: fmarten
'''
from opsim.BaseProxy import BaseProxy
from opsim.Client import Client
from datetime import datetime
from pytz import timezone, utc
import pandas as pd
import logging
import pandapipes as pps
from pipe_toolbox import calculate_new_buildings, get_losses, calculate_old_buildings, \
    calculate_new_buildings_transient, create_results, create_results_transient
from pandapipes.test.pipeflow_internals.test_transient import _output_writer
from pandapipes.timeseries import run_timeseries
import tempfile


class OpSim_Proxy(BaseProxy):

    def __init__(self, client, name, logger):
        super().__init__(client, logger, name)
        self.local_tz = timezone('Europe/Amsterdam')

    # Method to initialize your simulation variables (dummy content)
    def initSimulation(self):

        only_new_buildings_net = True
        self.transient = True
        # Create result table
        self.path_results = r'C:\Users\HybridBOT\Documents' \
                            + r'\HybridbotFW_Git_Cosimulation\Results\pipes'
        self.result_table = pd.DataFrame(index=range(112),
                                         columns=['junction_name', 'pipe_name', 'v_mean',
                                                  'p_bar', 't_k'])
        # 1. prepare net
        # 1.1 import net
        if only_new_buildings_net == True:
            self.net = pps.from_json(
                r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\net_16_05_ng.json")
        if only_new_buildings_net == False:
            self.net = pps.from_json(
                r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\net_v11_04.json")

        # 1.2 get indices of heat exchangers and flow controls that need Opsim input or predefined data accordingly to them beeing "Neubaugebiet" or not
        new_building_routes = [1, 2, 31, 4, 32]
        # TODO:route 8 taucht nicht auf in den Zeitreihen
        old_building_routes = [5, 10]
        # nur neubaugebiet

        # 1.2.2 import timeseries for old houses
        houses_routes = pd.read_csv(
            r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\straßen_nur_ng.csv")
        self.new_houses = houses_routes
        self.old_houses = houses_routes[houses_routes['Trasse'].isin(old_building_routes)]

        cols = \
            pd.read_excel(
                r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\mdot_T_Speichersimulation_HAST_trassen_kurz_nur_neu.xlsx",
                'mdot_HAST', header=None, nrows=1).values[0]
        mdot = pd.read_excel(
            r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\mdot_T_Speichersimulation_HAST_trassen_kurz_nur_neu.xlsx",
            'mdot_HAST',
            header=None, skiprows=1)
        t_flow_data = pd.read_excel(
            r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\mdot_T_Speichersimulation_HAST_trassen_kurz_nur_neu.xlsx",
            'TVL_HAST',
            header=None, skiprows=1)
        t_return_data = pd.read_excel(
            r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\mdot_T_Speichersimulation_HAST_trassen_kurz_nur_neu.xlsx",
            'TRL_HAST',
            header=None, skiprows=1)

        # 2. calculations
        mdot.columns = cols
        t_flow_data.columns = cols
        t_return_data.columns = cols
        mdot_new = mdot.iloc[:, 1::]
        self.mdot = mdot.loc[:, old_building_routes]
        self.t_flow_data = t_flow_data.loc[:, old_building_routes]
        self.t_return_data = t_return_data.loc[:, old_building_routes]
        self.m_cols = mdot.columns
        self.m_new_cols = mdot_new.columns

        # @Erik, hier kannst du alle deine Zeitreihen laden in self.tables
        self.createInputTable()
        self.logger.info('%s is initialized!', self.componentName)

        # timeseries setup
        self.dt = 60
        self.time_steps = range(15)

    def createInputTable(self):
        self.inputTable = pd.DataFrame(columns=[
            'TEMPERATURE_T_1_THERMAL_STORAGE',
            'TEMPERATURE_T_2_THERMAL_STORAGE', 'MASS_FLOW'])
        for asset in self.readableAssets:
            self.inputTable.loc[asset.gridAssetId] = 0

        self.inputTable = self.inputTable.sort_index()
        self.inputTable = self.inputTable.T

    # Write incoming OpSim data pakets into inputTable
    def readMeasurements(self, inputFromClient):
        osm = inputFromClient
        for thismsg in osm:
            messagetype = str(thismsg.__class__.__name__)
            if messagetype == 'OpSimAggregatedMeasurements':
                assetId = thismsg.assetId.replace('/Measurement', '')
                for meas in thismsg.opSimMeasurements:
                    self.inputTable.at[meas.measurementType,
                                       assetId] = meas.measurementValue
        return self.inputTable

    def processMeasurement(self, meas, assetId):
        load_index = self.map_loadnames.at[assetId, 'load_index']
        try:
            self.net.load.loc[
                load_index,
                ['p_mw', 'q_mvar']] = [meas.measurementValue / 1e6, 0]
        except:
            self.logger.error('Error when processing setpoint. Data: %s', meas)

    #  Run when my simulation is called for the first time
    def stepZeroSimulation(self):
        pass

    # Run when my simulation is called for the second, third,.... time
    def stepSimulation(self, inputFromClient, timeToAdvanceTo):
        dt = datetime.utcfromtimestamp(self.simulationTime / 1000)
        dt = utc.localize(dt).astimezone(self.local_tz)
        self.logger.info(
            '%s step call at simulation time = %s, in present timezone = %s',
            self.componentName, self.simulationTime, dt)

        inputTable = self.readMeasurements(inputFromClient)
        # print("Opsim Input: " + str(inputTable))

        self.timestep = self.callNumber

        # 2.1 Execute every timestep : get timestep, temperature and mass flow values from opsim / itera
        # OpSimData = pd.DataFrame(1, index=['timestep', 't_flow', 't_return', 'mdot'], columns=range(25))
        OpSimData = inputTable
        # print("Calculation Timestep:" + str(self.timestep))

        # net3 = calculate_old_buildings(self.m_cols, self.timestep, net2, self.t_flow_data, self.t_return_data, self.mdot, self.old_houses)

        transient = self.transient
        if transient == True:
            # #pps.pipeflow(net, mode='hydraulics')
            # 2.2 get temperature and mass flow values of old buildings for each timestep
            net2, heat_sum = calculate_new_buildings_transient(OpSimData, self.callNumber, self.m_new_cols, self.net,
                                                               self.new_houses)
            net3 = net2
            # 2.3 run calculation
            ow = _output_writer(net3, self.time_steps, ow_path=tempfile.gettempdir())
            run_timeseries(net3, self.time_steps, transient=True, mode="all", iter=10, dt=self.dt)
            res_junction = ow.np_results["res_junction.t_k"]
            sorted_res = create_results_transient(res_junction, net3)
            self.heat_sum = heat_sum
            self.heat_loss = get_losses(net3, sorted_res, self.heat_sum)
        elif transient == False:
            net2 = calculate_new_buildings(OpSimData, self.callNumber, self.m_new_cols, self.net, self.new_houses)
            net3 = net2
            pps.pipeflow(net3, mode='all', transient=False)
            net3.pipe = net3.pipe.drop(net3.res_pipe[net3.res_pipe.isna().any(axis=1)].index)
            net3.junction = net3.junction.drop(net3.res_junction[net3.res_junction.isna().any(axis=1)].index)
            results = create_results(net3)
            sorted_res = results.sort_values(by=['type', 'name'])
        # @Erik, pandapipes hier (Zeitschritt findest du in self.callNumber)

        self.saveResult(sorted_res, self.heat_loss, self.heat_loss)
        self.updateSimulationTime(increment=self.operationInterval)
        self.updateCallNumber(increment=1)

    # Additional (optional) methods from here #################################

    # Create Input table

    def saveResult(self, sorted_res, losses, heat_sum):
        # update result table

        self.result_table = sorted_res
        # self.result_table['junction_name'] = self.net.junction.name
        # self.result_table['pipe_name'] = self.net.pipe.name
        # self.result_table['v_mean'] = self.net.res_pipe.v_mean_m_per_s
        # self.result_table['p_bar'] = self.net.res_junction.p_bar
        # self.result_table['t_k'] = self.net.res_junction.t_k
        evaluation = pd.DataFrame()
        evaluation['heat_sum'] = heat_sum
        evaluation['heat_loss'] = losses
        self.evaluation.to_pickle(
            r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_evaluation" + str(
                self.callNumber) + '.p')
        self.result_table.to_pickle(

            r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_" + str(
                self.callNumber) + '.p')
        print(self.result_table)

###FILE 2
import pandas as pd
import numpy as np


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
    # print('heat value :  ' + str(heat_value))
    # timestep = OpSimData.loc['timestep', :]
    control_list = []
    route_list = set()
    input_data = pd.DataFrame()
    all_houses_off = pd.DataFrame(np.nan, index=range(new_houses.shape[0]), columns=['index', 'active'])

    for i in range(new_houses.shape[0]):
        control_list.append(m_cols[i])
        control_variable = control_list.count(control_list[i]) - 1
        idx = net.heat_exchanger[net.heat_exchanger['name'].str.endswith('he_' + str(m_cols[i]))].index[
            control_variable]
        mdot_value2 = mdot[i]
        mdot_value = mdot.iloc[i]
        all_houses_off.loc[i, 'index'] = m_cols[i]
        route_list.add(m_cols[i])
        if mdot_value < 0.01:
            net.heat_exchanger.loc[idx, 'qext_w'] = 0
            net.flow_control.loc[idx, 'controlled_mdot_kg_per_s'] = 0.00001
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
    print("net_fc" + str(net.flow_control.iloc[:, 3]))
    for ii in route_list:
        end_fcs = net.flow_control[net.flow_control['name'].str.contains('end_route_' + str(ii))].index
        if all(all_houses_off[all_houses_off['index'] == ii]['active']) == False:
            net.flow_control.iloc[end_fcs, 3] = 0.15
            mdot_v = 0.15
            heat = 4180 * mdot_v * 20
            net.heat_exchanger.iloc[end_fcs, 4] = heat
            print("following_routes_needed_bypass: " + str(end_fcs))
        else:
            net.flow_control.iloc[end_fcs, 3] = 0.00001
            mdot_v = 0.15
            heat = 4180 * mdot_v * 20
            net.heat_exchanger.iloc[end_fcs, 4] = 0

    print("net_fc" + str(net.flow_control.iloc[:, 3]))

    input_data.to_pickle(
        r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_input_" + str(
            callNumber) + '.p')
    # print(input_data)
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
    results = pd.DataFrame()
    results = res_junction
    results = results.rename(net.junction['name'].reset_index(drop=True), axis=1)

    return


def get_losses(net, results, heat_sum):
    return_junction = net.circ_pump_pressure['return_junction'].iloc[0]
    flow_junction = net.circ_pump_pressure['flow_junction'].iloc[0]
    m_flow = net.res_circ_pump_pressure['mdot_flow_kg_per_s'].iloc[0]
    return_T = results.iloc[-1:, return_junction]
    flow_T = results.iloc[-1:, flow_junction]
    return (m_flow * (flow_T - return_T) * 4180) - heat_sum


if __name__ == '__main__':
    # Start OpSim Client, Proxy and Logger
    sim_name = 'rohrnetz_sim'
    c = Client(sim_name)
    p = OpSim_Proxy(c, sim_name, c.logger)
    c.addProxy(p)
    # Connect simulation to OpSim Message Bus
    logging.info('Starting connection to OpSim message bus')
    c.reconnect('amqp://guest:guest@localhost:5672/myvhost')
