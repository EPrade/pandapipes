'''
Created on Mo Nov 14 17:08:54 2022
Code template to create pandapipes proxy for HybridBot
@author: fmarten
'''

from opsim.BaseProxy import BaseProxy
from opsim.Client import Client
from datetime import datetime
from pytz import timezone, utc
import logging
import sys
import pandapipes as pps
import pandas as pd
sys.path.append(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Python Proxy")
from pipe_toolbox import calculate_new_buildings, calculate_new_buildings_transient_m_controlled, create_results, get_losses, heat_loss_pipe, create_results_transient, save_last_timestep
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
        #self.transient = True
        self.transient = False

        # Create result table
        self.path_results = r'C:\Users\HybridBOT\Documents'\
            + r'\HybridbotFW_Git_Cosimulation\Results\pipes'
        self.result_table = pd.DataFrame(index=range(112),
                                         columns=['junction_name', 'pipe_name', 'v_mean',
                                                  'p_bar', 't_k'])
        # 1. prepare net
        # 1.1 import net
        if only_new_buildings_net == True:
            self.net = pps.from_json(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\net_16_05_ng.json")
        if only_new_buildings_net == False:
            self.net = pps.from_json(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\net_v11_04.json")

        # 1.2 get indices of heat exchangers and flow controls that need Opsim input or predefined data accordingly to them beeing "Neubaugebiet" or not
        new_building_routes = [1, 2, 31, 4, 32]
        # TODO:route 8 taucht nicht auf in den Zeitreihen
        old_building_routes = [5, 10]
        # nur neubaugebiet



        # 1.2.2 import timeseries for old houses
        houses_routes = pd.read_csv(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\straßen_nur_ng.csv")
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
        mdot_new = mdot.iloc[:,1::]
        self.mdot = mdot.loc[:, old_building_routes]
        self.t_flow_data = t_flow_data.loc[:, old_building_routes]
        self.t_return_data = t_return_data.loc[:, old_building_routes]
        self.m_cols = mdot.columns
        self.m_new_cols = mdot_new.columns

        # @Erik, hier kannst du alle deine Zeitreihen laden in self.tables
        self.createInputTable()
        self.logger.info('%s is initialized!', self.componentName)


        #timeseries setup
        self.dt = 60
        self.time_steps = range(5)

        self.t_return_set = 273.15 + 50

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
                ['p_mw', 'q_mvar']] = [meas.measurementValue/1e6, 0]
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
        #print("Opsim Input: " + str(inputTable))

        self.timestep = self.callNumber

        # 2.1 Execute every timestep : get timestep, temperature and mass flow values from opsim / itera
        #OpSimData = pd.DataFrame(1, index=['timestep', 't_flow', 't_return', 'mdot'], columns=range(25))
        OpSimData = inputTable
        #print("Calculation Timestep:" + str(self.timestep))





        print("length_self.net: " + str(self.net.junction.shape))

        #net3 = calculate_old_buildings(self.m_cols, self.timestep, net2, self.t_flow_data, self.t_return_data, self.mdot, self.old_houses)


        transient = self.transient
        if transient==True:
        # #pps.pipeflow(net, mode='hydraulics')
            if self.timestep != 0:
                t_vor = pd.read_csv(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Python Proxy\temp.t_vor.csv")
            else:
                t_vor = OpSimData.loc['TEMPERATURE_T_1_THERMAL_STORAGE', :]
            # 2.2 get temperature and mass flow values of old buildings for each timestep
            #net2, heat_sum = calculate_new_buildings_transient(OpSimData, self.callNumber, self.m_new_cols, self.net, self.new_houses)
            t_flow_last_ts = t_vor

            net2, heat_sum = calculate_new_buildings_transient_m_controlled(OpSimData, self.callNumber, self.m_new_cols, self.net, self.new_houses, t_flow_last_ts, self.t_return_set)
            net3 = net2

            # 2.3 run calculation
            ow = _output_writer(net3, self.time_steps, ow_path=tempfile.gettempdir())
            run_timeseries(net3, self.time_steps, transient=True, mode="all", iter=200, dt=self.dt)
            res_junction = ow.np_results["res_junction.t_k"]
            sorted_res = create_results_transient(res_junction, net3)
            self.heat_sum = heat_sum
            self.heat_loss = get_losses(net3, sorted_res, self.heat_sum)
            save_last_timestep(net3)
            evaluate = True
            #print(self.heat_loss)
        elif transient==False:
            net = pps.from_json(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Initial_Data\net_16_05_ng.json")
            net2 = calculate_new_buildings(OpSimData, self.callNumber, self.m_new_cols, net, self.new_houses)
            net3 = net2
            print("length_net3: " + str(net3.junction.shape))
            #pps.to_json(self.net, r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\net_original.json")
            #net3.pipe.iloc[:, 8] = 0
            print("----Conducting Pipeflow-----")
            #pps.to_json(net3, r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\net.json")
            pps.pipeflow(net3, mode='all', transient=False)
            print("----Pipeflow Successful-----")
            self.heat_sum = net3.heat_exchanger['qext_w'].sum()
            self.heat_loss = heat_loss_pipe(net3)
            net4 = net3
            net4.pipe = net4.pipe.drop(net4.res_pipe[net4.res_pipe.isna().any(axis=1)].index)
            net4.junction = net4.junction.drop(net4.res_junction[net4.res_junction.isna().any(axis=1)].index)
            print("length_net4: " + str(net4.junction.shape))
            #pps.to_json(net3, r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\net2.json")
            results = create_results(net4)
            sorted_res = results.sort_values(by=['type', 'name'])#

            evaluate = True
        # @Erik, pandapipes hier (Zeitschritt findest du in self.callNumber)



        self.saveResult(sorted_res, self.heat_loss, self.heat_sum, evaluate)
        self.updateSimulationTime(increment=self.operationInterval)
        self.updateCallNumber(increment=1)

    # Additional (optional) methods from here #################################

    # Create Input table


    def saveResult(self, sorted_res, losses, heat_sum, evaluate):
        # update result table
        print("-----Saving Results-----")
        print("heat_sum: " + str(heat_sum))
        print("losses: " + str(losses))
        self.result_table = sorted_res
        #self.result_table['junction_name'] = self.net.junction.name
        #self.result_table['pipe_name'] = self.net.pipe.name
        #self.result_table['v_mean'] = self.net.res_pipe.v_mean_m_per_s
        #self.result_table['p_bar'] = self.net.res_junction.p_bar
        #self.result_table['t_k'] = self.net.res_junction.t_k
        if evaluate == True:
            evaluation = pd.DataFrame()
            evaluation['data'] = 0
            evaluation.loc['heat_sum'] = heat_sum
            evaluation.loc['heat_loss'] = losses
            print(evaluation)
            evaluation.to_pickle(r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_evaluation" + str(self.callNumber)+'.p')
        self.result_table.to_pickle(

                r"C:\Users\HybridBOT\Documents\HybridbotFW_Git_Cosimulation\Results\pipeflow\resultfile_" + str(self.callNumber)+'.p')
        #print(self.result_table)

if __name__ == '__main__':

    # Start OpSim Client, Proxy and Logger
    sim_name = 'rohrnetz_sim'
    c = Client(sim_name)
    p = OpSim_Proxy(c, sim_name, c.logger)
    c.addProxy(p)
    # Connect simulation to OpSim Message Bus
    logging.info('Starting connection to OpSim message bus')
    c.reconnect('amqp://guest:guest@localhost:5672/myvhost')
