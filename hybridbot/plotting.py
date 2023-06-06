import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plot_data_no_dso = pd.DataFrame()
for i in range(96):
    path = r"C:\Users\eprade\Documents\hybridbot\heating grid\res\Results_06_06\Res_no_DSO\pipeflow\resultfile_evaluation" + str(i) + ".p"
    data = pd.read_pickle(path)
    data = data.transpose()
    data = data.rename(index={"data":i})
    plot_data_no_dso = pd.concat([plot_data_no_dso,data])


plot_data_dso = pd.DataFrame()
for i in range(96):
    path = r"C:\Users\eprade\Documents\hybridbot\heating grid\res\Results_06_06\Res_12_7_DSO\pipeflow\resultfile_evaluation" + str(i) + ".p"
    data = pd.read_pickle(path)
    data = data.transpose()
    data = data.rename(index={"data":i})
    plot_data_dso = pd.concat([plot_data_dso,data])

plt.figure(figsize=(10,3.8))
ax = plt.gca()
#df_opsim_results.extgrid_p_ref.plot(ax=ax, style='-',label='Reference result with nominal Heatpump power')
plot_data_dso.index, plot_data_no_dso.index = x_ax = np.linspace(0, 23.75, 96), np.linspace(0, 23.75, 96)
(plot_data_no_dso['heat_sum']/1000).plot(ax=ax, style='.-',label='OpSim result no DSO')
(plot_data_dso['heat_sum']/1000).plot(ax=ax, style='--',label='OpSim result with DSO')
#x_axis = plot_data_no_dso.index.values

#ax.set_xticks(x_ax)
#x_axis = x_ax
#ax.xaxis.set_major_locator(plt.MaxNLocator(24))
#plt.xticks(rotation=90)
#plt.axhline(y = 12.7, color = 'r', linestyle = '-')

plt.xlabel("Time[h]")
plt.ylabel("Heat Demand[kW]")
ax.legend()
plt.grid()
plt.show()