import matplotlib.pyplot as plt
import pandas as pd



plot_data_no_dso = pd.DataFrame()
for i in range(96):
    path = r"C:\Users\eprade\Documents\hybridbot\heating grid\res\res2\Res_noDSO\Results\pipeflow\resultfile_evaluation" + str(i) + ".p"
    data = pd.read_pickle(path)
    data = data.transpose()
    data = data.rename(index={"data":i})
    plot_data_no_dso = pd.concat([plot_data_no_dso,data])


plot_data_dso = pd.DataFrame()
for i in range(96):
    path = r"C:\Users\eprade\Documents\hybridbot\heating grid\res\res2\Res_12.7_DSO\Results\pipeflow\resultfile_evaluation" + str(i) + ".p"
    data = pd.read_pickle(path)
    data = data.transpose()
    data = data.rename(index={"data":i})
    plot_data_dso = pd.concat([plot_data_dso,data])

ax = plt.gca()
#df_opsim_results.extgrid_p_ref.plot(ax=ax, style='-',label='Reference result with nominal Heatpump power')
plot_data_no_dso['heat_sum'].plot(ax=ax, style='.-',label='OpSim result no DSO')
plot_data_dso['heat_sum'].plot(ax=ax, style='--',label='OpSim result with DSO')
ax.set_xticks(plot_data_no_dso.index.values)
plt.xticks(rotation=90)
plt.axhline(y = 12.7, color = 'r', linestyle = '-')
ax.legend()
plt.grid()
plt.show()