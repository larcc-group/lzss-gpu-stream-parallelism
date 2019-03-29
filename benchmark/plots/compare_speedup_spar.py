
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  dataset_helper import DatasetHelper,available_colors
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams['font.family'] = 'Helvativa'
plt.rcParams['font.serif'] = 'Helvativa'
plt.rcParams['font.monospace'] = 'Helvativa'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


plt.tick_params(labelsize=20)
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

font = {'family' : 'Helvetica',
       
        'size'   : 20}
mpl.rcParams.update({'font.size': 20})
mpl.rc('font', **font)
hfont = {'fontname':'Helvetica','fontsize':20}
hfont2 = {'fontname':'Helvetica','fontsize':14}
# Colorblind-friendly colors
colors = [ [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255], 
          [213/255,94/255,0], [0,114/255,178/255]]
cuda_style = [0,158/255,115/255]
spar_style = [0,114/255,178/255]
opencl_style = [213/255,94/255,0]
style_plot = [[230/255,159/255,0],  [86/255,180/255,233/255], [0,158/255,115/255],[0,114/255,178/255]]

hatches = ['//', "\\\\" , '||', '+']   #, '+', 'x', 'o', 'O', '.', '*'}
# style_plot = [cuda_style,opencl_style,cuda_style,opencl_style,spar_style,spar_style]


# Create the figure

sql = """select dataset,plan,threads + 1 as threads, (select avg(total_time) from result as result_sequential where result_sequential.mode=  result.mode and result_sequential.plan='CPU_SEQUENTIAL_ORIGINAL' and result_sequential.dataset = result.dataset ) as sequential_time, avg(total_time)  as avg_time, STDEV(total_time) as stdev

 from result where mode='{0}' and plan <> 'CPU_ORIGINAL'
AND plan <> 'CPU_SEQUENTIAL' and plan <> 'CPU_SEQUENTIAL_ORIGINAL' AND dataset like "{1}"
 group by dataset,plan,threads
--  order by dataset,plan,threads
 
UNION 
SELECT dataset,plan,threads,sequential_time, avg_time,stdev FROM ( 
select dataset,plan,threads + 1 as threads, (select avg(total_time) from result as result_sequential where result_sequential.mode=  result.mode and result_sequential.plan='CPU_SEQUENTIAL_ORIGINAL' and result_sequential.dataset = result.dataset) as sequential_time, avg(total_time)  as avg_time, STDEV(total_time) as stdev

 from result where mode='{0}'
AND plan = 'CPU_SEQUENTIAL' and plan <> 'CPU_SEQUENTIAL_ORIGINAL' AND dataset like "{1}"
 and threads = 40
 group by dataset,plan,threads

) AS T 

 UNION 
SELECT dataset,plan,threads,sequential_time, avg_time,stdev FROM ( 
select dataset,plan,threads + 1 as threads, (select avg(total_time) from result as result_sequential where result_sequential.mode=  result.mode and result_sequential.plan='CPU_SEQUENTIAL_ORIGINAL' and result_sequential.dataset = result.dataset) as sequential_time, avg(total_time)  as avg_time, STDEV(total_time) as stdev

 from result where mode='{0}'
AND plan = 'CPU_SEQUENTIAL' and plan <> 'CPU_SEQUENTIAL_ORIGINAL' AND dataset like "{1}"
 group by dataset,plan,threads
 order by 5  limit 1
) AS T 



 order by dataset,threads,plan"""

datasets = [
    {"name":"Silesia", "dataset_name":"%silesia%", "file_name":"silesia"},
    {"name":"Linux Source", "dataset_name":"%linux%", "file_name":"linux"}
]

def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2. - .13, 1.0*height + 3,
                '{:10.2f}'.format(round(height,2)),
                ha='center', va='bottom',**hfont2)

def format_name(name, threads):
    if name == "CPU_SEQUENTIAL":
        return "SPar("+str(threads)+" THREAD"+("S)" if str(threads) != "1" else ")")
    return name.replace("GPU_","")+"("+str(threads)+" GPU"+("S)" if threads > 1 else ")")
    
for dataset in datasets:
    for mode in ["memory","file"]:
        with DatasetHelper() as db:
            sql_formatted = sql.format(mode,dataset["dataset_name"])
            print("SQL:{0}".format(sql_formatted))

            data = [x for x in db.select_data(sql_formatted)]
            labels = [format_name(x[1] ,x[2] ) for x in data]
            print(labels)
            y_pos = np.arange(len(labels))  
            print(y_pos)   
            values = [x[3]/x[4] for x in data]
            # tempo_medio_sequential / (tempo_medio_p + desvio_tempo_p) - tempo_medio_sequential / tempo_medio_p
            y_error = [ ( x[3] / (x[4] + x[5] ) - x[3]/x[4]) if x[5] != None else 0  for x in data]
            print(y_error)
            plt.rcdefaults()
            fig, ax = plt.subplots()
            gpu_1 = [values[0], values[1]]
            gpu_1_err = [y_error[0], y_error[1]]
            gpu_2 = [values[2], values[3]]
            gpu_2_err = [y_error[2], y_error[3]]
            thread_8 = [values[4]]
            thread_8_err = [y_error[4]]
            thread_max = [values[5]]
            thread_max_err = [y_error[5]]
            rect = ax.bar([0], thread_8,align='center',edgecolor="black",yerr = thread_8_err ,color=style_plot[2],error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),hatch=hatches[2],label="8 Threads")
            autolabel(ax,rect)

            rect = ax.bar([1], thread_max,align='center',edgecolor="black",yerr = thread_max_err ,color=style_plot[3],error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),hatch=hatches[3],label="{0} Threads".format(data[5][2]))
            autolabel(ax,rect)
            rect = ax.bar([2,3], gpu_1,align='center',edgecolor="black",yerr = gpu_1_err ,color=style_plot[0],error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),hatch=hatches[0],label="1 GPU")
            autolabel(ax,rect)
            rect = ax.bar([4,5], gpu_2,align='center',edgecolor="black",yerr = gpu_2_err ,color=style_plot[1],error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),hatch=hatches[1],label="2 GPUs")
            autolabel(ax,rect)

            # rect = ax.bar(y_pos, values,align='center',yerr = y_error ,color=style_plot,error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),hatch=hatches)
            ax.set_xticks(y_pos)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.set_ylabel("Speedup",**hfont)
            ax.set_xticklabels(["SPar","SPar","SPar +\n CUDA","SPar +\n OpenCL","SPar +\n CUDA","SPar +\n OpenCL"], rotation=45,**hfont2)
            ax.set_title("LZSS{0}({1})".format((" in Memory" if mode == "memory" else ""),dataset["name"]),**hfont)
            ax.legend(fontsize=12)
            bottom, top = plt.ylim()  # return the current ylim
            ax.set_ylim((0, max(values)+12)) 
            
            plt.savefig("img/compare_speedup_{1}_{0}_spar.pdf".format(mode,dataset["file_name"]),bbox_inches='tight', pad_inches=0)
            plt.savefig("img/compare_speedup_{1}_{0}_spar.png".format(mode,dataset["file_name"]),bbox_inches='tight', pad_inches=0)
            # plt.show()