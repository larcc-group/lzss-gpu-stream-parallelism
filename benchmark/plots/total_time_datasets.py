import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  dataset_helper import DatasetHelper,color_for,better_name

def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2. - .13, 1.0*height,
                '{:10.0f}'.format(round(height,0)),
                ha='center', va='bottom',size=10)
mpl.rc('hatch',color = "white")
with DatasetHelper() as dataset_helper:
    figs, axs = plt.subplots(1,2,False,True)
    label_memory = ["Memória","Arquivo"]
    for j,mode  in enumerate(["memory","file"]):
        ax = axs[j]
        fig = figs
        plans = [x[0] for x in dataset_helper.select_data('''SELECT distinct plan FROM result''')]
        datasets = [x[0] for x in dataset_helper.select_data('''SELECT distinct dataset FROM result''')]
        ind = np.arange(len(datasets))  # the x locations for the groups

        width = (1 - 0.1)/len(datasets)  # the width of the bars

        rects = []
        
        for i,plan in enumerate(plans):
            print(plan,i)
            mean = []
            std = []
            for dataset in datasets:
                vec = dataset_helper.select_vector("SELECT total_time FROM result WHERE plan ='{0}' and dataset = '{1}' and mode='{2}'".format(plan,dataset,mode))
                mean.append(vec.mean())
                std.append(vec.std())
            print("Mean:",mean)
            print("Std:",std)
            style = color_for(plan)
            color = style[0]
            rect = ax.bar(ind + (width * i), mean, width, yerr=std,error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),
                            label=plan,color=color,hatch=style[2])
            rects.append(rect)
            autolabel(ax,rect)
            
        # rects2 = ax.bar(ind + width/2, women_means, width, yerr=women_std,
        #                 color='IndianRed', label='Women')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Tempo(seg.)')
        ax.set_title(label_memory[j])
        ax.set_xticks(ind + (0.9/len(datasets) ))
        ax.set_xticklabels([better_name(x) for x in  datasets])
        # ax.set_yscale('log')
        ax.legend()

        fig.set_size_inches(12.5, 5.5)
        fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)  # create some space below the plots by increasing the bottom-value

# plt.yscale("log", nonposy="clip")
plt.savefig("img/lzss_total_time.pdf",bbox_inches='tight', transparent="True", pad_inches=0)
