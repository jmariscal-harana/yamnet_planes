import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

##########################################
path_results = '/home/ups/Proyectos/Vigia_sonido/Datasets/airplanes_v3/holdout_data/'

# # Plots for fixed hop
# inference_hop = '024'
# title = 'Inference hop: 0.{} [s]'.format(inference_hop[1:])

# datetimes = ['20200824_234802', '20200823_025302', '20200823_210326', '20200825_011907']
# labels = ['Undersampling', 'Data augmentation', 'Data augmentation*2', 'Hybrid']


# Plots for fixed dataset
datetime = '20200825_011907
title = 'Hybrid'

inference_hops = ['096', '048', '024', '0096']
labels = ['0.96 s', '0.48 s', '0.24 s', '0.096 s']

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
ax = fig.add_axes([0., 0., 1., 1., ])

P = []
N = []

# for loc, datetime in enumerate(datetimes):
for loc, inference_hop in enumerate(inference_hops):
    # path_results_pdf_file = path_results+'test_yamnet_'+inference_hop+'.pdf'
    path_results_pdf_file = path_results+'test_yamnet_'+title+'.pdf'

    path_results_csv_file = 'test_yamnet_'+datetime+'_'+inference_hop+'_thresholding_ROC_PR.csv'
    path_results_csv = os.path.join(path_results, path_results_csv_file)

    
    ##########################################
    csv_data = pd.read_csv(path_results_csv) 

    TP = csv_data["TP"]
    TN = csv_data["TN"]
    FP = csv_data["FP"]
    FN = csv_data["FN"]

    P = np.append(P, np.mean(TP + FN)).astype(int)
    N = np.append(N, np.mean(TN + FP)).astype(int)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    PRE = TP / (TP + FP)
    REC = TPR

    TPR = TPR.tolist()
    FPR = FPR.tolist()
    PRE = PRE.tolist()
    REC = REC.tolist()

    # Calculate AUC
    AUC = np.trapz(TPR,FPR)


    ##########################################
    # Plot ROC only
    markersize = 2

    # PR plots
    ax.plot(REC,PRE,marker='o', markersize=markersize,label="{}".format(labels[loc]))

random_classifier = np.min(P/(P+N))
ax.plot([0,1],[random_classifier,random_classifier],'--',label="Random")

# fig.suptitle('{}'.format(title), fontsize=30)
ax.set_title('{}'.format(title), fontsize=30)

plt.xlim([0,1])
plt.ylim([0,1])

ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)

plt.xlabel('Recall', fontsize=22)
plt.ylabel('Precision', fontsize=22)

# ax.legend(loc=3, prop={'size': 22})

fig.savefig(path_results_pdf_file, bbox_inches='tight')


##########################################
# # Plot ROC and PR side by side
# fig, ax = plt.subplots(1, 2)
# fig.suptitle('Yamnet')

# markersize = 1

# # ROC plots
# ax[0].scatter(FPr4,TPr4,marker='.', s=markersize,label="Undersampling AUC ={:10.4f}".format(auc4))
# ax[0].scatter(FPr3,TPr3,marker='.', s=markersize,label="Data aug      AUC ={:10.4f}".format(auc3))
# ax[0].scatter(FPr2,TPr2,marker='.', s=markersize,label="Data aug*2    AUC ={:10.4f}".format(auc2))
# ax[0].scatter(FPr1,TPr1,marker='.', s=markersize,label="Hybrid        AUC ={:10.4f}".format(auc1))

# ax[0].plot([0,1],[0,1],'--')


# # PR plots
# ax[1].scatter(rec4,prec4,marker='.', s=markersize,label="Undersampling    AUC ={:10.4f}".format(auc3))
# ax[1].scatter(rec3,prec3,marker='.', s=markersize,label="Data aug         AUC ={:10.4f}".format(auc3))
# ax[1].scatter(rec2,prec2,marker='.', s=markersize,label="Data aug*2       AUC ={:10.4f}".format(auc2))
# ax[1].scatter(rec1,prec1,marker='.', s=markersize,label="Hybrid           AUC ={:10.4f}".format(auc1))

# ax[1].plot([0,1],[0.5,0.5],'--')

# for i,a in enumerate(ax.flat):
#     if i == 0:
#         a.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
#     else:
#         a.set(xlabel='True Positive Rate (Recall)', ylabel='Precision')

# ax[0].legend(loc=4)
# plt.show()

# fig.savefig("yamnet_ROC_PR.pdf", bbox_inches='tight')
