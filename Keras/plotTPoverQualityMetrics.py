import csv
import matplotlib.pyplot as plt
import numpy as np

root_dir = './Data/NNResults/'
ftitle_csv = 'metrics_combined_without_mgc_fft_2_3000epochs'
fname_csv = root_dir + ftitle_csv + '.csv'
data_list = []
# Read overview CSV file specific to each layout
with open(fname_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Append ordered dict to list
        data_list.append(row)

n = []
tp = []
tpr = []
spc = []
fpr = []
acc = []
mcc = []
for line in data_list:
    n.append(line['n'])
    tp.append(line['TP'])
    tpr.append(line['TPR'])
    spc.append(line['SPC'])
    fpr.append(line['FPR'])
    acc.append(line['ACC'])
    mcc.append(line['MCC'])

n = np.asarray(n, dtype='int32')
tp = np.asarray(tp, dtype='int32')
tpr = np.asarray(tpr, dtype='float32')
spc = np.asarray(spc, dtype='float32')
fpr = np.asarray(fpr, dtype='float32')
acc = np.asarray(acc, dtype='float32')
mcc = np.asarray(mcc, dtype='float32')

b, m = np.polynomial.polynomial.Polynomial.fit(tp, tpr, 1)
print(b, m)
tpr_corr = b + m * tpr
print(tpr_corr)

tp_lin = np.linspace(0, np.max(tp, axis=0), np.max(tp, axis=0)+1)
print(tp_lin)

tpr_lin = np.zeros(tp_lin.size+1)
for i in range(tp.size):
    tpr_lin[tp[i]] = tpr[i]
tpr_lin = tpr_lin[:-1]


plt.plot(tp_lin, tpr_lin, 'x')
plt.xlabel('Zahl der problematischen Sektoren im Layout')
plt.ylabel('Empfindlichkeit (TPR)')
fname_pdf = root_dir + 'figures/' + ftitle_csv + '_TPRoverTP.pdf'
plt.savefig(fname_pdf)
plt.clf()


acc_lin = np.zeros(tp_lin.size+1)
for i in range(tp.size):
    acc_lin[tp[i]] = acc[i]
acc_lin = acc_lin[:-1]


plt.plot(tp_lin, acc_lin, 'x')
plt.xlabel('Zahl der problematischen Sektoren im Layout')
plt.ylabel('Genauigkeit (ACC)')
fname_pdf = root_dir + 'figures/' + ftitle_csv + '_ACCoverTP.pdf'
plt.savefig(fname_pdf)
plt.clf()


fpr_lin = np.zeros(tp_lin.size+1)
for i in range(tp.size):
    fpr_lin[tp[i]] = fpr[i]
fpr_lin = fpr_lin[:-1]


plt.plot(tp_lin, fpr_lin, 'x')
plt.xlabel('Zahl der problematischen Sektoren im Layout')
plt.ylabel('Falsch-Positiv-Rate (FPR)')
fname_pdf = root_dir + 'figures/' + ftitle_csv + '_FPRoverTP.pdf'
plt.savefig(fname_pdf)
plt.clf()


spc_lin = np.zeros(tp_lin.size+1)
for i in range(tp.size):
    spc_lin[tp[i]] = spc[i]
spc_lin = spc_lin[:-1]

plt.plot(tp_lin, spc_lin, 'x')
plt.xlabel('Zahl der problematischen Sektoren im Layout')
plt.ylabel('Spezifizität (SPC)')
fname_pdf = root_dir + 'figures/' + ftitle_csv + '_SPCoverTP.pdf'
plt.savefig(fname_pdf)
plt.clf()


mcc_lin = np.zeros(tp_lin.size+1)
for i in range(tp.size):
    mcc_lin[tp[i]] = mcc[i]
mcc_lin = mcc_lin[:-1]

plt.plot(tp_lin, mcc_lin, 'x')
plt.xlabel('Zahl der problematischen Sektoren im Layout')
plt.ylabel('Matthews Korrelationskoeffizient (MCC)')
# axes = plt.gca()
# axes.set_ylim([0.0, 1.0])

fname_pdf = root_dir + 'figures/' + ftitle_csv + '_MCCoverTP.pdf'
plt.savefig(fname_pdf)
plt.clf()