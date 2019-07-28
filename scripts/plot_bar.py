import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 6
accuracy_trw = (53.11, 46.78, 89.33, 94.78, 98.44, 99.67)
accuracy_sgd = (27.78, 33.94, 86.94, 63.83, 91.33, 98.56)
time_trw = (9.73, 8.88, 9.17, 11.01, 10.31, 49.38)
time_sgd = (9.74, 8.83, 10.80, 9.61, 11.82, 29.22)

# create plot
fig, ax = plt.subplots(figsize=[12,10])
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, accuracy_trw, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Trust Region Newton (TRN)')

rects2 = plt.bar(index + bar_width, accuracy_sgd, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Stochastic Gradient Descent (SGD)')

plt.xlabel('Features Group')
plt.ylabel('Accuracy (\%)')
plt.xticks(index + bar_width, ('Time-Domain', 'Spectrum', 'MFCCs', 'Chroma',
                               'Freq-Domain', 'All'))
plt.legend(loc=2)

plt.tight_layout()
plt.savefig('trn_sgd_accuracy.pdf')

fig, ax = plt.subplots(figsize=[12,10])

rects1 = plt.bar(index, time_trw, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Trust Region Newton (TRN)')

rects2 = plt.bar(index + bar_width, time_sgd, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Stochastic Gradient Descent (SGD)')

plt.xlabel('Features Group')
plt.ylabel('Training Time (sec)')
plt.xticks(index + bar_width, ('Time-Domain', 'Spectrum', 'MFCCs', 'Chroma',
                               'Freq-Domain', 'All'))
plt.legend(loc=2)

plt.tight_layout()
plt.savefig('trn_sgd_time.pdf')
plt.show()
