import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.preprocessing import Normalizer

home = str(Path.home())

results_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/Final_results'

weights = pd.read_csv(results_path + '/Weights/weights_summary.csv')
weights = weights.loc[:, ['Dataset', 'F1', 'ROD', 'Fold', 'Fairness']]
weights['Method'] = 'Weighted FC-FS-BS'

adult_baselines = pd.read_csv(results_path + '/Baselines/Adult/baselines_adult_results.csv')
COMPAS_baselines = pd.read_csv(results_path + '/Baselines/COMPAS/baselines_COMPAS_results.csv')
credit_baselines = pd.read_csv(results_path + '/Baselines/German credit/baselines_german_credit_results.csv')

adult_baselines.loc[adult_baselines['Method'] == 'dropped', 'Method'] = 'Dropped'
adult_baselines.loc[adult_baselines['Method'] == 'original', 'Method'] = 'Original'
adult_baselines.loc[adult_baselines['Method'] == 'capuchin', 'Method'] = 'Capuchin'
COMPAS_baselines.loc[COMPAS_baselines['Method'] == 'dropped', 'Method'] = 'Dropped'
COMPAS_baselines.loc[COMPAS_baselines['Method'] == 'original', 'Method'] = 'Original'
COMPAS_baselines.loc[COMPAS_baselines['Method'] == 'capuchin', 'Method'] = 'Capuchin'
credit_baselines.loc[credit_baselines['Method'] == 'dropped', 'Method'] = 'Dropped'
credit_baselines.loc[credit_baselines['Method'] == 'original', 'Method'] = 'Original'
credit_baselines.loc[credit_baselines['Method'] == 'capuchin', 'Method'] = 'Capuchin'

credit_baselines['Dataset'] = 'German credit'
COMPAS_baselines['Dataset'] = 'COMPAS'
adult_baselines['Dataset'] = 'Adult'

credit_baselines['Fairness'] = 0
COMPAS_baselines['Fairness'] = 0
adult_baselines['Fairness'] = 0

adult_baselines = adult_baselines.loc[:, ['Dataset', 'F1', 'ROD', 'Fold', 'Method', 'Fairness']]
COMPAS_baselines = COMPAS_baselines.loc[:, ['Dataset', 'F1', 'ROD', 'Fold', 'Method', 'Fairness']]
credit_baselines = credit_baselines.loc[:, ['Dataset', 'F1', 'ROD', 'Fold', 'Method', 'Fairness']]

union = pd.concat([adult_baselines, COMPAS_baselines, credit_baselines, weights], ignore_index=True)


columns = list(adult_baselines)
results = pd.DataFrame(columns=columns)
for ids, set in enumerate(['Adult', 'COMPAS', 'German credit']):
    for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for i in range(1, 6):
            copy = union.copy()
            v_1 = copy.loc[(copy['Fold'] == i) & (copy['Dataset'] == set) & ((copy['Fairness'] == f) | (copy['Fairness'] == 0)), 'ROD'].to_numpy() * -1
            norm = Normalizer().fit_transform(v_1.reshape(1, -1))
            norm = norm.reshape(v_1.shape[0], 1)
            copy.loc[(copy['Fold'] == i) & (copy['Dataset'] == set) & ((copy['Fairness'] == f) | (copy['Fairness'] == 0)), 'ROD'] = 1 - norm
            print(copy)
            results = results.append(copy.loc[(copy['Fold'] == i) & (copy['Dataset'] == set) & ((copy['Fairness'] == f) | (copy['Fairness'] == 0)), ['Dataset', 'F1', 'ROD', 'Fold', 'Method', 'Fairness']], ignore_index=True)

print(results.groupby(['Method', 'Dataset'])['ROD', 'F1'].mean())


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, gridspec_kw={'hspace': 0.05, 'wspace':0.01}, sharey='row')
ax1.set_title('F1')
ax2.set_title('ROD')
#ax3.set_title('German credit')

###### Adult
my_pal = 'bright'
sns.set_style("whitegrid")
sns.boxplot(y='F1', x='Fairness',
                 data=results.loc[results['Dataset'] == 'Adult', :],
                 palette=my_pal, hue='Method', ax=ax1, width=1.5, whis=2.5)#,
                 #hue='Method')
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

ax1.set_ylabel('Adult')

sns.boxplot(y='ROD', x='Fairness',
                 data=results.loc[results['Dataset']=='Adult',:],
                 palette=my_pal, hue='Method', ax=ax2, width=1.5, whis=2.5)#,
                 #hue='Method')

ax1.set(ylim=(0, 1.01), xlim=(-1, 10))
ax2.set(ylim=(0, 1.01), xlim=(-1, 10))


#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
#_ = plt.xticks(rotation=90)
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.grid(axis='y')
ax2.grid(axis='y')

####### COMPAS

my_pal = 'bright'
sns.set_style("whitegrid")
sns.boxplot(y='F1', x='Fairness',
                 data=results.loc[results['Dataset'] == 'COMPAS', :],
                 palette=my_pal, hue='Method', ax=ax3, width=1.5, whis=2.5)#,
                 #hue='Method')
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

ax3.set_ylabel('COMPAS')
sns.boxplot(y='ROD', x='Fairness',
                 data=results.loc[results['Dataset'] == 'COMPAS', :],
                 palette=my_pal, hue='Method', ax=ax4, width=1.5, whis=2.5)#,
                 #hue='Method')

ax3.set(ylim=(0, 1.01), xlim=(-1, 10))
ax4.set(ylim=(0, 1.01), xlim=(-1, 10))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
#_ = plt.xticks(rotation=90)
ax4.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax3.get_legend().remove()
ax4.get_legend().remove()
ax3.grid(axis='y')
ax4.grid(axis='y')

######## German credit

my_pal = 'bright'
sns.set_style("whitegrid")
sns.boxplot(y='F1', x='Fairness',
                 data=results.loc[results['Dataset'] == 'German credit',:],
                 palette=my_pal, hue='Method', ax=ax5, width=1.5, whis=2.5)#,
                 #hue='Method')
# ax5.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
ax5.set_ylabel('German credit')

sns.boxplot(y='ROD', x='Fairness',
                 data=results.loc[results['Dataset'] == 'German credit', :],
                 palette=my_pal, hue='Method', ax=ax6, width=1.5, whis=2.5)#,
                 #hue='Method')

ax5.set(ylim=(0, 1.01), xlim=(-1, 10))
ax6.set(ylim=(0, 1.01), xlim=(-1, 10))
#plt.setp(ax.get_legend().get_texts(), fontsize='5') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='6')
#_ = plt.xticks(rotation=90)
# ax6.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
#ax3.legend(loc='best')
ax6.get_legend().remove()
ax5.legend(loc='lower left', ncol=2, fontsize='small')
ax5.grid(axis='y')
ax6.grid(axis='y')
ax5.set_xlabel('Fairness Importance')
ax6.set_xlabel('Fairness Importance')

for ax in f.get_axes():
    ax.label_outer()

plt.show()


