import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

path = 'results/run_3.csv'
path_out = 'results/run_3_processed.csv'

df = pd.read_csv(path)

df.loc[df['model_name'].str.contains('TheBloke/Mistral-7B-v0.1-GPTQ'), 'method'] = 'Publiskais GPTQ 4bit'
df.loc[df['model_name'].str.contains('TheBloke/Mistral-7B-v0.1-AWQ'), 'method'] = 'Publiskais AWQ 4bit'
df.loc[df['model_name'].str.contains('TheBloke/Llama-2-7B-GPTQ'), 'method'] = 'Publiskais GPTQ 4bit'
df.loc[df['model_name'].str.contains('TheBloke/Llama-2-7B-AWQ'), 'method'] = 'Publiskais AWQ 4bit'

df.loc[df['method'] == 'original', 'method'] = 'Nekvantizēts'

prefixes = ['speed', 'loss', 'perplexity', 'similarities', 'copa', 'wic', 'axg']

for prefix in prefixes:
    cols = [col for col in df.columns if prefix in col]
    df[f"mean_{prefix}"] = df[cols].mean(axis=1)
    df[f"std_{prefix}"] = df[cols].std(axis=1)
    df[f"min_{prefix}"] = df[cols].min(axis=1)
    df[f"max_{prefix}"] = df[cols].max(axis=1)

columns = [col for col in df.columns if 'copa' in col or 'wic' in col or 'axg' in col]
df['mean_accuracy'] = df[columns].mean(axis=1)
df['max_accuracy'] = df[columns].max(axis=1)
df['mean_similarities'] = (df['mean_similarities'] + 1) / 2
df['max_similarities'] = (df['max_similarities'] + 1) / 2

df.loc[(df['method'] == 'bitsandbytes') & (df['bits'] == 4), 'method'] = 'bitsandbytes 4bit'
df.loc[(df['method'] == 'bitsandbytes') & (df['bits'] == 8), 'method'] = 'bitsandbytes 8bit'

df.loc[(df['method'] == 'gptq') & (df['bits'] == 2), 'method'] = 'gptq 2bit'
df.loc[(df['method'] == 'gptq') & (df['bits'] == 3), 'method'] = 'gptq 3bit'
df.loc[(df['method'] == 'gptq') & (df['bits'] == 4), 'method'] = 'gptq 4bit'
df.loc[(df['method'] == 'gptq') & (df['bits'] == 8), 'method'] = 'gptq 8bit'
df.loc[(df['method'] == 'awq'), 'method'] = 'awq 4bit'

df.to_csv(path_out, index=False)

fig, ax = plt.subplots()
df.boxplot(column=['mean_speed'], by='method', ax=ax, grid=False)
plt.suptitle('')
plt.title('Dažādo metožu ātrums')
plt.ylabel('Ātrums (taloni sekundē) (Augstāks ir labāk)')
plt.xlabel('Metode')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()

cols = ['method', 'mean_speed', 'max_speed']
df_speed = df[cols].groupby('method').mean()
df_speed['delta_mean'] = df_speed['mean_speed'] - df_speed.loc['Nekvantizēts', 'mean_speed']
df_speed['delta_max'] = df_speed['max_speed'] - df_speed.loc['Nekvantizēts', 'max_speed']

for method in df_speed.index:
    ttest = ttest_ind(df[df['method'] == 'Nekvantizēts']['mean_speed'], df[df['method'] == method]['mean_speed'])
    df_speed.loc[method, 'ttest'] = ttest.pvalue
    if ttest.pvalue < 0.05:
        df_speed.loc[method, 'ttest'] = ttest.pvalue
    else:
        df_speed.loc[method, 'ttest'] = ttest.pvalue
df_speed = df_speed.round(2)
df_speed.sort_values(by='delta_max', inplace=True)

print(df_speed.to_csv(sep='&').replace('\n', '\\\\\n'))


fig, ax = plt.subplots()
df.boxplot(column=['mean_accuracy'], by='method', ax=ax, grid=False, whis=10.)
plt.suptitle('')
plt.title('Dažādo metožu precizitāte')
plt.ylabel('Precizitāte (Augstāks ir labāk)')
plt.xlabel('Metode')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


cols = ['method', 'mean_accuracy', 'max_accuracy']
df_acc = df[cols].groupby('method').mean()
df_acc['delta_mean'] = df_acc['mean_accuracy'] - df_acc.loc['Nekvantizēts', 'mean_accuracy']
df_acc['delta_max'] = df_acc['max_accuracy'] - df_acc.loc['Nekvantizēts', 'max_accuracy']

for method in df_acc.index:
    ttest = ttest_ind(df[df['method'] == 'Nekvantizēts']['mean_accuracy'], df[df['method'] == method]['mean_accuracy'])
    df_acc.loc[method, 'ttest'] = ttest.pvalue
    if ttest.pvalue < 0.05:
        df_acc.loc[method, 'ttest'] = ttest.pvalue
    else:
        df_acc.loc[method, 'ttest'] = ttest.pvalue

df_acc = df_acc.round(2)
df_acc.sort_values(by='delta_max', inplace=True)

print(df_acc.to_csv(sep='&').replace('\n', '\\\\\n'))


fig, ax = plt.subplots()
df.boxplot(column=['mean_similarities'], by='method', ax=ax, grid=False, whis=100.)
plt.suptitle('')
plt.title('Dažādo metožu izvadu semantiskā līdzība')
plt.ylabel('Semantiska līdzība (Augstāks ir labāk)')
plt.xlabel('Metode')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

cols = ['method', 'mean_similarities', 'max_similarities']
df_sim = df[cols].groupby('method').mean()
df_sim['delta_mean'] = df_sim['mean_similarities'] - df_sim.loc['Nekvantizēts', 'mean_similarities']
df_sim['delta_max'] = df_sim['max_similarities'] - df_sim.loc['Nekvantizēts', 'max_similarities']

for method in df_sim.index:
    ttest = ttest_ind(df[df['method'] == 'Nekvantizēts']['mean_similarities'], df[df['method'] == method]['mean_similarities'])
    df_sim.loc[method, 'ttest'] = ttest.pvalue
    if ttest.pvalue < 0.05:
        df_sim.loc[method, 'ttest'] = ttest.pvalue
    else:
        df_sim.loc[method, 'ttest'] = ttest.pvalue

df_sim = df_sim.round(2)
df_sim.sort_values(by='delta_max', inplace=True)
print(df_sim.to_csv(sep='&').replace('\n', '\\\\\n'))

fig, ax = plt.subplots()
df.boxplot(column=['mean_perplexity'], by='method', ax=ax, grid=False, whis=100.)
plt.suptitle('')
plt.title('Dažādo metožu perplexity vērtības')
plt.ylabel('Perplexity (Zemāks ir labāk)')
plt.xlabel('Metode')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

cols = ['method', 'mean_perplexity', 'min_perplexity']
df_perp = df[cols].groupby('method').mean()
df_perp['delta_mean'] = df_perp['mean_perplexity'] - df_perp.loc['Nekvantizēts', 'mean_perplexity']
df_perp['delta_min'] = df_perp['min_perplexity'] - df_perp.loc['Nekvantizēts', 'min_perplexity']

for method in df_perp.index:
    ttest = ttest_ind(df[df['method'] == 'Nekvantizēts']['mean_perplexity'], df[df['method'] == method]['mean_perplexity'])
    df_perp.loc[method, 'ttest'] = ttest.pvalue
    if ttest.pvalue < 0.05:
        df_perp.loc[method, 'ttest'] = ttest.pvalue
    else:
        df_perp.loc[method, 'ttest'] = ttest.pvalue

df_perp = df_perp.round(2)
df_perp.sort_values(by='delta_min', inplace=True)
print(df_perp.to_csv(sep='&').replace('\n', '\\\\\n'))


mistral_models = []
llama2_models = []
llama3_models = []
for model in df['model_name'].unique():
    if 'mistral' in model.lower():
        mistral_models.append(model)
    elif 'llama-2' in model.lower():
        llama2_models.append(model)
    elif 'llama-3' in model.lower():
        llama3_models.append(model)

df.loc[df['model_name'].isin(mistral_models), 'model_name'] = 'Mistral'
df.loc[df['model_name'].isin(llama2_models), 'model_name'] = 'Llama-2'
df.loc[df['model_name'].isin(llama3_models), 'model_name'] = 'Llama-3'
df['gpu_utilization'] = df['gpu_utilization'] / 1024 / 1024 / 1024

for model_name in ['Mistral', 'Llama-2', 'Llama-3']:
    df_model = df[df['model_name'] == model_name]
    # Atņemam nekvantizetā izmēru no gpt modeļiem, jo kvantizācijas bibliotēka korekti neatbrīvo atmiņu pēc kvantizācijas
    df_model.loc[df_model['method'] == 'gptq 2bit', 'gpu_utilization'] -= df_model.loc[df_model['method'] == 'Nekvantizēts', 'gpu_utilization'].values[0]
    df_model.loc[df_model['method'] == 'gptq 3bit', 'gpu_utilization'] -= df_model.loc[df_model['method'] == 'Nekvantizēts', 'gpu_utilization'].values[0]
    df_model.loc[df_model['method'] == 'gptq 4bit', 'gpu_utilization'] -= df_model.loc[df_model['method'] == 'Nekvantizēts', 'gpu_utilization'].values[0]

    fig, ax = plt.subplots()
    df_model.boxplot(column=['gpu_utilization'], by='method', ax=ax, grid=False)
    plt.suptitle('')
    plt.title(f'{model_name} atmiņas izmantošana')
    plt.ylabel('Atmiņas izmantošana (GB)')
    plt.xlabel('Metode')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    cols = ['method', 'gpu_utilization']
    df_gpu = df_model[cols].groupby('method').mean()
    df_gpu['delta'] = df_gpu['gpu_utilization'] - df_gpu.loc['Nekvantizēts', 'gpu_utilization']

    df_gpu = df_gpu.round(2)
    df_gpu.sort_values(by='delta', inplace=True)
    print(df_gpu.to_csv(sep='&').replace('\n', '\\\\\n'))


df_gptq = df[df['method'].str.contains('gptq')]

fig, ax = plt.subplots()
df_gptq.boxplot(column=['mean_accuracy'], by='dataset', ax=ax, grid=False)
plt.suptitle('')
plt.title('GPTQ precizitāte')
plt.ylabel('Precizitāte (Augstāks ir labāk)')
plt.xlabel('Datukopa')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

cols = ['dataset', 'mean_accuracy', 'max_accuracy']
df_acc = df_gptq[cols].groupby('dataset').mean()
df_acc['delta_mean'] = df_acc['mean_accuracy'] - df_acc.loc['c4-new', 'mean_accuracy']
df_acc['delta_max'] = df_acc['max_accuracy'] - df_acc.loc['c4-new', 'max_accuracy']
for dataset in df_acc.index:
    ttest = ttest_ind(df_gptq[df_gptq['dataset'] == 'c4-new']['mean_accuracy'], df_gptq[df_gptq['dataset'] == dataset]['mean_accuracy'])
    df_acc.loc[dataset, 'ttest'] = ttest.pvalue
    if ttest.pvalue < 0.05:
        df_acc.loc[dataset, 'ttest'] = ttest.pvalue
    else:
        df_acc.loc[dataset, 'ttest'] = ttest.pvalue
df_acc = df_acc.round(2)
df_acc.sort_values(by='delta_max', inplace=True)
print(df_acc.to_csv(sep='&').replace('\n', '\\\\\n'))


# Get top 3 results from accuracy for each model from the full df
df_acc = df[df['method'] != 'Nekvantizēts'][~df['method'].str.contains('Publiskais')]
df_acc = df_acc.sort_values(by='mean_accuracy', ascending=False)
df_acc = df_acc.groupby('model_name').head(3)

print(df_acc[['model_name', 'method', 'mean_accuracy', 'version', 'q_group_size', 'llm_int8_threshold', 'block_size']].to_csv(sep='&', index=False).replace('\n', '\\\\\n'))

# Get top 3 results from similarity for each model from the full df
df_sim = df[df['method'] != 'Nekvantizēts'][~df['method'].str.contains('Publiskais')][df['method'].str.contains('4bit')]
df_sim = df_sim.sort_values(by='mean_accuracy', ascending=False)
df_sim = df_sim.groupby('model_name').head(3)

print(df_sim[['model_name', 'method', 'mean_accuracy', 'version', 'q_group_size', 'model_seqlen', 'exllama_version', 'group_size', 'dataset', 'damp_percent', 'batch_size', 'desc_act', 'sym', 'true_sequential']].to_csv(sep='&', index=False).replace('\n', '\\\\\n'))