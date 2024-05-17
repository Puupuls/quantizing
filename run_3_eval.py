import pandas as pd

path = 'results/run_3.csv'
path_out = 'results/run_3_processed.csv'

df = pd.read_csv(path)

prefixes = ['speed', 'loss', 'perplexity', 'similarities', 'copa', 'boolq', 'multirc', 'wic', 'axg']

for prefix in prefixes:
    cols = [col for col in df.columns if prefix in col]
    df[f"mean_{prefix}"] = df[cols].mean(axis=1)
    df[f"std_{prefix}"] = df[cols].std(axis=1)
    df[f"min_{prefix}"] = df[cols].min(axis=1)
    df[f"max_{prefix}"] = df[cols].max(axis=1)

base_cols = ['model_name', 'method']
cols = base_cols + [f"std_{prefix}" for prefix in prefixes]
print(df[cols].to_string(index=False))
cols = base_cols + [f"mean_{prefix}" for prefix in prefixes]
print(df[cols].to_string(index=False))
cols = base_cols + [f"max_{prefix}" for prefix in prefixes]
print(df[cols].to_string(index=False))

df.to_csv(path_out, index=False)