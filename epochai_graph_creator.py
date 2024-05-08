import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np

file = 'epochai_all_systems.csv'
df = pd.read_csv(file)

df.sort_values(by='Publication date', inplace=True)
df.reset_index(drop=True, inplace=True)

# drop where Parameters are null
df.dropna(subset=['Parameters'], inplace=True)
df.dropna(subset=['Publication date'], inplace=True)
df.reset_index(drop=True, inplace=True)

names = df['System']
dates = df['Publication date']
sizes = df['Parameters']

# Convert dates to datetime objects
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

# Convert sizes to integers
sizes = [s for s in sizes]

# Calculate polinomial regression
x = np.array([i.toordinal() for i in dates])
y = np.array(sizes)
y = np.log(y)

z = np.polyfit(x, y, 3)
p = np.poly1d(z)
y_pred = np.exp(p(x))

# Draw high resolution plot for year vs size

plt.figure(figsize=(12, 8), dpi=500)
plt.scatter(dates, sizes, color='green')
plt.plot(dates, y_pred, color='red')

# draw grid
plt.grid(True)

# set x axis to linear and y to logarithmic
plt.yscale('log')
# plt.xscale('linear')

# set labels and title
plt.xlabel('Gads')
plt.ylabel('Parametru skaits')

plt.tight_layout()
plt.show()