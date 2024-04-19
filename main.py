# This is a sample Python script.
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


data = pd.read_csv('Results_21Mar2022.csv', index_col=False)
data.drop(data.columns[[0]], axis=1, inplace=True)

plot = sns.catplot(x="diet_group", y="mean_ghgs", hue="sex", kind="box", data=data, height=6.5, aspect=1.2)
# Rotate the x-axis labels for better readability
plot.set_xticklabels(rotation=45)
# Adjust layout for a better fit
plot.fig.tight_layout()


age_mapping = {
    '20-29': 25,
    '30-39': 35,
    '40-49': 45,
    '50-59': 55,
    '60-69': 65,
    '70-79': 75
}
data['fixed_median_age'] = data['age_group'].map(age_mapping)
df_filtered = data.loc[:, data.columns.str.startswith('mean') | data.columns.isin(['diet_group', 'sex'])]

df_filtered['fixed_median_age'] = data['fixed_median_age']

# Apply One-Hot Encoding to 'diet_group' and 'sex'
ohe = OneHotEncoder(sparse=False)
encoded_features = ohe.fit_transform(df_filtered[['diet_group', 'sex']])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(['diet_group', 'sex']))

# Drop 'diet_group' and 'sex' from the original dataframe and concatenate with the encoded features
df_filtered.drop(['diet_group', 'sex'], axis=1, inplace=True)
df_prepared = pd.concat([df_filtered, encoded_df], axis=1)

# Now we are ready to plot the heatmap of all variables
mask = np.triu(np.ones_like(df_prepared.corr(), dtype=bool))
plt.figure(figsize=(14, 10))
sns.heatmap(df_prepared.corr(),mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heatmap of Variables")
plt.xticks(rotation=30, ha="right", fontsize = 7)
plt.yticks(rotation=0, fontsize = 7)
plt.tight_layout()


#scatter plot of acid and ghgs
features = data[['mean_land', 'mean_watuse', 'mean_ghgs', 'mean_acid']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=['mean_land', 'mean_watuse', 'mean_ghgs','mean_acid'])

plt.figure(figsize=(10, 6), facecolor='lightgray')
plt.subplot(1, 2, 1)
plt.scatter(features_scaled_df['mean_ghgs'], features_scaled_df['mean_acid'], color='cyan', label='Data points')


coefficients_linear = np.polyfit(features_scaled_df['mean_ghgs'], features_scaled_df['mean_acid'], 1)
polynomial_linear = np.poly1d(coefficients_linear)
xs_linear = np.linspace(features_scaled_df['mean_ghgs'].min(), features_scaled_df['mean_ghgs'].max())
ys_linear = polynomial_linear(xs_linear)
plt.plot(xs_linear, ys_linear, color='red', linestyle='--', label='Linear fit line')

coefficients_quad = np.polyfit(features_scaled_df['mean_ghgs'], features_scaled_df['mean_acid'], 2)
polynomial_quad = np.poly1d(coefficients_quad)
xs_quad = np.linspace(features_scaled_df['mean_ghgs'].min(), features_scaled_df['mean_ghgs'].max())
ys_quad = polynomial_quad(xs_quad)
plt.plot(xs_quad, ys_quad, color='green', linestyle='--', label='Quadratic fit line')


plt.annotate('Linear fit', xy=(xs_linear[-5], ys_linear[-5]), xytext=(10, -10),
             textcoords="offset points", ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.5", fc="w"),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="red"))

# 注释二次拟合
plt.annotate('Quadratic fit', xy=(xs_quad[-5], ys_quad[-5]), xytext=(20, -10),
             textcoords="offset points", ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.5", fc="w"),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color="green"))

plt.xlabel('mean_ghgs')
plt.ylabel('mean_acid')


#fit plot of water use, land use and ghgs


plt.subplot(1, 2, 2, projection='3d')
X = features_scaled_df[['mean_land', 'mean_watuse']]
y = features_scaled_df['mean_ghgs']
model = LinearRegression().fit(X, y)
coefficients = model.coef_
intercept = model.intercept_

# 为了绘制拟合平面，生成网格数据
x = np.linspace(features_scaled_df['mean_land'].min(), features_scaled_df['mean_land'].max(), num=10)
y = np.linspace(features_scaled_df['mean_watuse'].min(), features_scaled_df['mean_watuse'].max(), num=10)
x, y = np.meshgrid(x, y)
z = coefficients[0] * x + coefficients[1] * y + intercept

ax = plt.gca()
ax.plot_surface(x, y, z, color='r', alpha=0.5)
ax.set_xlabel('Mean Land')
ax.set_ylabel('Mean Water Use')
ax.set_zlabel('Mean GHGs')

ax.scatter(features_scaled_df['mean_land'], features_scaled_df['mean_watuse'], features_scaled_df['mean_ghgs'], color='b')

plt.show()