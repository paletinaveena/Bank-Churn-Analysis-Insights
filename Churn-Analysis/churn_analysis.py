# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('data/Churn_Modelling.csv')

# Display summary statistics
print("Summary Statistics of the Dataset:")
print(df.describe())

# Select relevant features for PCA
features_for_pca = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features_for_pca])

# Perform PCA to obtain principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df['PC1'] = principal_components[:, 0]
df['PC2'] = principal_components[:, 1]

# Perform K-Means clustering on the principal components
kmeans = KMeans(n_clusters=2, random_state=0)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='Dark2', s=100)
plt.title('Clustering Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', loc='upper right')
plt.show()

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 35, 65, np.inf], labels=['Young', 'Middle-Aged', 'Old'])

# Create a faceted bar plot
g = sns.catplot(x='AgeGroup', hue='Exited', col='AgeGroup', data=df, kind='count', palette='Set1')
g.set_axis_labels('Age Group', 'Count')
g.set_titles('{col_name}')

# Create a stacked bar plot for gender vs. churn
gender_churn = df.groupby(['Geography', 'Gender', 'Exited']).size().unstack().fillna(0)
gender_churn.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Gender vs. Churn')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.legend(title='Churn', labels=['Retained', 'Churned'])
plt.show()

# Create a new categorical variable for zero balance
df['ZeroBalance'] = df['Balance'].apply(lambda x: 'Zero' if x == 0 else 'Nonzero')

# Create a mosaic plot
from statsmodels.graphics.mosaicplot import mosaic
mosaic(df, ['ZeroBalance', 'Exited'], title='Account Balance vs. Churn')
plt.show()
