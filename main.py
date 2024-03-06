import pandas as pd
import functions as fun
import pca.PCA as pca
import graphics as g
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Loading
table = pd.read_csv('./dataIN/dataset.csv', index_col=0)
vars = table.columns.values[1:]
obs = table.index.values
X = table[vars].values

# Standardizing
EstandardAcum = fun.standardize(X)
EstandardAcum_df = pd.DataFrame(data=EstandardAcum, index=obs, columns=vars)
EstandardAcum_df.to_csv('./dataOUT/EstandardAcum.csv')

# PCA model
pcaModel = pca.PCA(EstandardAcum)

# Extracting and printing the eigenvalues
alpha = pcaModel.getAlpha()
print("Eigenvalues:")
print(alpha)

# Explained variance by principal components
g.principalComponents(eigenvalues=alpha)
g.show()


# Extracting principal components
prinComp = pcaModel.getPrinComp()
components = ['C'+str(j+1) for j in range(prinComp.shape[1])]
prinComp_df = pd.DataFrame(data=prinComp, index=obs, columns=components)
prinComp_df.to_csv('./dataOUT/PrincComp.csv')

# Extracting the factor loadings
factorLoadings = pcaModel.getFactorLoadings()
factorLoadings_df = pd.DataFrame(data=factorLoadings, index=vars, columns=components)

# Creating the correlogram of factor loadings
g.correlogram(matrix=factorLoadings_df, title='Correlogram of factor loadings')
g.show()


# Extracting and visualizing scores
scores = pcaModel.getScores()
scores_df = pd.DataFrame(data=scores, index=obs, columns=components)
g.intensity_map(matrix=scores_df, title='Standardized principal components of scores')
g.show()

# Quality of point representations
qualObs = pcaModel.getQualObs()
qualObs_df = pd.DataFrame(data=qualObs, index=obs, columns=components)
g.intensity_map(matrix=qualObs_df, title='Quality of points representation')
g.show()


# Contribution of observations
contribObs = pcaModel.getContribObs()
contribObs_df = pd.DataFrame(data=contribObs, index=obs, columns=components)
g.intensity_map(matrix=contribObs_df, title="Contribution of observations to the axes' variance")
g.show()

# Communalities
commun = pcaModel.getCommun()
commun_df = pd.DataFrame(data=commun, index=vars, columns=components)
g.intensity_map(matrix=commun_df, title='Communalities')
g.show()

# Cluster Analysis
columns_for_clustering = ['PIB', 'Natalitate', 'Spor Natural', 'Mortalitate', 'Nuptialitate', 'Populatie(>500000)']
data_for_clustering = table[columns_for_clustering]

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_for_clustering)

# Optimal number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_standardized)
    inertia.append(kmeans.inertia_)


# Ploting the elbow method
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Cluster Number')
plt.show()


# Optimal no of clusterss
optimal_clusters = 3

# K-means clustering
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_standardized)

# Cluster labels for the original dataset
table['Cluster'] = clusters

# Result
print(table[['Cluster']])


