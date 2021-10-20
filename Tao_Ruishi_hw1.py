#Ruishi Tao
#ITP499 Fall2021
#HW1
import pandas as pd
#Read the dataset into a dataframe. Be sure to import the header.
pd.set_option('display.max_columns',None)
df = pd.read_csv('wineQualityReds.csv',header=0)
# Drop Wine from the dataframe
df.drop(df.columns[0],axis=1,inplace=True)
print(df)
# Extract Quality and store it in a separate variable
dfQuality= df['quality']
print(dfQuality)
# Drop Quality from dataframe.
df1=df.drop(columns=['quality'])
# Print the dataframe and Quality.
print(df1)
print(dfQuality)
#Normalize all columns of the dataframe. Use th(e Normalizer class from sklearn.preprocessing.

from sklearn.preprocessing import Normalizer
norm = Normalizer()
df_norm=pd.DataFrame(norm.transform(df),columns=df.columns)
#df_norm=(df-df.min())/(df.max()-df.min())
# Print the normalized dataframe
print(df_norm)
#Create a range of k values from 1:11 for KMeans clustering. Iterate on the k values and store the inertia for each clustering in a list.
from sklearn.cluster import KMeans
ks = range(1,11)
inertias= []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_norm)
    inertias.append(model.inertia_)
# Plot the chart of inertia vs number of clusters.
import matplotlib.pyplot as plt
plt.plot(ks,inertias,"-o")
plt.xlabel("Number of Clusters,k")
plt.ylabel("Inertia")
plt.xticks(ks)
plt.show()
# What K (number of clusters) would you pick for KMeans? (1)
# I would pick K=6
# Now cluster the wines into K clusters. Use random_state = 2021 when you instantiate the KMeans model. Assign the respective cluster number to each wine. Print the dataframe showing the cluster number for each wine. (2)
model = KMeans(n_clusters=6, random_state=2021)
model.fit(df_norm)
labels = model.predict(df_norm)
df_norm["Cluster Label"]=pd.Series(labels)
print(df_norm)
#Add the quality back to the dataframe. (1)
df_norm['quality'] = dfQuality
print(df_norm)
#Now print a crosstab (from Pandas) of cluster number vs quality. Comment if the clusters represent the quality of wine. (3)
print(pd.crosstab(df_norm['quality'],df_norm['Cluster Label'],rownames=['quality'],colnames=['cluster']))