# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1- Import necessary libraries 
2- Load the customer dataset 
3- Select relevant features for clustering 
4-Handle missing or invalid data 
5- Scale or normalize the features 
6-Determine optimal number of clusters (K) 
7- Initialize and fit the K-Means model 
8-Assign cluster labels to each customer 
9- Visualize the clusters 
10- Interpret and profile
11-apply segmention insights to business strategy
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:A.B.Gnana pragathika 
RegisterNumber:25018821  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
# Step 2: Suppress warnings
warnings.filterwarnings("ignore")
# Step 3: Create a synthetic customer dataset
data = {
    'CustomerID': range(1, 21),
    'AnnualIncome': [15, 16, 17, 18, 19, 20, 60, 62, 64, 65, 66, 67,
                     120, 122, 124, 125, 126, 127, 128, 130],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99,
                      15, 77, 13, 79, 35, 66, 29, 98]
}
df = pd.DataFrame(data)
# Step 4: Select features and scale them
X = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 5: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# Step 6: Visualize the clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(
        df[df['Cluster'] == i]['AnnualIncome'],
        df[df['Cluster'] == i]['SpendingScore'],
        label=f'Cluster {i}',
        color=colors[i]
    )
# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=200,
    c='yellow',
    label='Centroids',
    marker='X'
)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
Thus the program to implement the K Means Clustering for Customer Segmentation is
written and verified using python programming.
plt.legend()
plt.grid(True)
plt.show()

## Output:
![K Means Clustering for Customer Segmentation](sam.png)
<img width="626" height="449" alt="image" src="https://github.com/user-attachments/assets/5f1e614d-7e4a-47de-8050-6269e3317d3d" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
