# Ex.No:10 -- Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Start by importing the required libraries (pandas, matplotlib.pyplot, KMeans from sklearn.cluster).

2.Load the Mall_Customers.csv dataset into a DataFrame.

3.Check for missing values in the dataset to ensure data quality.

4.Select the features Annual Income (k$) and Spending Score (1-100) for clustering.

5.Use the Elbow Method by running KMeans for cluster counts from 1 to 10 and record the Within-Cluster Sum of Squares (WCSS).

6.Plot the WCSS values against the number of clusters to determine the optimal number of clusters (elbow point).

7.Fit the KMeans model to the selected features using the chosen number of clusters (e.g., 5).

8.Predict the cluster label for each data point and assign it to a new column called cluster.

9.Split the dataset into separate clusters based on the predicted labels.

10.Visualize the clusters using a scatter plot, and optionally mark the cluster centroids.


## Program:
~~~
Developed by: RAJKUMAR g
RegisterNumber:  212223230166
~~~
~~~
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]
print("Name:A.Lahari")
print("Reg.No:212223230111")

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of. clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
~~~



## Output:

## data.head()
![image](https://github.com/user-attachments/assets/711b78af-195f-446d-90ae-b4e0a9680807)

## data.info()
![image](https://github.com/user-attachments/assets/52e20cff-585f-48b6-8537-d2e7087ea1e0)

## data.isnull().sum()
![image](https://github.com/user-attachments/assets/c90992fa-e76f-491c-9cd8-a750d1af4eef)

## ELbow method
![image](https://github.com/user-attachments/assets/f38c13cd-ae58-4d51-9387-b9e1bf5ca0c8)

## KMeans
![image](https://github.com/user-attachments/assets/3e84ad70-b0e7-4ed8-916f-a773ecd4997f)

## y_pred
![image](https://github.com/user-attachments/assets/d380d7af-9035-4f49-ba39-317c4200f96d)

## Customer Segments
![image](https://github.com/user-attachments/assets/c5b54933-a895-4bb9-b81d-d75ae6532ff7)







## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
