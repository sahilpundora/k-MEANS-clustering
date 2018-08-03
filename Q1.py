from matplotlib import pyplot as plt
import csv
import math
import numpy as np


#TO READ THE FILE AND CREATE DATASET-----------------------------------------
f= open('C:\\Users\\sahil\\Desktop\\A3_Q1_dataset.csv','rb') #mention the location where the dataset is located
readfile= csv.reader(f)
feat1=[]
feat2=[]
data=[]

for row in readfile:
    feat1=(float(row[0]))
    feat2=(float(row[1]))
    x=[feat1,feat2]
    data.append(x)

data=np.array(data)
# print data
print 'Total Rows in the data are...', len(data)
print 'Data has not been normalized because we do not know what the two Columns mean \n Neither do we know their ranges...'

'''''
#WE COULD EASILY NORMALIZE THE DATA BY UNCOMMENT THE FOLLOWING BLOCK
#NOT NECESSARY IN THIS DATASET

print 'Normalizing the Dataset.....'
#Find min and max values amongst all the columns
minvals={}
maxvals={}
for j in range(0,2):
	minD=data[0][j]
	maxD=data[0][j]
	for i in range(0,len(data)):
		if data[i][j] < minD:
			minD=data[i][j]
		if data[i][j] > maxD:
			maxD=data[i][j]
		minvals[j]=minD
		maxvals[j]=maxD


#Normalize all the values using Min and Max calculate in the previous step
#make a new table normaldata, which is a copy of the original dataset fulltable but with one more condition
#It is not the exact original, but normalized values of original data
normaldata = data.copy()
for j in range(0,2):
	for i in range(0, len(data)):
		 normaldata[i][j]= (normaldata[i][j] - minvals[j]) / (maxvals[j]-minvals[j])
print 'Normalizing Successful.....'

# print normaldata
'''



#CALCULATE AND PLOT THE DENDROGRAM----------------------------------------------
#perform hierarchial clustering
from scipy.cluster.hierarchy import dendrogram, linkage
#save the clustering results in variable z
Z = linkage(data, method='ward', metric='euclidean')
#create a dendrogram from the result of the hierarchial clustering
dendrogram(Z, leaf_rotation = 90, leaf_font_size = 8) #rotate the x axis labels, set the font size for x axis labels
#add title and axis labels
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.axhline(y=60, color='black')
plt.savefig('Q1_Dendrogram with Cutting')
plt.show()

print 'From the Dendrogram, threshold at 60 seems to be a good-cutting point that divides dataset into THREE clusters. \n We will now proceed with 3-Means clustering'

#USE 3-MEANS CLUSTERING
#initialize cluster centers
print 'Following has been chosen as the intial centroids for the 3 clusters-'
print 'C1(9,8), C2(3,4), C3(6,8)'
c1=[9.0, 8.0]
c2=[3.0, 4.0]
c3=[6.0, 8.0]


prev1 = []
prev2 = []
prev3 = []
cluster1 = []
cluster2 = []
cluster3 = []

#to calculate EUCLIDEAN distance
def distance(a,b,c):
    d = math.sqrt((a - c[0]) ** 2 + (b - c[1]) ** 2)
    return d

#to calculate Cluster Mean
def clusterMean(cluster):
    a = 0
    b = 0
    for instance in cluster:
        a += instance[0]
        b += instance[1]
    sums = [a, b]
    if len(cluster)!=0:
        mean = [float(Sum)/len(cluster) for Sum in sums]
    else:
        mean = 0
    return mean

t = 0 #to check if the clusters have stopped changing
p=0 #to mention the number of max iterations
while p<100 and t<1:
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range (0, len(data)):
        d1 = distance(data[i][0], data[i][1], c1)
        d2 = distance(data[i][0], data[i][1], c2)
        d3 = distance(data[i][0], data[i][1], c3)
        if d1<d2 and d1<d3:
            cluster1.append(data[i])
        elif d2<d3 and d2<d1:
            cluster2.append(data[i])
        else:
            cluster3.append(data[i])
    # print 'cluster1',cluster1
    # print 'cluster2',cluster2
    # print 'cluster3',cluster3

    cluster1=np.array(cluster1)
    cluster2=np.array(cluster2)
    cluster3=np.array(cluster3)

    #to check if the points to the associated clusters have stopped being reassigned
    if np.array_equal(cluster1, prev1) and np.array_equal(cluster2, prev2) and np.array_equal(cluster3, prev3):
        t += 1

    #to compute cluster mean
    c1 = clusterMean(cluster1)
    c2 = clusterMean(cluster2)
    c3 = clusterMean(cluster3)
    prev1 = cluster1
    prev2 = cluster2
    prev3 = cluster3

    centroids=[c1,c2,c3]
    centroids=np.array(centroids)

    #plot it out
    #I have not plotted only the 0,5,10,100 because the loop only takes about 8 iterations
    #all the plots (very limited (only 8)) have been put in the Homework pdf to give more clarity
    #following could be uncommented if to plot only the above
    # if p in (0,5,10,100):
    if p in (0,1,2,3,4,5,6,7,8,9,10):
        plt.figure(p)
        plt.scatter(cluster1[:,0], cluster1[:,1], c='b', marker='D')
        plt.scatter(cluster2[:,0], cluster2[:,1], c='g', marker='>')
        plt.scatter(cluster3[:,0], cluster3[:,1], c='y', marker='+')
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='r', marker='o')
        plt.savefig('my_file_name %i' % p)
        plt.show()
        p+=1
