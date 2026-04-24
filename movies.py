# Section: Data Collection and Cleaning --- Bobby

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from dateutil import parser


m = pd.read_csv("C:\\Users\\rfd\\UVASemester8\\ECE2410\\Project\\IntroMLFinalProject\\movie.csv")

# ORIGINAL instances (movies): 636 
# ORIGINIAL attributes (information about movies): 11

# MODIFIED instances (movies): 634
# MODIFIED attributes (information about movies): 8



#PREP
newMovieData = pd.DataFrame()

# KEY FEATURES

mpaa_rating = []
for r in m.iloc[:,2]:
    if r.strip() == "G":
        mpaa_rating.append(0)
    elif r.strip() == "PG":
        mpaa_rating.append(1)
    elif r.strip() == "PG-13":
        mpaa_rating.append(2)
    elif r.strip() == "R":
        mpaa_rating.append(3)

newMovieData["title"] = m["title"]

newMovieData["mpaa_rating"] = pd.Series(mpaa_rating)

newMovieData["budget"] = m["budget"]

newMovieData["gross"] = m["gross"]

release_dates = []
for rd in m.iloc[:,5]:
    #date = parser.parse(rd)
    #release_dates.append(date)
    if int(rd[-2]) in [8,9]:
        # *late-80's and 90's* movies
        release_dates.append(0)
    elif int(rd[-4:-1]) == 200:
        # *early-2000's* movies
        release_dates.append(1)
    elif int(rd[-4:-1]) in [201, 202]:
        # *2010's and early 2020's* movies
        release_dates.append(2)
newMovieData["release_date"] = pd.Series(release_dates)

genres = []
for g in m.iloc[:,6]:
    if g.strip() in ["Romantic Comedy", "Comedy, Romance, Music", "Musical", "Drama", "Romance", "Comedy"]:
        # *feel good or drama* movies
        genres.append(0)
    elif g.strip() in ["Adventure", "Thriller", "Action", "Horror", "Western"]:
        # *action/adventure or thriller* movies
        genres.append(1)
    elif g.strip() in ["Family", "Animation", "Science Fiction", "Fantasy"]:
        # *family/sci-fi/fantasy* movies
        genres.append(2)
    elif g.strip() in ["Crime", "War", "Mystery", "History"]:
        # *real world adaptation* movies
        genres.append(3)
newMovieData["genre"] = pd.Series(genres)

newMovieData["runtime"] = m["runtime"]

newMovieData["rating_count"] = m["rating_count"]

# TARGET VARIABLE

newMovieData["rating"] = m["rating"]


X = newMovieData.drop(["title", "rating"]) # remove label and target variable from X
y = newMovieData["rating"]


# Applying the NaN checker 
#count_nan = newMovieData['release_date'].isnull().sum()
# printing the number of values present
# in the column
#print('Number of NaN values present: ' + str(count_nan))


# drop data with Nan
newMovieData.dropna(how="any", inplace=True)
print(len(newMovieData)) # 634 movies


# Section: Method 1 Implementation (Feature Scaling) --- Bobby

from sklearn.model_selection import train_test_split

























# # FIT
# # fit data to the k-means clustering algorithm (Lloyd's or Elkan's)
# # use elbow method to find appropriate number of clusters (using within cluster sum of squares errors: WCSS value)
# def elbow(dataSet, numClustersToConsider):
#     wcss = []
#     for i in range(1,numClustersToConsider+1):
#         kmeans = KMeans(n_clusters=i)
#         kmeans.fit(dataSet)
#         # inertia calculates the sum of the distances between each datapoint and its centroid. squaring the distance, and summing these squares across a cluster
#         wcss.append(kmeans.inertia_)
#     k = np.arange(1,11)
#     plt.plot(k,wcss)
#     plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust numerical label size here
#     plt.xlabel("Number of Clusters", fontsize=20)
#     plt.ylabel("WCSS", fontsize=20)
#     plt.show()

# #elbow(dataSet=newMovieData.drop("title", axis=1), numClustersToConsider=10)
# # four clusters

# #CLUSTER
# # run kmeans wih 4 clusters
# numClusters = 4
# kmeans = KMeans(n_clusters=numClusters)
# kmeans.fit(newMovieData.drop("title", axis=1))


# # POST-ANALYSIS AND INTERPRETATION
# # find which individuals were clustered together
# labels = kmeans.labels_
# results = {}
# resultsWithTitles = {}
# movie_index = newMovieData.index
# for i in range(numClusters):
#     results["Cluster " + str(i+1)] = []
#     resultsWithTitles["Cluster " + str(i+1)] = []

# for i in range(len(newMovieData)):
#     results["Cluster " + str(labels[i]+1)].append(newMovieData.index[i])
#     resultsWithTitles["Cluster " + str(labels[i]+1)].append(newMovieData["title"][i])


# # print(len(results["Cluster 1"]))
# # print(len(results["Cluster 2"]))
# # print(len(results["Cluster 3"]))
# # print(len(results["Cluster 4"]))

# # Cluster 1 has 218 individuals
# # Cluster 2 has 103 individuals
# # Cluster 3 has 303 individuals
# # Cluster 4 has 10 individuals

# # print(results["Cluster 1"])
# # print(results["Cluster 2"])
# # print(results["Cluster 3"])
# # print(results["Cluster 4"])


# # plot the clusters using PCA with two principle components
# def pca():
#     pca = PCA(2)
#     # transform the data using the PCA object
#     reduced_data = pca.fit_transform(newMovieData.drop("title", axis=1))
#     label = kmeans.fit_predict(reduced_data)
    
#     #filter rows of original data
#     filtered_label0 = reduced_data[label == 0]
#     filtered_label1 = reduced_data[label == 1]
#     filtered_label2 = reduced_data[label == 2]
#     filtered_label3 = reduced_data[label == 3]
 
#     #Plotting the results
#     plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
#     plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black')
#     plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'orange')
#     plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'blue')
#     plt.xlabel("First Principal Component", fontsize=20)
#     plt.ylabel("Second Principal Component", fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust numerical label size here
#     plt.legend(["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"], fontsize=18)
#     plt.show()

# print(pca())

