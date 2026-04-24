import pandas as pd
import numpy as np
import hitting_sets
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from dateutil import parser


m = pd.read_csv("C:\\Users\\rfd\\OneDrive\Desktop\Computing for Global Challenges\Independent Work\\Movies\\movie.csv")

# ORIGINAL instances (movies): 636 
# ORIGINIAL attributes (information about movies): 11

# MODIFIED instances (movies): 634
# MODIFIED attributes (information about movies): 8



#PREP
newMovieData = pd.DataFrame()

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

newMovieData["rating"] = m["rating"]

newMovieData["rating_count"] = m["rating_count"]

# Applying the NaN checker 
#count_nan = newMovieData['release_date'].isnull().sum()
# printing the number of values present
# in the column
#print('Number of NaN values present: ' + str(count_nan))

# drop data with Nan
newMovieData.dropna(how="any", inplace=True)
#print(len(newMovieData))

# FIT
# fit data to the k-means clustering algorithm (Lloyd's or Elkan's)
# use elbow method to find appropriate number of clusters (using within cluster sum of squares errors: WCSS value)
def elbow(dataSet, numClustersToConsider):
    wcss = []
    for i in range(1,numClustersToConsider+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dataSet)
        # inertia calculates the sum of the distances between each datapoint and its centroid. squaring the distance, and summing these squares across a cluster
        wcss.append(kmeans.inertia_)
    k = np.arange(1,11)
    plt.plot(k,wcss)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust numerical label size here
    plt.xlabel("Number of Clusters", fontsize=20)
    plt.ylabel("WCSS", fontsize=20)
    plt.show()

#elbow(dataSet=newMovieData.drop("title", axis=1), numClustersToConsider=10)
# four clusters

#CLUSTER
# run kmeans wih 4 clusters
numClusters = 4
kmeans = KMeans(n_clusters=numClusters)
kmeans.fit(newMovieData.drop("title", axis=1))


# POST-ANALYSIS AND INTERPRETATION
# find which individuals were clustered together
labels = kmeans.labels_
results = {}
resultsWithTitles = {}
movie_index = newMovieData.index
for i in range(numClusters):
    results["Cluster " + str(i+1)] = []
    resultsWithTitles["Cluster " + str(i+1)] = []

for i in range(len(newMovieData)):
    results["Cluster " + str(labels[i]+1)].append(newMovieData.index[i])
    resultsWithTitles["Cluster " + str(labels[i]+1)].append(newMovieData["title"][i])


# print(len(results["Cluster 1"]))
# print(len(results["Cluster 2"]))
# print(len(results["Cluster 3"]))
# print(len(results["Cluster 4"]))

# Cluster 1 has 218 individuals
# Cluster 2 has 103 individuals
# Cluster 3 has 303 individuals
# Cluster 4 has 10 individuals

# print(results["Cluster 1"])
# print(results["Cluster 2"])
# print(results["Cluster 3"])
# print(results["Cluster 4"])


# plot the clusters using PCA with two principle components
def pca():
    pca = PCA(2)
    # transform the data using the PCA object
    reduced_data = pca.fit_transform(newMovieData.drop("title", axis=1))
    label = kmeans.fit_predict(reduced_data)
    
    #filter rows of original data
    filtered_label0 = reduced_data[label == 0]
    filtered_label1 = reduced_data[label == 1]
    filtered_label2 = reduced_data[label == 2]
    filtered_label3 = reduced_data[label == 3]
 
    #Plotting the results
    plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
    plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black')
    plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'orange')
    plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'blue')
    plt.xlabel("First Principal Component", fontsize=20)
    plt.ylabel("Second Principal Component", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)  # Adjust numerical label size here
    plt.legend(["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"], fontsize=18)
    plt.show()

print(pca())


# (t1-t2) mpaa_rating for general audiences (G or PG) OR for mature audiences (PG-13 or R)
# (t3-t4) budget below OR above the median
# (t5-t6) gross below OR above the median
# (t7-t9) release date in the late 80's and 90's OR early 2000's OR 2010's and 2020's 
# (t10-t13) genre is *feel good or drama* OR *action/adventure or thriller* OR *family/sci-fi/fantasy* OR *real world adaptation* movies
# (t14-t15) runtime is below OR above the median
# (t16-t17) rating is below OR above the median
# (t18-t19) rating-count is below OR above the median

numTags = 19

# Now that we know which individuals are in each cluster, form a cluster matrix with the tags defined above

resultValues = list(results.values())
# a list of the movies in the order in which they appear in the original dataframe
movies = list(newMovieData.index)

allClusters = []
for i in range(numClusters):
    ilpMatrix = []
    # add first line of the cluster matrix
    ilpMatrix.append([len(resultValues[i]), numTags])
    for j in range(len(resultValues[i])):
        # define a tagset for an individual participant
        tagSet = []
        for k in range(numTags):
            tagSet.append(0)
        # find the tag set for this movies in the tag matrix using the original dataframe as an index reference 
        indexOfTagSet = movies.index(resultValues[i][j])
        # now find the movie at this index and develop the tagset
        m = newMovieData.iloc[indexOfTagSet]
        if m["mpaa_rating"] in [0, 1]:
            tagSet[0] = 1
        if m["mpaa_rating"] in [2, 3]:
            tagSet[1] = 1
        if m["budget"] < newMovieData["budget"].median():
            tagSet[2] = 1
        if m["budget"] >= newMovieData["budget"].median():
            tagSet[3] = 1
        if m["gross"] < newMovieData["gross"].median():
            tagSet[4] = 1
        if m["gross"] >= newMovieData["gross"].median():
            tagSet[5] = 1
        if m["release_date"] == 0:
            tagSet[6] = 1
        if m["release_date"] == 1:
            tagSet[7] = 1
        if m["release_date"] == 2:
            tagSet[8] = 1
        if m["genre"] == 0:
            tagSet[9] = 1
        if m["genre"] == 1:
            tagSet[10] = 1
        if m["genre"] == 2:
            tagSet[11] = 1
        if m["genre"] == 3:
            tagSet[12] = 1
        if m["runtime"] < newMovieData["runtime"].median():
            tagSet[13] = 1
        if m["runtime"] >= newMovieData["runtime"].median():
            tagSet[14] = 1
        if m["rating"] < newMovieData["rating"].median():
            tagSet[15] = 1
        if m["rating"] >= newMovieData["rating"].median():
            tagSet[16] = 1
        if m["rating_count"] < newMovieData["rating_count"].median():
            tagSet[17] = 1
        if m["rating_count"] <= newMovieData["rating_count"].median():
            tagSet[18] = 1

        # insert the datapoint value at the first index of the tag set 
        tagSet.insert(0, j+1)
        # append the tag set to the ilp matrix for this cluster
        ilpMatrix.append(tagSet)
    # append the ilpMatrix to the allClusters list
    allClusters.append(ilpMatrix)


# See how many of the datapoints in each cluster have tags in the hitting sets
def plotDisjunctiveHeuristic():
    pca = PCA(2)
    # transform the data using the PCA object
    reduced_data = pca.fit_transform(newMovieData.drop("title", axis=1))
    label = kmeans.fit_predict(reduced_data)

    dataInCluster1 = []
    dataInCluster2 = []
    dataInCluster3 = []
    dataInCluster4 = []

    for i in range(len(labels)):
        if labels[i] == 0:
            dataInCluster1.append(reduced_data[i])
        elif labels[i] == 1:
            dataInCluster2.append(reduced_data[i])
        elif labels[i] == 2:
            dataInCluster3.append(reduced_data[i])
        elif labels[i] == 3:
            dataInCluster4.append(reduced_data[i])

    #print(dataInCluster1)
    #print(dataInCluster2)
    #print(dataInCluster3)
    #print(dataInCluster4)

    # filter rows of original data by tags in Disjunctive Heuristic hitting set
    cluster1DH = cluster1["Disjunctive Heuristic"]
    cluster2DH = cluster2["Disjunctive Heuristic"]
    cluster3DH = cluster3["Disjunctive Heuristic"]
    cluster4DH = cluster4["Disjunctive Heuristic"]

    cluster1TagsAndDataPoints = {}
    for i in range(len(cluster1DH)):
        # find which datapoints have the tags in cluster1
        cluster1TagsAndDataPoints[str(cluster1DH[i])] = hitting_sets.hasTag(cluster1DH[i], allClusters[0])

    cluster2TagsAndDataPoints = {}
    for i in range(len(cluster2DH)):
        # find which datapoints have the tags in cluster2
        cluster2TagsAndDataPoints[str(cluster2DH[i])] = hitting_sets.hasTag(cluster2DH[i], allClusters[1])

    cluster3TagsAndDataPoints = {}
    for i in range(len(cluster3DH)):
        # find which datapoints have the tags in cluster3
        cluster3TagsAndDataPoints[str(cluster3DH[i])] = hitting_sets.hasTag(cluster3DH[i], allClusters[2])

    cluster4TagsAndDataPoints = {}
    for i in range(len(cluster4DH)):
        # find which datapoints have the tags in cluster4
        cluster4TagsAndDataPoints[str(cluster4DH[i])] = hitting_sets.hasTag(cluster4DH[i], allClusters[3])


        
    # color array for different tags in cluster1
    colors1 = np.r_[np.linspace(0.1, 1, 5), np.linspace(0.1, 1, 5)] 
    mymap1 = plt.get_cmap("ocean")
    # get the colors from the color map
    colors1 = mymap1(colors1)

    #color array for different tags in cluster2
    colors2 = np.r_[np.linspace(0.1, 1, 5), np.linspace(0.1, 1, 5)] 
    mymap2 = plt.get_cmap("Dark2_r")
    # get the colors from the color map
    colors2 = mymap2(colors2)

    #color array for different tags in cluster3
    colors3 = np.r_[np.linspace(0.1, 1, 5), np.linspace(0.1, 1, 5)] 
    mymap3 = plt.get_cmap("icefire")
    # get the colors from the color map
    colors3 = mymap3(colors3)

    #color array for different tags in cluster4
    colors4 = np.r_[np.linspace(0.1, 1, 5), np.linspace(0.1, 1, 5)] 
    mymap4 = plt.get_cmap("magma")
    # get the colors from the color map
    colors4 = mymap4(colors4)


    #preparing the results for cluster1
    resultsCluster1 = {}
    for i in range(len(cluster1DH)):
        dataWithThisTag = cluster1TagsAndDataPoints[str(cluster1DH[i])]
        # a dictionary entry with key as a tag string and value as a tuple containing a color string and a list of ints representing data in the cluster that have the key tag
        resultsCluster1[str(cluster1DH[i])] = (colors1[i], dataWithThisTag)
        
    #plot first cluster data
    tags1 = list(resultsCluster1.keys())
    colorAndDatawWithTags1 = list(resultsCluster1.values())
    #looping through each tag
    for i in range(len(resultsCluster1)):
        #looping though each datapoint in cluster1
        tagsToPlotPC1 = []
        tagsToPlotPC2 = []
        for j in range(len(dataInCluster1)):
            dataIndex = j
            if dataIndex in colorAndDatawWithTags1[i][1]:
                tagsToPlotPC1.append(dataInCluster1[j][0])
                tagsToPlotPC2.append(dataInCluster1[j][1])
        plt.scatter(tagsToPlotPC1, tagsToPlotPC2, color = colorAndDatawWithTags1[i][0], alpha=0.7)
    plt.legend(tags1)
    plt.title("PCA for Disjunctive Heuristic Hitting Set (Cluster 1)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

    
    #preparing the results for cluster2
    resultsCluster2 = {}
    for i in range(len(cluster2DH)):
        dataWithThisTag = cluster2TagsAndDataPoints[str(cluster2DH[i])]
        # a dictionary entry with key as a tag string and value as a tuple containing a color string and a list of ints representing data in the cluster that have the key tag
        resultsCluster2[str(cluster2DH[i])] = (colors2[i], dataWithThisTag)
        
    
    #plot second cluster data
    tags2 = list(resultsCluster2.keys())
    colorAndDatawWithTags2 = list(resultsCluster2.values())
    #looping through each tag
    for i in range(len(resultsCluster2)):
        #looping though each datapoint in cluster1
        tagsToPlotPC1 = []
        tagsToPlotPC2 = []
        for j in range(len(dataInCluster2)):
            dataIndex = j
            if dataIndex in colorAndDatawWithTags2[i][1]:
                tagsToPlotPC1.append(dataInCluster2[j][0])
                tagsToPlotPC2.append(dataInCluster2[j][1])
        plt.scatter(tagsToPlotPC1, tagsToPlotPC2, color = colorAndDatawWithTags2[i][0], alpha=.7)
    plt.legend(tags2)
    plt.title("PCA for Disjunctive Heuristic Hitting Set (Cluster 2)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()


    #preparing the results for cluster3
    resultsCluster3 = {}
    for i in range(len(cluster3DH)):
        dataWithThisTag = cluster3TagsAndDataPoints[str(cluster3DH[i])]
        # a dictionary entry with key as a tag string and value as a tuple containing a color string and a list of ints representing data in the cluster that have the key tag
        resultsCluster3[str(cluster3DH[i])] = (colors3[i], dataWithThisTag)
        
    #plot third cluster data
    tags3 = list(resultsCluster3.keys())
    colorAndDatawWithTags3 = list(resultsCluster3.values())
    #looping through each tag
    for i in range(len(resultsCluster3)):
        #looping though each datapoint in cluster1
        tagsToPlotPC1 = []
        tagsToPlotPC2 = []
        for j in range(len(dataInCluster3)):
            dataIndex = j
            if dataIndex in colorAndDatawWithTags3[i][1]:
                tagsToPlotPC1.append(dataInCluster3[j][0])
                tagsToPlotPC2.append(dataInCluster3[j][1])
        plt.scatter(tagsToPlotPC1, tagsToPlotPC2, color = colorAndDatawWithTags3[i][0], alpha=0.7)
    plt.legend(tags3)
    plt.title("PCA for Disjunctive Heuristic Hitting Set (Cluster 3)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

    #preparing the results for cluster4
    resultsCluster4 = {}
    for i in range(len(cluster4DH)):
        dataWithThisTag = cluster4TagsAndDataPoints[str(cluster4DH[i])]
        # a dictionary entry with key as a tag string and value as a tuple containing a color string and a list of ints representing data in the cluster that have the key tag
        resultsCluster4[str(cluster4DH[i])] = (colors4[i], dataWithThisTag)
        
    #plot fourth cluster data
    tags4 = list(resultsCluster4.keys())
    colorAndDatawWithTags4 = list(resultsCluster4.values())
    #looping through each tag
    for i in range(len(resultsCluster4)):
        #looping though each datapoint in cluster1
        tagsToPlotPC1 = []
        tagsToPlotPC2 = []
        for j in range(len(dataInCluster4)):
            dataIndex = j
            if dataIndex in colorAndDatawWithTags4[i][1]:
                tagsToPlotPC1.append(dataInCluster4[j][0])
                tagsToPlotPC2.append(dataInCluster4[j][1])
        plt.scatter(tagsToPlotPC1, tagsToPlotPC2, color = colorAndDatawWithTags4[i][0], alpha=0.7)


    plt.legend(tags4)
    plt.title("PCA for Disjunctive Heuristic Hitting Set (Cluster 4)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()


# See which movies were placed into each cluster
clustersWithTitles = list(resultsWithTitles.values())
for i in range(len(clustersWithTitles)):
    print("Cluster " + str(i+1) + ": " + str(clustersWithTitles[i]))
    print()
    print()
print()



# use four hitting set algorithms to find descriptors for the clusters
cluster1 = {}
cluster1["Disjunctive Heuristic"] = hitting_sets.solveDisjunctive(allClusters[0])
cluster1["Disjunctive Exact"] = hitting_sets.hitting_set_ilp(allClusters[0])
cluster1["CNF"] = hitting_sets.solveCNF(allClusters[0])
cluster1["DNF"] = hitting_sets.solveDNF(allClusters[0])
cluster1["Tag Percentages"] = hitting_sets.tagPercentages(allClusters[0])

cluster2 = {}
cluster2["Disjunctive Heuristic"] = hitting_sets.solveDisjunctive(allClusters[1])
cluster2["Disjunctive Exact"] = hitting_sets.hitting_set_ilp(allClusters[1])
cluster2["CNF"] = hitting_sets.solveCNF(allClusters[1])
cluster2["DNF"] = hitting_sets.solveDNF(allClusters[1])
cluster2["Tag Percentages"] = hitting_sets.tagPercentages(allClusters[1])

cluster3 = {}
cluster3["Disjunctive Heuristic"] = hitting_sets.solveDisjunctive(allClusters[2])
cluster3["Disjunctive Exact"] = hitting_sets.hitting_set_ilp(allClusters[2])
cluster3["CNF"] = hitting_sets.solveCNF(allClusters[2])
cluster3["DNF"] = hitting_sets.solveDNF(allClusters[2])
cluster3["Tag Percentages"] = hitting_sets.tagPercentages(allClusters[2])

cluster4 = {}
cluster4["Disjunctive Heuristic"] = hitting_sets.solveDisjunctive(allClusters[3])
cluster4["Disjunctive Exact"] = hitting_sets.hitting_set_ilp(allClusters[3])
cluster4["CNF"] = hitting_sets.solveCNF(allClusters[3])
cluster4["DNF"] = hitting_sets.solveDNF(allClusters[3])
cluster4["Tag Percentages"] = hitting_sets.tagPercentages(allClusters[3])

clusters = [cluster1, cluster2, cluster3, cluster4]



for i in range(len(clusters)):
    print("CLUSTER " + str(i+1))
    print("size = " + str(len(allClusters[i])))
    print("Disjunctive Heuristic: " + str(clusters[i]["Disjunctive Heuristic"]))
    print("Disjunctive Exact: " + str(clusters[i]["Disjunctive Exact"]))
    print("CNF: " + str(clusters[i]["CNF"]))
    print("DNF: " + str(clusters[i]["DNF"]))
    print("Tag Percentages: " + str(clusters[i]["Tag Percentages"]))
    print()
    print()



#make unique hitting sets for cluster 3 and cluster 4 by applying filter
dhuC1C2 = hitting_sets.solveDisjunctiveUniqueHittingSets(allClusters[0], allClusters[1], maxSharedPercentage=50.00)
dhuC3C4 = hitting_sets.solveDisjunctiveUniqueHittingSets(allClusters[2], allClusters[3], maxSharedPercentage=50.00)
print("Clusters 1 and 2 Disjunctive Heuristics with Tags Considered for Hitting Set Generation Appearing in AT MOST 50% of both Clusters")
print("CLUSTER 1 Filtered Disjunctive Heuristic: " + str(dhuC1C2[0]))
print("CLUSTER 2 Filtered Disjunctive Heuristic: " + str(dhuC1C2[1]))
print()
print("Clusters 3 and 4 Disjunctive Heuristic with Tags Considered for Hitting Set Generation Appearing in AT MOST 50% of both Clusters")
print("CLUSTER 3 Filtered Disjunctive Heuristic: " + str(dhuC3C4[0]))
print("CLUSTER 4 Filtered Disjunctive Heuristic: " + str(dhuC3C4[1]))


#print(plotDisjunctiveHeuristic())