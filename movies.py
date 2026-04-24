# Section: Data Collection and Cleaning --- Bobby

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from dateutil import parser


m = pd.read_csv("C:\\Users\\Zach\\Documents\\FINALPROJECTML\\IntroMLFinalProject\\movie.csv")

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


X = newMovieData.drop(["title", "rating"], axis = 1) # remove label and target variable from X
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

def train_test_split(X, y, test_size = 0.2, random_state = 42):  # Source Notebook 3: 1.3, direct implementation of given code
    np.random.seed(random_state)

    N = len(y)
    n_test = int(N * test_size)

    indices = np.random.permutation(N)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f"Train: {len(y_train)}, Test: {len(y_test)}")



