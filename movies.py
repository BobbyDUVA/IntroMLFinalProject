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


newMovieData.dropna(how="any", inplace=True)
print(len(newMovieData)) # 634 movies


X = newMovieData.drop(["title", "rating"], axis = 1).to_numpy() # remove label and target variable from X
y = newMovieData["rating"].to_numpy()


# Section: Method 1 Implementation (Feature Scaling) --- Bobby

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

def normalize_zscore(X_train, X_test): #Source Notebook 3: 1.4, direct implementation of given code again
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm

X_train_norm, X_test_norm = normalize_zscore(X_train, X_test)
print(f"Normalized mean: {np.mean(X_train_norm, axis=0).round(4)}")
print(f"Normalized std:  {np.std(X_train_norm, axis=0).round(4)}")

def knn_predict_regression(X_train, y_train, X_test, k=3): #Source Notebook 3: 1.5
    #kNN Pred using kNN - Regression, almost identical to what was given in class, just minor changes that are explained
    # https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbors-knn-regression-with-scikit-learn/
    predictions = []
    
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        
        pred = np.average(k_nearest_labels)
        # I just averaged instead of doing a count comparison, otherwise the code is identical to that provided in the notebook.
        predictions.append(pred)
    
    return np.array(predictions)

def compute_metrics(y_true, y_pred): #Idea taken directly from Notebook 3 again, section 1.6
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    #I just added RMSE for a comparison since you know it's, 
    #regression instead of classification this time, different ball game.
    #I would add accuracy, recall, precision, but as you are well aware that only works 
    #for classification problems.
    return rmse

y_pred = knn_predict_regression(X_train, y_train, X_test, k=5)
rmse_non_normalized = compute_metrics(y_test, y_pred)
print("RMSE NON NORMALIZED: " + str(rmse_non_normalized))

y_pred_norm = knn_predict_regression(X_train_norm, y_train, X_test_norm, k=5)
rmse_normalized = compute_metrics(y_test, y_pred_norm)
print("RMSE NORMALIZED: " + str(rmse_normalized))

from sklearn.linear_model import LinearRegression

def linear_regression_predict(X_train, y_train, X_test): #Source Notebook 8: Part 3
    #Linear regression model, copied almost directly from the notebook
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred


y_pred_linear = linear_regression_predict(X_train, y_train, X_test)
rmse_linear_non_normalized = compute_metrics(y_test, y_pred_linear)
print("LINEAR REGRESSION RMSE NON NORMALIZED: " + str(rmse_linear_non_normalized))


y_pred_linear_norm = linear_regression_predict(X_train_norm, y_train, X_test_norm)
rmse_linear_normalized = compute_metrics(y_test, y_pred_linear_norm)
print("LINEAR REGRESSION RMSE NORMALIZED: " + str(rmse_linear_normalized))

print("We expect the feature normalization to not change anything for the linear regression model " \
"any discrepancies are likely due to floating point rounding or something of that nature")