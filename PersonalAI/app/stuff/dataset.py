import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# pass in column names for each CSV as the column name is not given in the file and read them using pandas.
# You can check the column names from the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')

print(users.shape)
print(users.head())
print('\n')

# So, we have 943 users in the dataset and each user has 5 features,
# i.e. user_ID, age, sex, occupation and zip_code. Now let’s
# look at the ratings file.
print(ratings.shape)
print(ratings.head())
print('\n')

# We have 100k ratings for different user and movie combinations.
# Now finally examine the items file.
print(items.shape)
print(items.head())
print('\n')

# The dataset has already been divided into train and test by GroupLens where
# the test data has 10 ratings for each user, i.e. 9,430 rows in total.
# We will read both these files into our Python environment.
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
print(ratings_train.shape, ratings_test.shape)


# 4. Building Collaborative Filtering Model From Scratch

# We need to calculate the number of unique users and movies.
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

# Create a user-item matrix which can be used to calculate the
# similarity between users and items
data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

# Now, we will calculate the similarity. We can use the pairwise_distance
# function from sklearn to calculate the cosine similarity.
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

print(user_similarity)
print('\n')
print(item_similarity)

# This gives us the item-item and user-user similarity in an array form.
# The next step is to make predictions based on these similarities.
# Let’s define a function to do just that.
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) \
               / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

print("\n")

# Finally, we will make predictions based on user similarity and item similarity.
user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')
print(user_prediction)
print(item_prediction)

# 5. Building a simple popularity and collaborative filtering model
# using Turicreate
import turicreate
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)

# First we’ll build a model which will recommend movies based on
# the most popular choices, i.e., a model where all the users receive
# the same recommendation(s). We will use the turicreate recommender
# function popularity_recommender for this.
popularity_model = turicreate.popularity_recommender\
    .create(train_data, user_id='user_id', item_id='movie_id',
            target='rating')

# Various arguments we have used are:
# 1. train_data: the SFrame which contains the required training data
# 2. user_id: the column name which represents each user ID
# 3. item_id: the column name which represents each item to be recommended (movie_id)
# 4. target: the column name representing scores/ratings given by the user

# Prediction time: We will recommend the top 5 items for the first 5
# users in our dataset
popularity_reccom = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_reccom.print_rows(num_rows=25)

# This confirms that all the recommended movies have an average
# rating of 5, i.e. all the users who watched the movie gave it
# a top rating. Thus our popularity system works as expected.

# Building a collaborative filtering model. Let's train the item similarity
# model and make top 5 recommencations for the first 5 users

# Training The Model
item_sim_model = turicreate.item_similarity_recommender\
    .create(train_data, user_id='user_id', item_id='movie_id',
            target='rating', similarity_type='cosine')

# Making Recommendations

# Here we can see that the recommendations (movie_id) are
# different for each user. So personalization exists, i.e.
# for different users we have a different set of recommendations.
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
item_sim_recomm.print_rows(num_rows=25)


# In this model, we do not have the ratings for each movie given by each
# user. We must find a way to predict all these missing ratings. For that,
# we have to find a set of features which can define how a user rates the 
# movies. These are called latent features.
 
# We need to find a way to  extract the most important latent 
# features from the the existing features. 

# Matrix factorization, 
# covered in the next section, is one such technique which uses the 
# lower dimension dense matrix and helps in extracting the 
# important latent features.


# Introduction to Matrix Factorization + Building the recommendation engine by
# matrix factorization
class MF():

    # Initializing the user-movie rating matrix, no. of latent features, alpha, and beta
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items, = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and movie-feature matrix
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
        self.sgd()
        mse = self.mse()
        training_process.append((i, mse))
        if (i + 1) % 20 == 0:
            print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    # Rating for user i and movie j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_matrix(self):
        return mf.b + mf.b_u[:, np.newaxis] + mf.b_i[np.newaxis:, ] + mf.P.dot(mf.Q.T)


# Now we have a function that can predict the ratings. The input for this function are:
#
# R – The user-movie rating matrix
# K – Number of latent features
# alpha – Learning rate for stochastic gradient descent
# beta – Regularization parameter for bias
# iterations – Number of iterations to perform stochastic gradient descent

# We have to convert the user item ratings to matrix form. It can be
# done using the pivot function in python.
R = np.array(ratings.pivot(index = 'user_id', columns='movie_id', values='rating').fillna(0))

# fillna(0) will fill all the missing ratings with 0. Now we have the
# R matrix. We can initialize the number of latent features, but
# the number of these features must be less than or equal to the
# number of original features.

# Now let us predict all the missing ratings. Let’s take K=20,
# alpha=0.001, beta=0.01 and iterations=100.
mf = MF(R, K = 20, alpha = 0.001, beta = 0.01, iterations = 100)
training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()
