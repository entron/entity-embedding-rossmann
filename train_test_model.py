import pickle
from models import *
import numpy
numpy.random.seed(42)
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn.preprocessing import OneHotEncoder
sys.setrecursionlimit(10000)

train_ratio = 0.1
shuffle_data = False
one_hot_as_input = False
embeddings_as_input = False
saved_embeddings_fname = "embeddings_unshuffled.pickle"  # Use plot_embeddings.ipynb to create

f = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f)

num_records = len(X)
train_size = int(train_ratio * num_records)

if shuffle_data:
    print("Using shuffled data")
    sh = numpy.arange(X.shape[0])
    numpy.random.shuffle(sh)
    X = X[sh]
    y = y[sh]

if embeddings_as_input:
    print("Using learned embeddings as input")
    X = embed_features(X, saved_embeddings_fname)

if one_hot_as_input:
    print("Using one-hot encoding as input")
    enc = OneHotEncoder(sparse=False)
    enc.fit(X)
    X = enc.transform(X)

X_train = X[:train_size]
X_val = X[train_size:(train_size + 10000)]
y_train = y[:train_size]
y_val = y[train_size:(train_size + 10000)]


def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = numpy.random.randint(num_row, size=n)
    return X[indices, :], y[indices]

X_train, y_train = sample(X_train, y_train, 100000)  # Simulate data sparsity

# data = [X_train, y_train, X_val, y_val]
# with open('data.pickle', 'wb') as f:
#     pickle.dump(data, f, -1)

models = []

print("Fitting NN_with_EntityEmbedding...")
for i in range(5):
    models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val))

# print("Fitting NN...")
# for i in range(5):
#     models.append(NN(X_train, y_train, X_val, y_val))

# print("Fitting LinearModel...")
# models.append(LinearModel(sX_train, y_train, X_val, y_val))

# print("Fitting RF...")
# models.append(RF(X_train, y_train, X_val, y_val))

# print("Fitting KNN...")
# models.append(KNN(X_train, y_train, X_val, y_val))

# print("Fitting XGBoost...")
# models.append(XGBoost(X_train, y_train, X_val, y_val))

# print("Fitting HistricalMedian...")
# models.append(HistricalMedian(X_train, y_train, X_val, y_val))

# with open('models.pickle', 'wb') as f:
#     pickle.dump(models, f)


def evaluate_models(models, X, y):
    assert(min(y) > 0)
    guessed_sales = numpy.array([model.guess(X) for model in models])
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = numpy.absolute((y - mean_sales) / y)
    result = numpy.sum(relative_err) / len(y)
    return result

print("Evaluate combined models...")
print("Training error...")
r_train = evaluate_models(models, X_train, y_train)
print(r_train)

print("Validation error...")
r_val = evaluate_models(models, X_val, y_val)
print(r_val)
