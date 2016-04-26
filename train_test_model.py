import pickle
import numpy
numpy.random.seed(123)
from models import *
from sklearn.preprocessing import OneHotEncoder
import sys
sys.setrecursionlimit(10000)

train_ratio = 0.9
shuffle_data = False
one_hot_as_input = False
embeddings_as_input = False
save_embeddings = False
save_models = False
saved_embeddings_fname = "embeddings.pickle"  # Use plot_embeddings.ipynb to create

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
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]


def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = numpy.random.randint(num_row, size=n)
    return X[indices, :], y[indices]

X_train, y_train = sample(X_train, y_train, 200000)  # Simulate data sparsity
print("Number of samples used for training: " + str(y_train.shape[0]))

models = []

print("Fitting NN_with_EntityEmbedding...")
for i in range(5):
    models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val))

# print("Fitting NN...")
# for i in range(5):
#     models.append(NN(X_train, y_train, X_val, y_val))

# print("Fitting RF...")
# models.append(RF(X_train, y_train, X_val, y_val))

# print("Fitting KNN...")
# models.append(KNN(X_train, y_train, X_val, y_val))

# print("Fitting XGBoost...")
# models.append(XGBoost(X_train, y_train, X_val, y_val))


if save_embeddings:
    model = models[0].model
    weights = model.get_weights()
    store_embedding = weights[0]
    dow_embedding = weights[1]
    year_embedding = weights[4]
    month_embedding = weights[5]
    day_embedding = weights[6]
    german_states_embedding = weights[7]
    with open(saved_embeddings_fname, 'wb') as f:
        pickle.dump([store_embedding, dow_embedding, year_embedding,
                    month_embedding, day_embedding, german_states_embedding], f, -1)

if save_models:
    with open('models.pickle', 'wb') as f:
        pickle.dump(models, f)


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
