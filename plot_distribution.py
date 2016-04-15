import pickle
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.stats.mstats import normaltest
from scipy.optimize import curve_fit
from sklearn import manifold
import matplotlib
import math
import itertools


font = {'family': 'normal',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)


def gaus(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def plot_sales_along_axes(X_embedded, X, y, axes):
    y_projected = []
    for axis in axes:
        nr_categories = 1
        colors = cm.rainbow(np.linspace(0, 1, nr_categories))
        for category_index, c in zip(range(nr_categories), colors):
            x_projected = []
            y_projected = []
            for record_embedded, record, sales in zip(X_embedded, X, y):
                if record[2] == 2 and record[3] == 0 and record[4] == 0 and record[5] == 10:
                    projected = np.dot(axis, record_embedded)
                    x_projected.append(projected)
                    y_projected.append(sales)

            plt.scatter(x_projected, y_projected)
        plt.show()


def plot_distribution_along_axis(X_embedded, X, axes):
    for axis in axes:
        nr_categories = 1
        colors = cm.rainbow(np.linspace(0, 1, nr_categories))
        for category_index, c in zip(range(nr_categories), colors):
            x_projected = []
            for record_embedded, record in zip(X_embedded, X):
                if True:
                    projected = np.dot(axis, record_embedded)
                    x_projected.append(projected)

            hist, bins = np.histogram(x_projected, bins=50)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
            popt, pcov = curve_fit(gaus, center, hist, p0=[1.0, 0.0, 1.0])
            plt.plot(center, gaus(center, *popt), color='red', linewidth=2)
            print(normaltest(x_projected))
        plt.show()


# same notation as in: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Multivariate_normality_tests
def mardia_test(sample):
    (n, k) = sample.shape
    x_bar = np.mean(sample, axis=0)
    sigma = np.cov(sample.T)
    sigma_inv = np.linalg.inv(sigma)
    A = 0.0
    for i, j in itertools.product(range(n), range(n)):
        x_i = sample[i, :]
        x_j = sample[j, :]
        A += (np.dot((x_i - x_bar).T, np.dot(sigma_inv, (x_j - x_bar))))**3
    A /= (6*n)

    B = 0.0
    for i in range(n):
        x_i = sample[i, :]
        B += (np.dot((x_i - x_bar).T, np.dot(sigma_inv, (x_i - x_bar))))**2
    B /= n
    B -= k*(k + 2)
    B *= math.sqrt(n/(8*k*(k + 2)))

    print("A", A)
    print("B", B)

    return (A, B)


def plot_surface_slice(X_embedded, X, axis1, axis2):
    x1_projected = []
    x2_projected = []
    for record_embedded, record in zip(X_embedded, X):
        # if record[1] == 0 and record[4] == 0:
        if record[2] == 2 and record[3] == 0 and record[4] == 0 and record[5] == 10:
            x1_projected.append(np.dot(record_embedded, axis1))
            x2_projected.append(np.dot(record_embedded, axis2))

    # print(np.unique(x1_projected))
    plt.scatter(x1_projected, x2_projected)
    plt.show()


def plot_tsne_embedding(X_embedded, X):
    x_store = X_embedded  # X_embedded[X[:, 1] == 0]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(x_store)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


def embedd_features(X, feature_index):
    # f_embeddings = open("embeddings.pickle", "rb")
    f_embeddings = open("embeddings_shuffled.pickle", "rb")
    embeddings = pickle.load(f_embeddings)

    index_embedding_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    embedding_index = index_embedding_mapping[feature_index]
    X_embedded = []

    (num_records, num_features) = X.shape
    for record in X:
        feat = int(record[feature_index])
        embedded_features = embeddings[embedding_index][feat].tolist()
        X_embedded.append(embedded_features)

    return numpy.array(X_embedded)

f = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f)

X_store_index = numpy.zeros((1115, 8))
for i in range(1115):
    X_store_index[i, 1] = i

X_dow_index = numpy.zeros((7, 8))
for i in range(7):
    X_dow_index[i, 2] = i

X_embedded_store = embedd_features(X_store_index, 1)
print(X_embedded_store)
print(X_embedded_store.shape)

mardia_test(X_embedded_store)

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_embedded_store)
print("principal components...")
print(pca.components_)
print("-"*40)

print(pca.explained_variance_ratio_)

mardia_test(X_pca[:, 0:2])

plot_sales_along_axes(X_embedded_store, X, y, pca.components_[0:2])
plot_distribution_along_axis(X_embedded_store, X, pca.components_[0:4])

random_direction = np.random.rand(X_embedded_store.shape[1])
random_direction = random_direction / (np.dot(random_direction, random_direction))**0.5

plot_sales_along_axes(X_embedded_store, X, y, [random_direction])
