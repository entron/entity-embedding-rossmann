This is the code used in the paper **"[Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737)"**. If you want to get the original version of the code used for the Kaggle competition, please use [**the Kaggle branch**](https://github.com/entron/entity-embedding-rossmann/tree/kaggle).

To run the code one needs first download and unzip the `train.csv` and `store.csv` files on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and put them in this folder.

The following packages are needed if you want to recover the result in the paper (we used python 3):

```
pip3 install -U scikit-learn
pip3 install -U xgboost
pip3 install keras
```
Please refer to [Keras](https://github.com/fchollet/keras) for more details regarding how to install keras. 

Next, run the following scripts to extract the csv files and prepare the features:

```
python3 extract_csv_files.py
python3 prepare_features.py
``` 

To run the models:

```
python3 train_test_model.py
```

You can anaylize the embeddings with the ipython notebook included. This is the learned embeeding of German States printed in 2D (with the Kaggle branch):

[![](https://plot.ly/~entron/0/.png)](https://plot.ly/~entron/0.embed)

and this is the learned embeddings of 1115 Rossmann stores printed in 3D:

[![](https://plot.ly/~entron/2/.png)](https://plot.ly/~entron/2.embed)
