This is the code used in the paper *"Entity Embeddings of Categorical Variables"* The original version of the code used for the kaggle competition is in [the kaggle branch](https://github.com/entron/entity-embedding-rossmann/tree/kaggle).

To run the code one needs first download and unzip the `train.csv` and `store.csv` files on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and put them inside this folder.

Next run the following scripts to extract and prepare features:

```
python3 extract.py
python3 prepare_features.py
``` 

To test the neural network model run (you need to have [keras](https://github.com/fchollet/keras) installed first)

```
python3 test_model.py
```

By default it will run one neural net with 0.97 data for training and the rest for test. It takes 20 minutes to run on Nvidia GTX 980 GPU, and it may take a few hours to run on CPU. 

You can change these two parameters in `test_model.py` if you want to use more models or a different train-test ratio, and the following is what I used for finial submission:

```
num_networks = 10
train_ratio = 1
```

After the script finishes it will generate a file `predictions.csv` which is used for submission to Kaggle.

You can anaylize the embeddings with the ipython notebook included. This is the learned embeeding of German States printed in 2D:

[![](https://plot.ly/~entron/0/.png)](https://plot.ly/~entron/0.embed)

and this is the learned embeddings of 1115 Rossmann stores printed in 3D:

[![](https://plot.ly/~entron/2/.png)](https://plot.ly/~entron/2.embed)

Acknowledge:

Thank our founder Ozel Christo, Andrei Ciobotar and all colleagues at Neokami for supporting and encouragement! 
