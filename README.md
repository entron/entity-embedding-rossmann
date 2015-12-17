This code is used to generate sales prediction for the Kaggle Rossmann Sales competition with only neural networks.

To effectively incorporate category features, we proposed **category embedding** to encode category features using learnt vectors. Like semantic embedding in natural language processing, category embedding enable use to express and learn the complex relations between different categories in a multi-dimentioal vector space. The final neural network is a simple two layer neural network on top of all embeddings and inputs other non-category features. By averaging the predictions of 10 networks we got rank 3 leader board without much tuning. We will publish a paper to describe the method in more detail.

To run the code one needs first download the `train.csv`, `test.csv` and `store.csv` files on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales) and put in the folder. We have already included the extra store states, weather and google trend data shared in the competition forum by dune_dweller, MCFG and Tobias Wolfanger respectively, so you don't have to download them.

Next run the following scripts to extract and prepare features:

```
python3 extract.py
python3 extract_weather.py
python3 extract_google_trend.py
python3 extract_fb_features.py # Extract forward/backward looking features
python3 prepare_features.py
``` 

To test the neural network model run

```
python3 test_model.py
```

By default it will run one neural net with 0.97 data for training and the rest for test. You can change these two parameters in `test_model.py` if you want differently:

```
num_networks = 1
train_ratio = 0.97
```

After the script finishes it will generate a file `predictions.csv` which is used to submit to Kaggle.