This code is used to generate sales prediction for the Kaggle Rossmann Sales competition with deep neural networks.

To effectively incorporate category features, we proposed **category embedding** to encode category features using learnt vectors. Like semantic embedding in natural language processing, category embedding enable us to express and learn the complex relations between different categories in a multi-dimentioal vector space. The final neural network is a simple two layer neural network on top of all embeddings and other non-category features. By averaging the predictions of 10 networks we got rank 3 leader board without much tuning. We will publish a paper to describe the method in more detail.

To run the code one needs first download and unzip the `train.csv`, `test.csv` and `store.csv` files on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and put them inside this folder. I have already included the extra store states, weather and google trend data shared in the competition forum by dune_dweller, MCFG and Tobias Wolfanger respectively, so you don't need to download them.

Next run the following scripts to extract and prepare features:

```
python3 extract.py
python3 extract_weather.py
python3 extract_google_trend.py
python3 extract_fb_features.py # Extract forward/backward looking features
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

Acknowledge:

Thank our founder Ozel Christo, Andrei Ciobotar and all colleagues at Neokami for supporting and encouragement! Thank our team member Felix Berkhahn and Aleksandra Pachalieva for helping me out near the end of the competition for preparing additional features and visualization.