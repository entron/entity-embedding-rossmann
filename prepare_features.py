import pickle
from datetime import datetime
from isoweek import Week
import math


with open('train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)
    num_records = len(train_data)
with open('test_data.pickle', 'rb') as f:
    test_data = pickle.load(f)
with open('store_data.pickle', 'rb') as f:
    store_data = pickle.load(f)
with open('weather.pickle', 'rb') as f:
    weather = pickle.load(f)
with open('fb.pickle', 'rb') as f:
    fb = pickle.load(f)
with open('google_trends.pickle', 'rb') as f:
    googletrend = pickle.load(f)


def abc2int(char):
    d = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    return d[char]


def state2int(state):
    d = {'HB,NI': 0, 'HH': 1, 'TH': 2, 'RP': 3, 'ST': 4, 'BW': 5,
         'SN': 6, 'BE': 7, 'HE': 8, 'SH': 9, 'BY': 10, 'NW': 11}
    return d[state]


def PromoInterval2int(promointerval):
    char = promointerval[0]
    d = {'0': 0, 'J': 1, 'F': 2, 'M': 3}
    return d[char]


def hasCompetitionmonths(date, CompetitionOpenSinceYear, CompetitionOpenSinceMonth):
    if CompetitionOpenSinceYear == 0:
        return 0
    dt_competition_open = datetime(year=CompetitionOpenSinceYear,
                                   month=CompetitionOpenSinceMonth,
                                   day=15)
    months_since_competition = (date - dt_competition_open).days // 30
    if months_since_competition < 0:
        return 0
    return min(months_since_competition, 24)


def hasPromo2weeks(date, Promo2SinceYear, Promo2SinceWeek):
    if Promo2SinceYear == 0:
        return 0
    start_promo2 = Week(Promo2SinceYear, Promo2SinceWeek).monday()
    weeks_since_promo2 = (date.date() - start_promo2).days // 7
    if weeks_since_promo2 < 0:
        return 0
    return min(weeks_since_promo2, 25)


def latest_promo2_months(date, promointerval, Promo2SinceYear, Promo2SinceWeek):
    if not hasPromo2weeks(date, Promo2SinceYear, Promo2SinceWeek):
        return 0
    promo2int = PromoInterval2int(promointerval)
    if promo2int == 0:
        return 0

    if date.month < promo2int:
        latest_promo2_start_year = date.year - 1
        latest_promo2_start_month = promo2int + 12 - 3
    else:
        latest_promo2_start_year = date.year
        latest_promo2_start_month = ((date.month - promo2int) // 3) * 3 + promo2int

    latest_promo2_start_day = datetime(year=latest_promo2_start_year,
                                       month=latest_promo2_start_month,
                                       day=1)
    weeks_since_latest_promo2 = (date - latest_promo2_start_day).days // 30
    return weeks_since_latest_promo2


def feature_list(record):
    dt = datetime.strptime(record['Date'], '%Y-%m-%d')
    store_index = int(record['Store'])
    year = dt.year
    month = dt.month
    day = dt.day
    week_of_year = dt.isocalendar()[1]
    day_of_week = int(record['DayOfWeek'])
    try:
        store_open = int(record['Open'])
    except:
        store_open = 1
    state_holiday = abc2int(record['StateHoliday'])
    school_holiday = int(record['SchoolHoliday'])
    # num_customers = int(record['Customers'])
    promo = int(record['Promo'])
    try:
        distance = int(store_data[store_index - 1]['CompetitionDistance'])
    except:
        distance = 0
    has_competition_for_months = hasCompetitionmonths(dt,
                                                      int(store_data[store_index - 1]['CompetitionOpenSinceYear']),
                                                      int(store_data[store_index - 1]['CompetitionOpenSinceMonth']))

    has_promo2_for_weeks = hasPromo2weeks(dt,
                                          int(store_data[store_index - 1]['Promo2SinceYear']),
                                          int(store_data[store_index - 1]['Promo2SinceWeek']))

    latest_promo2_for_months = latest_promo2_months(dt,
                                                    store_data[store_index - 1]['PromoInterval'],
                                                    int(store_data[store_index - 1]['Promo2SinceYear']),
                                                    int(store_data[store_index - 1]['Promo2SinceWeek']))
    weather_key = (store_data[store_index - 1]['State'], record['Date'])
    fb_key = (store_index, record['Date'])
    google_trend_key_DE = ('DE', year, week_of_year)
    google_trend_key_state = (store_data[store_index - 1]['State'], year, week_of_year)
    return [store_open,
            store_index,
            day_of_week,
            promo,
            year,
            month,
            day,
            state_holiday,
            school_holiday,
            has_competition_for_months,
            has_promo2_for_weeks,
            latest_promo2_for_months,
            math.log(distance + 1) / 10,
            abc2int(store_data[store_index - 1]['StoreType']),
            abc2int(store_data[store_index - 1]['Assortment']),
            PromoInterval2int(store_data[store_index - 1]['PromoInterval']),
            int(store_data[store_index - 1]['CompetitionOpenSinceYear']),
            int(store_data[store_index - 1]['Promo2SinceYear']),
            state2int(store_data[store_index - 1]['State']),
            week_of_year,
            ] + weather[weather_key] + [int(el) for el in fb[fb_key]] + [googletrend[google_trend_key_DE], googletrend[google_trend_key_state]]


train_data_X = []
train_data_y = []
for record in train_data:
    if record['Sales'] != '0' and record['Open'] != '':
        fl = feature_list(record)
        train_data_X.append(fl)
        train_data_y.append(int(record['Sales']))
print("Number of train datapoints: ", len(train_data_y))

test_data_X = []
for record in test_data:
    fl = feature_list(record)
    test_data_X.append(fl)
print("Number of test datapoints: ", len(test_data_X))

print(min(train_data_y), max(train_data_y))


with open('feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])

with open('feature_test_data.pickle', 'wb') as f:
    pickle.dump(test_data_X, f, -1)
