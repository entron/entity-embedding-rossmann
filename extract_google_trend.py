import csv
import pickle
from datetime import datetime
import os
import glob


csv_location = 'googletrend'
google_trend_files = glob.glob(csv_location + '/*.csv')

google_trend = {}

for one_state in google_trend_files:
    state = os.path.splitext(os.path.basename(one_state))[0]
    state_code = state[-2:]
    if state_code == 'NI':
        state_code = 'HB,NI'
    print(state_code)
    with open(one_state, 'r') as csvfile:
        trends = csv.reader(csvfile, delimiter=',')
        for row, trend in enumerate(trends):
            if row == 0:
                    continue
            # The sata is represented from Sunday till Saturday - take Saturday and check the week number
            trend_value = int(trend[1])
            end_day_of_range = trend[0].split(' - ')[1]
            dt = datetime.strptime(end_day_of_range, '%Y-%m-%d')
            year = dt.year
            month = dt.month
            day = dt.day
            week_of_year = dt.isocalendar()[1]

            key = (state_code, year, week_of_year)
            google_trend[key] = trend_value / 100

with open('google_trends.pickle', 'wb') as f:
    pickle.dump(google_trend, f, -1)
    print(len(google_trend))
