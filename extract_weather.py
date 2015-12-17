import os
import pickle
import glob
import csv


def csv2dicts(csvfile):
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        if row_index == 0:
            keys = row
            print(row)
            continue
        # if row_index % 10000 == 0:
        #     print(row_index)
        data.append({key: value for key, value in zip(keys, row)})
    return data


def set_nan_as_string(data, replace_str='0'):
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == '':
                x[key] = replace_str
        data[i] = x


def event2int(event):
    event_list = ['', 'Fog-Rain', 'Fog-Snow', 'Fog-Thunderstorm',
                  'Rain-Snow-Hail-Thunderstorm', 'Rain-Snow', 'Rain-Snow-Hail',
                  'Fog-Rain-Hail', 'Fog', 'Fog-Rain-Hail-Thunderstorm', 'Fog-Snow-Hail',
                  'Rain-Hail', 'Rain-Hail-Thunderstorm', 'Fog-Rain-Snow', 'Rain-Thunderstorm',
                  'Fog-Rain-Snow-Hail', 'Rain', 'Thunderstorm', 'Snow-Hail',
                  'Rain-Snow-Thunderstorm', 'Snow', 'Fog-Rain-Thunderstorm']
    return event_list.index(event)


def states_names_to_abbreviation(state_name):
    d = {}
    d['BadenWuerttemberg'] = 'BW'
    d['Bayern'] = 'BY'
    d['Berlin'] = 'BE'
    d['Brandenburg'] = 'BB'  # do not exist in store_state
    d['Bremen'] = 'HB'  # we use Niedersachsen instead of Bremen
    d['Hamburg'] = 'HH'
    d['Hessen'] = 'HE'
    d['MecklenburgVorpommern'] = 'MV'  # do not exist in store_state
    d['Niedersachsen'] = 'HB,NI'  # we use Niedersachsen instead of Bremen
    d['NordrheinWestfalen'] = 'NW'
    d['RheinlandPfalz'] = 'RP'
    d['Saarland'] = 'SL'
    d['Sachsen'] = 'SN'
    d['SachsenAnhalt'] = 'ST'
    d['SchleswigHolstein'] = 'SH'
    d['Thueringen'] = 'TH'

    return d[state_name]


csv_location = 'weather'
german_states_weather = glob.glob(csv_location + '/*.csv')
weather = {}

events = []
for one_state in german_states_weather:
    state_name = os.path.splitext(os.path.basename(one_state))[0]
    state_code = states_names_to_abbreviation(state_name)
    with open(one_state, 'r') as csvfile:
        daily_weather = csv.reader(csvfile, delimiter=';')
        for row_index, one_day in enumerate(daily_weather):
            if row_index == 0:
                continue
            date = one_day[0]
            key = (state_code, date)
            temperature = [int(one_day[1]), int(one_day[2]), int(one_day[3])]
            temperature = [(x - 10) / 30 for x in temperature]  # normalize
            humidity = [int(one_day[7]), int(one_day[8]), int(one_day[9])]
            humidity = [(x - 50) / 50 for x in humidity]  # normalize
            wind = [int(one_day[16]) / 50, int(one_day[17]) / 30]
            if one_day[20] == 'NA':
                cloud = [0]
            else:
                cloud = [int(one_day[20])]
            event = [event2int(one_day[21])]
            weather[key] = temperature + humidity + wind + cloud + event
            events.append(one_day[21])

print(set(events))
with open('weather.pickle', 'wb') as f:
    pickle.dump(weather, f, -1)
    print(len(weather))
