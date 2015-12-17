import numpy as np
import pandas as pd
import pickle


def generate_forward_backward_information(csv_path, window_size=14):
    dat = pd.read_csv(csv_path)
    columns = ["Promo", "StateHoliday", "SchoolHoliday"]

    column_dat = np.array(dat[columns])
    (nr_obs, nr_cols) = column_dat.shape
    generated_features = []
    for i in range(-window_size, window_size + 1):
        if i == 0:
            continue

        rolled = np.roll(column_dat, i, axis=0)
        if i < 0:
            rolled[i:, :] = column_dat[nr_obs - 1, :]
        else:
            rolled[0:i, :] = column_dat[0, :]

        for col in range(nr_cols):
            generated_features.append(rolled[:, col])

    return(np.array(generated_features).T)


def generate_forward_backward_information_accumulated(data, store_id, window_size=14, only_zero=False):
    dat_store = data[data["Store"] == store_id]
    columns = ["Promo", "StateHoliday", "SchoolHoliday"]
    # columns = ["StateHoliday"]

    column_dat = np.array(dat_store[columns], dtype="str")
    (nr_obs, nr_cols) = column_dat.shape
    generated_features = []
    generated_features.append([timestamp for timestamp in dat_store["Date"][::-1]])
    generated_features.append([store_id for i in range(nr_obs)])

    timestamps = np.array(dat_store["Date"], dtype="datetime64[D]")
    column_dat = column_dat[::-1]
    timestamps = timestamps[::-1]

    generated_feature_names = []
    generated_feature_names.append("Date")
    generated_feature_names.append("Store")

    for i_col, column in enumerate(columns):
        unique_elements = set(el for el in np.array(dat_store[column], dtype="str"))
        if only_zero:
            unique_elements = set("0")

        for unique_element in unique_elements:
            first_forward_looking = []
            last_backward_looking = []
            count_forward_looking = []
            count_backward_looking = []
            forward_looking_timestamps = smart_timestamp_accessor(timestamps)
            backward_looking_timestamps = smart_timestamp_accessor(timestamps)
            for i_obs in range(nr_obs):
                timestamp = timestamps[i_obs]
                timestamp_forward = timestamp + np.timedelta64(window_size, "D")
                timestamp_backward = timestamp + np.timedelta64(-window_size, "D")
                index_forward = forward_looking_timestamps.compute_index_of_timestamp(timestamp_forward)
                index_backward = backward_looking_timestamps.compute_index_of_timestamp(timestamp_backward)

                if i_obs == nr_obs - 1:
                    first_forward_looking.append(window_size + 1)
                    count_forward_looking.append(0)
                else:
                    forward_looking_data = column_dat[(i_obs + 1):(index_forward + 1), i_col]
                    forward_looking_data = forward_looking_data != unique_element
                    nr_occurences = np.sum(forward_looking_data)
                    if nr_occurences == 0:
                        first_forward_looking.append(window_size + 1)
                    else:
                        first_forward_looking.append(np.argmax(forward_looking_data) + 1)
                    count_forward_looking.append(nr_occurences)

                if i_obs == 0:
                    last_backward_looking.append(window_size + 1)
                    count_backward_looking.append(0)
                else:
                    backward_looking_data = column_dat[index_backward:i_obs, i_col]
                    backward_looking_data = backward_looking_data != unique_element
                    nr_occurences = np.sum(backward_looking_data)
                    if nr_occurences == 0:
                        last_backward_looking.append(window_size + 1)
                    else:
                        last_backward_looking.append(np.argmax(backward_looking_data[::-1]) + 1)
                    count_backward_looking.append(np.sum(backward_looking_data))

            generated_features.append(first_forward_looking)
            generated_features.append(last_backward_looking)
            if column == "StateHoliday":
                generated_features.append(count_forward_looking)
                generated_features.append(count_backward_looking)

            generated_feature_names.append(column + "_first_forward_looking")
            generated_feature_names.append(column + "_last_backward_looking")
            if column == "StateHoliday":
                generated_feature_names.append(column + "_count_forward_looking")
                generated_feature_names.append(column + "_count_backward_looking")

    return (np.array(generated_features).T, generated_feature_names)


def generate_forward_backward_accumulated_all_stores():
    train_path = "train.csv"
    test_path = "test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    store_ids = np.unique(train_data["Store"])

    fb = {}

    print("Generating features for train data ...")
    for store_id in store_ids:
        print(store_id)
        (generated_features, feature_names) = generate_forward_backward_information_accumulated(
            train_data, store_id, window_size=7, only_zero=True)
        date_index = feature_names.index("Date")
        for row in generated_features:
            key = (store_id, row[date_index])
            fb[key] = row[2:]

    print("Generating features for test data ...")
    for store_id in store_ids:
        print(store_id)
        (generated_features, feature_names) = generate_forward_backward_information_accumulated(
            test_data, store_id, window_size=7, only_zero=True)
        date_index = feature_names.index("Date")
        for row in generated_features:
            key = (store_id, row[date_index])
            fb[key] = row[2:]

    f = open("fb.pickle", "wb")
    pickle.dump(fb, f)

    return fb


class smart_timestamp_accessor(object):

    def __init__(self, timestamps):
        self.timestamps = timestamps
        self._last_index = 0

    def compute_index_of_timestamp(self, timestamp):
        if timestamp < self.timestamps[0]:
            return 0
        if timestamp > self.timestamps[len(self.timestamps) - 1]:
            return (len(self.timestamps) - 1)

        if timestamp == self.timestamps[self._last_index]:
            return self._last_index

        if timestamp > self.timestamps[self._last_index]:
            while timestamp > self.timestamps[self._last_index]:
                self._last_index += 1
            return self._last_index

        if timestamp < self.timestamps[self._last_index]:
            while timestamp < self.timestamps[self._last_index]:
                self._last_index -= 1
            return self._last_index

    def get_start_end_timestamp(self):
        timestamp_start = self.timestamps[0]
        nr_timestamps = len(self.timestamps)
        timestamp_end = self.timestamps[nr_timestamps - 1]
        return (timestamp_start, timestamp_end)

generate_forward_backward_accumulated_all_stores()
