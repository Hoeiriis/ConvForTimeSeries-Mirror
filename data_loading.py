import os
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class NMRDataLoader:

    def __init__(self, path):
        print("Loading data..")

        with cd(path):
            # Load the data from the CSV files
            training_data = pd.read_csv('numerai_training_data.csv', header=0)
            tournament_data = pd.read_csv('numerai_tournament_data.csv', header=0)

            with open("Time_logger.txt", "a") as text_file:
                text_file.write("Last visited: {}\n".format(datetime.datetime.now().time()))

        # Readying training data
        features = [f for f in list(training_data) if "feature" in f]
        self.X_train = np.array(training_data[features])
        self.Y_train = np.array(training_data['target']).reshape((-1, 1))
        self.info_train = np.array(training_data.drop(features, axis=1))

        # Readying tournament data
        info_tournament = np.array(tournament_data.drop(features, axis=1))

        split = 0
        for i in range(0, info_tournament.shape[0]):
            if info_tournament[i, 2] == "test":
                split = i
                break

        # Validation set
        self.X_val = np.array(tournament_data[features])[:split, :]
        self.Y_val = (np.array(tournament_data['target'])[:split]).reshape((-1, 1))
        self.info_val = np.array(tournament_data.drop(features, axis=1))[:split, :]

        # Tournament set
        self.X_tournament = np.array(tournament_data[features])
        self.id_tournament = np.array(tournament_data['id'])
        self.info_tournament = np.array(tournament_data.drop(features, axis=1))[split:, :]

        self.normalize_data()

        print("Original data shapes:")
        print("X_train shape: {}".format(self.X_train.shape))
        print("X_val shape: {}".format(self.X_val.shape))
        print("X_tournament shape: {}\n".format(self.X_tournament.shape))

    def split_train_eras(self):

        train_eras_split = []

        latest_era = self.info_train[0, 1]
        for i in range(0, self.info_train.shape[0]):
            if self.info_train[i, 1] != latest_era:
                latest_era = self.info_train[i, 1]
                train_eras_split.append(i)

        return train_eras_split

    def split_val_eras(self):
        val_eras_split = [0]

        latest_era = self.info_val[0, 1]
        for i in range(0, self.info_val.shape[0]):
            if self.info_val[i, 1] != latest_era:
                latest_era = self.info_val[i, 1]
                val_eras_split.append(i)

        return val_eras_split

    def get_data_small_val(self, val_era_size=None):

        val_eras_split = self.split_val_eras()

        if val_era_size is None:
            val_era_size = len(val_eras_split)

        if len(val_eras_split) < val_era_size:
            raise ValueError("There is {} val eras, but the "
                             "requested val_era_size is {}".format(len(val_eras_split), val_era_size))

        X_val_splitted = np.split(self.X_val, [val_eras_split[-val_era_size]])
        Y_val_splitted = np.split(self.Y_val, [val_eras_split[-val_era_size]])

        return_X_train = np.concatenate((self.X_train, X_val_splitted[0]), axis=0)
        return_Y_train = np.concatenate((self.Y_train, Y_val_splitted[0].reshape((-1, 1))), axis=0)

        return_X_val = X_val_splitted[1]
        return_Y_val = Y_val_splitted[1].reshape((-1, 1))

        return return_X_train, return_Y_train, return_X_val, return_Y_val, self.X_tournament, self.id_tournament

    def normalize_data(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_tournament = scaler.transform(self.X_tournament)

def make_pred_file(storage_path, predictions, ids, name="predictions.csv"):
    results = predictions.reshape((-1, ))
    results_df = pd.DataFrame(data={'probability': results})
    joined = pd.DataFrame(data={"id": ids}).join(results_df)

    with cd(storage_path):
        joined.to_csv(name, index=False)


if __name__ == "__main__":

    tournament_number = "73"
    path = "C:/SOFTWARE and giggles/CuteFlower/CuteFlower2.0/Data/tournament{}".format(tournament_number)

    nmr_loader = NMRDataLoader(path)
    X_train, Y_train, X_val, Y_val, X_tournament, id_tournament = nmr_loader.get_data_small_val()

    print("X_train shape: {}".format(X_train.shape))
    print("X_val shape: {}".format(X_val.shape))
    print("X_tournament shape: {}".format(X_tournament.shape))