import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from autoencoder import Autoencoder


class AeEnsembleClassifier:
    WARNING_THRESHOLD = 3
    DETECT_THRESHOLD = 5

    def __init__(self, batch_size: int = 32):
        self._models = {}
        self._batch_size = batch_size

    def _train_encoders(self, ae_train_data, target_feature_name):
        grouped_data = ae_train_data.groupby(by=[target_feature_name])

        for target_feature_value, group in grouped_data:
            target_feature_value = int(''.join(map(str, target_feature_value)))

            train, valid = train_test_split(group.drop(target_feature_name, axis=1), test_size=0.2)

            current_encoder = Autoencoder((train.shape[1],))

            _ = current_encoder.fit(train, valid, 100)

            ae_info = {
                "ae": current_encoder,
                "classifier": None
            }

            self._models[target_feature_value] = ae_info

    def _train_classifiers(self, classifiers_train_data: pd.DataFrame, target_feature_name):

        for target_feature_value, ae_info in self._models.items():

            features = classifiers_train_data.drop(target_feature_name, axis=1)
            labels = classifiers_train_data[target_feature_name]

            train = pd.DataFrame(data=ae_info["ae"].predict(features), index=features.index, columns=features.columns)
            train[target_feature_name] = labels

            current_classifier = RandomForestClassifier(random_state=42)

            positive_sample = train[train[target_feature_name] == target_feature_value]
            positive_sample[target_feature_name] = 1

            negative_samples = train[train[target_feature_name] != target_feature_value]
            negative_samples[target_feature_name] = 0

            c_train = pd.concat([positive_sample, negative_samples]).sample(frac=1.0, replace=False)

            current_classifier.fit(c_train.drop(target_feature_name, axis=1), c_train[target_feature_name])

            self._models[target_feature_value]["classifier"] = current_classifier


    def fit(self, data: pd.DataFrame, target_feature_name: str):
        ae_train_data, classifiers_train_data = train_test_split(data,
                                                                 test_size=0.5,
                                                                 stratify=data[target_feature_name],
                                                                 shuffle=True)

        self._train_encoders(ae_train_data, target_feature_name)
        self._train_classifiers(classifiers_train_data, target_feature_name)

    def predict(self, data: pd.DataFrame):
        predictions = pd.DataFrame()
        for target_feature_value, ae_info in self._models.items():
            ae_predict = pd.DataFrame(data=ae_info["ae"].predict(data), index=data.index, columns=data.columns)
            classifier_predict = ae_info["classifier"].predict_proba(ae_predict)[:, 1]
            predictions[target_feature_value] = pd.Series(classifier_predict, index=data.index)

        return predictions.apply(lambda instance: predictions.columns[np.argmax(instance)], axis=1)
