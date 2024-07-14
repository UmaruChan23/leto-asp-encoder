import statistics
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from autoencoder import Autoencoder


class AeEnsembleClassifier:
    WARNING_THRESHOLD = 3
    DETECT_THRESHOLD = 5

    def __init__(self, batch_size: int = 32, encoder: bool = True, ensemble: bool = True, encoder_epochs: int = 100):
        self._models = {}
        self._batch_size = batch_size
        self._single_classifier = RandomForestClassifier()
        self._encoder_mse = {}
        self._encoder = encoder
        self._encoder_epochs = encoder_epochs
        self._ensemble = ensemble

    def _train_encoders(self, ae_train_data, target_feature_name):
        grouped_data = ae_train_data.groupby(by=[target_feature_name])

        for target_feature_value, group in grouped_data:
            target_feature_value = int(''.join(map(str, target_feature_value)))

            train, valid = train_test_split(group.drop(target_feature_name, axis=1), test_size=0.2)

            current_encoder = Autoencoder((train.shape[1],))

            current_history = current_encoder.fit(train_data=train, validation_data=valid, epochs=self._encoder_epochs)

            self._encoder_mse[target_feature_value] = current_history['loss'][-1]

            ae_info = {
                "ae": current_encoder,
                "classifier": None
            }

            self._models[target_feature_value] = ae_info

    def _train_single_classifier(self, classifiers_train_data: pd.DataFrame, target_feature_name):
        all_train = pd.DataFrame()

        for target_feature_value, ae_info in self._models.items():
            features = classifiers_train_data.drop(target_feature_name, axis=1)
            labels = classifiers_train_data[target_feature_name]

            train = pd.DataFrame(data=ae_info["ae"].predict(features), index=features.index, columns=features.columns)
            train[target_feature_name] = labels

            all_train = pd.concat([all_train, train])

        all_train = all_train.sample(frac=1.0, replace=False)

        current_classifier = RandomForestClassifier(random_state=42)

        current_classifier.fit(all_train.drop(target_feature_name, axis=1), all_train[target_feature_name])

        self._single_classifier = current_classifier

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

    def _train_classifiers_no_encoder(self, classifiers_train_data: pd.DataFrame, target_feature_name):

        for target_feature_value, ae_info in self._models.items():
            features = classifiers_train_data.drop(target_feature_name, axis=1)
            labels = classifiers_train_data[target_feature_name]

            train = features
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
        if self._ensemble:
            if self._encoder:
                self._train_encoders(ae_train_data, target_feature_name)
                self._train_classifiers(classifiers_train_data, target_feature_name)
            else:
                self._train_encoders(ae_train_data, target_feature_name)
                self._train_classifiers_no_encoder(classifiers_train_data, target_feature_name)
        else:
            if self._encoder:
                self._train_encoders(ae_train_data, target_feature_name)
                self._train_single_classifier(classifiers_train_data, target_feature_name)
            else:
                self._train_single_classifier(classifiers_train_data, target_feature_name)

    def _merge_labels(self, predictions: dict):
        voted_array = []
        for i in range(max(map(len, predictions.values()))):
            voted_array.append(Counter(map(lambda x: x[i] if i < len(x) else None, predictions.values())).most_common(1)[0][0])
        return voted_array

    def predict(self, data: pd.DataFrame):
        if self._ensemble:
            return self.ensemble_predict(data)
        else:
            return self.single_predict(data)

    def single_predict(self, data: pd.DataFrame):
        predictions = {}
        for target_feature_value, ae_info in self._models.items():
            ae_predict = pd.DataFrame(data=ae_info["ae"].predict(data), index=data.index, columns=data.columns)
            classes = self._single_classifier.classes_
            classifier_predict = pd.DataFrame(self._single_classifier.predict_proba(ae_predict))
            classifier_predict = classifier_predict.apply(lambda instance: classifier_predict.columns[np.argmax(instance)], axis=1)
            classifier_predict = classifier_predict.apply(lambda instance: classes[instance])
            predictions[target_feature_value] = classifier_predict

        result = self._merge_labels(predictions)

        return result

    def ensemble_predict(self, data: pd.DataFrame):
        predictions = pd.DataFrame()
        if self._encoder:
            for target_feature_value, ae_info in self._models.items():
                ae_predict = pd.DataFrame(data=ae_info["ae"].predict(data), index=data.index, columns=data.columns)
                classifier_predict = ae_info["classifier"].predict_proba(ae_predict)[:, 1]
                predictions[target_feature_value] = pd.Series(classifier_predict, index=data.index)

            return predictions.apply(lambda instance: predictions.columns[np.argmax(instance)], axis=1)
        else:
            for target_feature_value, ae_info in self._models.items():
                classifier_predict = ae_info["classifier"].predict_proba(data)[:, 1]
                predictions[target_feature_value] = classifier_predict

            return predictions.apply(lambda instance: predictions.columns[np.argmax(instance)], axis=1)
