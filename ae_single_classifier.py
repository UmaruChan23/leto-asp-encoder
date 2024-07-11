import statistics

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from autoencoder import Autoencoder


class AeSingleClassifier:
    WARNING_THRESHOLD = 3
    DETECT_THRESHOLD = 5

    def __init__(self, batch_size: int = 32, encoder: bool = True, encoder_epochs: int = 100):
        self._encoder_model = None
        self._classifier_model = RandomForestClassifier()
        self._batch_size = batch_size
        self._encoder = encoder
        self._encoder_epochs = encoder_epochs

    def _train_encoder(self, ae_train_data, target_feature_name):

        train, valid = train_test_split(ae_train_data.drop(target_feature_name, axis=1), test_size=0.2)

        current_encoder = Autoencoder((train.shape[1],))

        current_history = current_encoder.fit(train_data=train, validation_data=valid, epochs=self._encoder_epochs)

        self._encoder_model = current_encoder

    def _train_classifier(self, classifiers_train_data: pd.DataFrame, target_feature_name):

        features = classifiers_train_data.drop(target_feature_name, axis=1)
        labels = classifiers_train_data[target_feature_name]

        train = pd.DataFrame(data=self._encoder_model.predict(features), index=features.index, columns=features.columns)
        train[target_feature_name] = labels

        current_classifier = RandomForestClassifier(random_state=42)

        c_train = train.sample(frac=1.0, replace=False)

        current_classifier.fit(c_train.drop(target_feature_name, axis=1), c_train[target_feature_name])

        self._classifier_model = current_classifier

    def fit(self, data: pd.DataFrame, target_feature_name: str):
        ae_train_data, classifiers_train_data = train_test_split(data,
                                                                 test_size=0.5,
                                                                 stratify=data[target_feature_name],
                                                                 shuffle=True)

        self._train_encoder(ae_train_data, target_feature_name)
        self._train_classifier(classifiers_train_data, target_feature_name)

    def predict(self, data: pd.DataFrame):
        ae_predict = pd.DataFrame(data=self._encoder_model.predict(data), index=data.index, columns=data.columns)
        classes = self._classifier_model.classes_
        classifier_predict = pd.DataFrame(self._classifier_model.predict_proba(ae_predict))
        classifier_predict = classifier_predict.apply(lambda instance: classifier_predict.columns[np.argmax(instance)], axis=1)
        classifier_predict = classifier_predict.apply(lambda instance: classes[instance])
        return classifier_predict

