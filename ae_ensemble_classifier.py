import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from autoencoder import Autoencoder


class AeEnsembleClassifier:
    WARNING_THRESHOLD = 3
    DETECT_THRESHOLD = 5

    def __init__(self, batch_size: int = 32):
        self._autoencoders = {}
        self._classifiers = []
        self._batch_size = batch_size
        self.NORMAL_APP = 1111

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

            self._autoencoders[target_feature_value] = ae_info

    def _train_classifiers(self, classifiers_train_data: pd.DataFrame, target_feature_name):

        for target_feature_value, ae_info in self._autoencoders.items():

            features = classifiers_train_data.drop(target_feature_name, axis=1)
            labels = classifiers_train_data[target_feature_name]

            train = pd.DataFrame(data=ae_info["ae"].predict(features), index=features.index, columns=features.columns)
            train[target_feature_name] = labels

            grouped_data = train.groupby(by=[target_feature_name])

            current_classifier = RandomForestClassifier(random_state=42)

            positive_sample = train
            positive_sample[target_feature_name] = 1

            negative_samples = classifiers_train_data[classifiers_train_data[target_feature_name] != target_feature_value]
            negative_samples[target_feature_name] = 0
            negative_samples = (negative_samples.groupby(by=[target_feature_name])
                                .apply(lambda x: x.sample(int(len(positive_sample) / (len(grouped_data) - 1)))))

            c_train = pd.concat([positive_sample, negative_samples]).sample(frac=1.0, replace=False)

            current_classifier.fit(c_train.drop(target_feature_name, axis=1), c_train[target_feature_name])

            self._autoencoders[target_feature_value]["classifier"] = current_classifier


    def fit(self, data: pd.DataFrame, target_feature_name: str) -> pd.DataFrame:
        ae_train_data, classifiers_train_data = train_test_split(data,
                                                                 test_size=0.5,
                                                                 stratify=data[target_feature_name],
                                                                 shuffle=True)

        self._train_encoders(ae_train_data, target_feature_name)
        self._train_classifiers(classifiers_train_data, target_feature_name)

        #TODO: доделать предикт, написать код эксперимента без фона
        print()

    def predict(self, data: pd.DataFrame):
        return None
