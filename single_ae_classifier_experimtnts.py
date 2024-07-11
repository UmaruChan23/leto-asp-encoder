import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, accuracy_score, \
    recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ae_single_classifier import AeSingleClassifier
from android_app_data_source import load_data, Encryption, Application

target_feature_name = "app_id"

# Загрузка данных
data = load_data(encryption=Encryption.YES, exclude_application=[Application.MI_RU])

# Разделение данных на обучающую и тестовую выборки
train, test = train_test_split(data, test_size=0.2, random_state=42)

X_train = train.drop(target_feature_name, axis=1)

scaler = MinMaxScaler()
scale_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

scale_train[target_feature_name] = train[target_feature_name]

models = AeSingleClassifier(encoder=True, encoder_epochs=500)
models.fit(scale_train, "app_id")

X_test = test.drop(target_feature_name, axis=1)

X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

true_labels = pd.DataFrame(test[target_feature_name])

predicted_labels = models.predict(X_test)

print(classification_report(true_labels, predicted_labels, digits=4))
print(accuracy_score(true_labels, predicted_labels))
print(precision_score(true_labels, predicted_labels, average='macro'))
print(recall_score(true_labels, predicted_labels, average='macro'))
print(f1_score(true_labels, predicted_labels, average='macro'))