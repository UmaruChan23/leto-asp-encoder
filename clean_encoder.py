import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score
from sklearn.model_selection import train_test_split

from ae_ensemble_classifier import AeEnsembleClassifier
from android_app_data_source import load_data, Encryption, Application

target_feature_name = "app_id"

# Загрузка данных
data = load_data(encryption=Encryption.YES, exclude_application=[Application.MI_RU])

# Разделение данных на обучающую и тестовую выборки
train, test = train_test_split(data, test_size=0.2, random_state=42)

models = AeEnsembleClassifier()
models.fit(data, "app_id")

X_test = test.drop(target_feature_name, axis=1)
true_labels = pd.DataFrame(test[target_feature_name])

predicted_labels = models.predict(X_test)

print(precision_score(true_labels, predicted_labels, average='macro'))