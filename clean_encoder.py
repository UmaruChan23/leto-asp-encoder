from ae_ensemble_classifier import AeEnsembleClassifier
from android_app_data_source import load_data, Encryption, Application

# Загрузка данных
data = load_data(encryption=Encryption.YES, exclude_application=[Application.MI_RU])

# Разделение данных на обучающую и тестовую выборки
#X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

models = AeEnsembleClassifier()
models.fit(data, "app_id")

