
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Görev 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

scoutium_attributes = pd.read_csv("data/scoutium_attributes.csv", sep=";")
scoutium_attributes.head()
scoutium_potential_labels = pd.read_csv("data/scoutium_potential_labels.csv", sep=";")
scoutium_potential_labels.info()

# Görev 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştirelim.  ("task_response_id", 'match_id', 'evaluator_id' "player_id"  4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

df = pd.merge(scoutium_potential_labels, scoutium_attributes, on=["task_response_id", "match_id", "evaluator_id", "player_id" ])

# Görev 3: position_id içerisindeki Kaleci (1) sınıfını verisetinden kaldırınız.

df = df[df["position_id"] != 1]

# Görev 4: potential_label içerisindeki below_average sınıfını verisetinden kaldırınız.( below_average sınıfı  tüm verisetinin %1'ini oluşturur)

df["potential_label"].value_counts()
df = df[df["potential_label"] != "below_average"]

#Görev 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

# Adım 1: İndekste “player_id”,“position_id” ve “potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
df = pd.pivot_table(df, index=["player_id","position_id","potential_label"], columns="attribute_id", values="attribute_value")
df.head()

#Adım 2: “reset_index” fonksiyonunu kullanarak index hatasından kurtulunuz ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

df = df.reset_index()
df.columns = [str(col) for col in df.columns]
df.head()

# Görev 6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

# LABEL ENCODING
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
for col in binary_cols:
    df = label_encoder(df, col)



# Görev 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye kaydediniz.
def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
#cat_th 10 dan 5 e düşürdüm çünkü bazı kolonları num kabul etmiyodu.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5)
num_cols

#ONE HOT ENCODİNG
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

# Görev 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için standardScaler uygulayınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


#Kategorik değişken için analiz.
def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)
#1 sınıfı %20, 0 sınıfı %80 dengesiz veri seti.

# Görev 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.

y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)
#Base Model
models = [('LR', LogisticRegression(random_state=12345)),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345,verbose=-1)),
          ]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

'''########## LR ##########
Accuracy: 0.8636
Auc: 0.8479
Recall: 0.5233
Precision: 0.785
F1: 0.6151
########## CART ##########
Accuracy: 0.7561
Auc: 0.7003
Recall: 0.6067
Precision: 0.5458
F1: 0.5197
########## RF ##########
Accuracy: 0.8598
Auc: 0.8804
Recall: 0.4533
Precision: 0.8917
F1: 0.5562
########## XGB ##########
Accuracy: 0.845
Auc: 0.8382
Recall: 0.5733
Precision: 0.7008
F1: 0.5919
########## LightGBM ##########
Accuracy: 0.8743
Auc: 0.8757
Recall: 0.56
Precision: 0.8286
F1: 0.6226'''

# Dengesiz veri setlerinde daha çok f1 score a bakarız


#Hiperparametre Optimizasyonu ve sonrası model
# Random Forests

rf_model = RandomForestClassifier()
rf_model.get_params()
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 7, "auto"],
             "min_samples_split": [5, 8, 15,],
             "n_estimators": [100,500]}
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(rf_final, X, y, cv=10,
                            scoring=["accuracy", "f1","roc_auc"])

cv_results["test_accuracy"].mean()#0.87
cv_results["test_f1"].mean()#0.57
cv_results["test_roc_auc"].mean()#0.88
########## RF ##########
#roc_auc (Before): 0.8743
#roc_auc (After): 0.8922

# XGBoost

xgboost_model = XGBClassifier()
xgboost_model.get_params()
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 15],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)
xgboost_best_grid.best_params_
xgboost_final = xgboost_model.set_params(max_depth=5, colsample_bytree=0.5,learning_rate=0.01,n_estimators=1000, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()#0.85
cv_results['test_f1'].mean()#0.59
cv_results['test_roc_auc'].mean()#0.88
########## XGBoost ##########
#roc_auc (Before): 0.8308
#roc_auc (After): 0.8591

# LightGBM

lgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1],
               "verbose": [-1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(colsample_bytree=0.5,learning_rate=0.01,n_estimators=500,verbose=-1).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()#0.86
cv_results['test_f1'].mean()#0.56
cv_results['test_roc_auc'].mean()#0.87
########## LightGBM ##########
#roc_auc (Before): 0.8502
#roc_auc (After): 0.8545

#CART

cart_model = DecisionTreeClassifier()

cart_params = {"max_depth": range(1,11),
               "min_samples_split": range(2,20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X,y)

cart_best_grid.best_params_
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_).fit(X,y)
cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()#0.878
cv_results["test_f1"].mean()#0.58
cv_results["test_roc_auc"].mean()#0.72
########## CART ##########
#roc_auc (Before): 0.6917
#roc_auc (After): 0.7236

# Görev 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(cart_final, X)


#TAHMİN!!!

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01],
                   "n_estimators": [500],
                   "colsample_bytree":[0.5]}


classifiers = [("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params)]
#hiperparametre optimizasyonu yapan sürecin fonk
def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

#fonk verdiğimiz üç modele değerleri sorup en çok oyu olan değeri döndürecek bize
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


voting_clf = voting_classifier(best_models, X, y)
random_user = X.sample(1)
voting_clf.predict(random_user) #kullanıcı için yetenek tahmni yaptık
# PLAYER_İD 0.301 OLAN oyuncu için 0(başarısız) tahmin etti.





