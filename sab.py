import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('kddcup99.csv')
# print(df.shape)

df.drop_duplicates(keep='first', inplace=True)
# print(df.shape)
# print(df.columns)

# df.info()

input_cols = list(df.columns)[1:-1]
target_col = 'label'
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]

scaler = MinMaxScaler()
scaler.fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])

le = LabelEncoder()

target = df['label']
df['label'] = le.fit_transform(target)
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])

# print(df.sample(5))

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
# print(train_df.shape)
# print(test_df.shape)

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

sel = SelectFromModel(RandomForestClassifier(n_estimators=5, random_state=42))
sel.fit(train_inputs, train_targets)
selected_feat = train_inputs.columns[(sel.get_support())]
# print(selected_feat)
# print(len(selected_feat))

rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(train_inputs[selected_feat], train_targets)
preds_rf = rf.predict(test_inputs[selected_feat])
score_rf = accuracy_score(test_targets, preds_rf)
print("Accuracy of Random Forests: ", score_rf)

dc = DecisionTreeClassifier()
dc.fit(train_inputs[selected_feat], train_targets)
preds_dc = dc.predict(test_inputs[selected_feat])
score_dc = accuracy_score(test_targets, preds_dc)
print("Accuracy of Decision Tree: ", score_dc)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_inputs[selected_feat], train_targets)
preds_knn = knn.predict(test_inputs[selected_feat])
score_knn = accuracy_score(test_targets, preds_knn)
print("Accuracy of K Nearest Neighbors: ", score_knn)

rf_1 = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_1.fit(train_inputs[["diff_srv_rate", "same_srv_rate", "dst_host_same_srv_rate", "flag",
                       "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]], train_targets)
preds_rf_1 = rf_1.predict(test_inputs[["diff_srv_rate", "same_srv_rate",
                                       "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]])
score_rf_1 = accuracy_score(test_targets, preds_rf_1)
print("Accuracy of Random Forest Classifier is: ", score_rf_1)

dc_1 = DecisionTreeClassifier()
dc_1.fit(train_inputs[["diff_srv_rate", "same_srv_rate", "dst_host_same_srv_rate",
                       "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]], train_targets)
preds_dc_1 = dc_1.predict(test_inputs[["diff_srv_rate", "same_srv_rate",
                                       "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]])
score_dc_1 = accuracy_score(test_targets, preds_rf_1)
print("Accuracy of Decision Tree Classifier is: ", score_dc_1)

knn_1 = KNeighborsClassifier(n_neighbors=7)
knn_1.fit(train_inputs[["diff_srv_rate", "same_srv_rate",
                        "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]], train_targets)
preds_knn_1 = knn_1.predict(test_inputs[["diff_srv_rate",
                                         "same_srv_rate", "dst_host_same_srv_rate", "flag", "count", "dst_host_srv_count", "dst_host_srv_serror_rate"]])
score_knn_1 = accuracy_score(test_targets, preds_knn_1)
print("Accuracy of K Nearest Neighbors is: ", score_knn_1)
