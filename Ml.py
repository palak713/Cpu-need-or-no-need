import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
from sklearn import metrics
pd.options.display.max_columns = None
pd.options.display.max_rows = None
dataset = pd.read_excel("Kaggle_Sirio_Libanes_ICU_Prediction.xls")
# create new column to indicate if a patient eventually went to ICU (ICU_SUM)
df_admitted = (dataset.groupby("PATIENT_VISIT_IDENTIFIER")["ICU"].sum()>0).reset_index()*1
df_admitted.columns = ["PATIENT_VISIT_IDENTIFIER", "ICU_SUM"]
dataset_admitted = pd.merge(dataset, df_admitted, on = "PATIENT_VISIT_IDENTIFIER")
#check for missing data
dataset.isna().sum()
# fill missing values
dataset_admitted.fillna(method='ffill', inplace = True)
dataset_admitted.fillna(method='bfill', inplace = True)
#drop rows with ICU == 1 ie drop data when the target variable is present, as stipulated by dataset author
dataset_ = dataset_admitted[dataset_admitted.ICU == 0].reset_index(drop= True)
#keeping only window 0-2 data
dataset_ = dataset_[dataset_.WINDOW == "0-2"].reset_index(drop = True)
#drop unnecessary columns
final_data = dataset_.drop(["PATIENT_VISIT_IDENTIFIER", "WINDOW", "ICU"],axis = 1)
#look for categorical columns and convert them
cat_columns = final_data.select_dtypes(object).columns
#print()
final_data = pd.get_dummies(final_data, columns = cat_columns)
#drop duplicated columns
#columns were values are equal
final_data= final_data.reset_index().T.drop_duplicates().T.set_index('index')
#we reduce dataset  variables by checking correlations  with target column
corr_data = final_data.corrwith(final_data["ICU_SUM"])
#print(corr_data)
#select columns from correlation data with conditions
np_corr_data = np.array(corr_data)
columns = []
for i in np_corr_data:
  if(i):
    if(i>0.04):
      columns.append(True)
    elif(i<-0.02):
      columns.append(True)
    else:
      columns.append(False)
  else:
    columns.append(False)

#print(len(columns), columns.count(True))
selection = np.array(columns)
#print(selection)
selected_final_data = final_data.loc[:, selection]
selected_final_data.head()
#create x and y data
X_data = selected_final_data.drop(['ICU_SUM'], axis = 1)
Y_data = selected_final_data[['ICU_SUM']]
#print(X_data.shape)
#print(Y_data.shape)
#print(X_data.head((2)))

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.30, random_state=1)
#fit model
model =tree.DecisionTreeClassifier(criterion='entropy',max_depth=4,max_leaf_nodes=10)
model.fit(X_train,Y_train)
print(X_test.head(1))
y_pred=model.predict(X_test.head(1))
#y_pred = model.predict([[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.605263,150,-1.0,-0.968586,-0.414634,-1.0,0.357143,-0.842887,-0.742004,-0.940421,-0.891993,-0.052411,-0.073171,-0.932246,-0.9324,-0.640015,-0.821577,-0.691477,-0.56213,-0.351985,0.545455,-0.562083,-0.518519,0.654321,-0.371429,-0.995521,-0.989329,-0.879518,-0.980804,-0.056469,-0.503396,-0.071815,0.556717,-0.08642,-0.517241,-0.071429,0.631579,-0.237113,-0.375,-1.0,-0.025641,0.179104,-0.151515,0.362319,0.894737,-0.373913,-0.447853,-0.053435,0.959596,-0.415941,-0.305054,1.0,0.0,0.0,1.0,0.0,0.0,0.0]])
print(y_pred)
#accuracy, precision and recall
#print("Accuracy:{:.6f}".format(metrics.accuracy_score(Y_test, y_pred)))
#print("Precision:{:.6f}".format(metrics.precision_score(Y_test, y_pred)))
#print("Recall:{:.6f}".format(metrics.recall_score(Y_test, y_pred)))
#pickle.dump(model,open("model.pkl","wb")#y_pred = model.predict([[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.605263,150,-1.0,-0.968586,-0.414634,-1.0,0.357143,-0.842887,-0.742004,-0.940421,-0.891993,-0.052411,-0.073171,-0.932246,-0.9324,-0.640015,-0.821577,-0.691477,-0.56213,-0.351985,0.545455,-0.562083,-0.518519,0.654321,-0.371429,-0.995521,-0.989329,-0.879518,-0.980804,-0.056469,-0.503396,-0.071815,0.556717,-0.08642,-0.517241,-0.071429,0.631579,-0.237113,-0.375,-1.0,-0.025641,0.179104,-0.151515,0.362319,0.894737,-0.373913,-0.447853,-0.053435,0.959596,-0.415941,-0.305054,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]]))

