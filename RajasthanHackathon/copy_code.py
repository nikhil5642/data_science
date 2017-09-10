import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import requests
import scrap
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
#from sklearn.metrics import accuracy_score
import xgboost as xgb
color = sns.color_palette()
from sklearn.metrics import accuracy_score
from flask import Flask
from flask import jsonify
from flask import request


import sklearn

app=Flask(__name__)

def getIndex(arr,str):
    j=0
    for i in range(len(arr)):
        if arr[i]==str:
           j=i 
    return j      

#page = requests.get("http://164.100.222.56/amb/1/mandishowtoday.asp")
#page=requests.get("http://0.0.0.0:8000/scrap.html")
app=Flask(__name__)
city=pd.read_csv('scv.csv')
city=city.iloc[:,1:]
json=city.iloc[:,:].to_json(orient='records')
sa={"city":"Damoh"}
  
 
train = pd.read_csv('scv.csv')
train=train.iloc[:,:9]
train=train.dropna()
#train.head()
train1 =train
train2=train.drop(['District_Name',], axis=1)

train2['P_r'] = train2.Production/train2.Area
train2=train2.drop(['Area', 'Production'], axis=1)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l_state=LabelEncoder()
l_state.fit(train2.State_Name)
state_enc=(list(l_state.classes_))
state_enc=np.array(state_enc)
#df_state=pd.DataFrame(state_enc)
#print(df_state)
train2.State_Name = l_state.transform(train2.State_Name)

l_crop=LabelEncoder()
l_crop.fit(train2.Crop)
enc=(list(l_crop.classes_))
enc_crop=np.array(enc)
#df_state=pd.DataFrame(enc)

train2.Crop = l_crop.transform(train2.Crop)

l=LabelEncoder()
l.fit(train2.Season)
enc=(list(l.classes_))
enc_season=np.array(enc)

train2.Season = l.transform(train2.Season)

l_year=LabelEncoder()
l_year.fit(train2.Crop_Year)
enc=(list(l_year.classes_))
enc_year=np.array(enc)
#df_year=pd.DataFrame(enc)
train2.Crop_Year = l_year.transform(train2.Crop_Year)

#train2

y_train = train2.Crop
x_train = train2.drop(labels= ['Crop'], axis=1)
#print y_train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=.1)
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

xgb.plot_importance(model, height=0.5)

num_boost_round = model.best_iteration
xgb.plot_importance(model,  height=0.5)

from xgboost.sklearn import XGBRegressor
xgb = XGBRegressor( nthread=-1,  missing= -1, n_estimators=300, learning_rate=0.02, max_depth=17, subsample=0.9
                   , min_child_weight=3, colsample_bytree=0.7, reg_alpha=100, reg_lambda=100, silent=False)
xgb.fit(x_train,y_train)
#print(x_train)
pred=xgb.predict(x_test)
predictions = [round(value) for value in pred]
"""x_test['result']=pred
x_test['crop']=y_test
x_test.to_csv('pred.csv')"""
#print accuracy_score(y_test,pred)
accuracy = accuracy_score(y_test, predictions)
  
print("Accuracy: %.2f%%" % (accuracy * 100.0))
@app.route('/predictor',methods=['POST','GET'])
def predictor():
   data=request.get_json(force=True)
   a=str(data.get("rain"))
   b=str(data.get("temperature"))
   c=str(data.get("season"))
   #c=data.get("humidity")
   d=str(data.get("state"))
   e=str(data.get("year"))
   f=str(data.get("P_r"))
   array=[]
   array.append([getIndex(state_enc,d),getIndex(enc_year,e),getIndex(enc_season,c),float(b),float(a),float(6)])
   array=np.array(array)
   df=pd.DataFrame(array,columns=['State_Name','Crop_Year','Season','temperature','Rainfall','P_r'])
   prediction=xgb.predict(df) 
   prediction=np.round(prediction)
   prediction=int(prediction)
   ans=enc_crop[prediction]
   ans=[ans]
   ans=np.array(ans)
   df=pd.DataFrame(ans,columns=['crop'])
   return df.to_json(orient='records')
    
if __name__ == '__main__':
 app.run(host='172.19.13.149')
   
