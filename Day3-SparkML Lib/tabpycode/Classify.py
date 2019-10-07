from tabpy_tools.client import Client
client = Client('http://localhost:9004/')

def CheckIrisClassify(s_len,s_width,p_len,p_width):
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import sklearn.preprocessing
	from sklearn.linear_model import LogisticRegression
	from sklearn import model_selection
	from sklearn import metrics

	from sklearn.preprocessing import LabelEncoder
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	
	df= pd.read_csv(url, names=names)
	encoder=LabelEncoder()
	df['class']=encoder.fit_transform(df['class'])

	X=df.iloc[:,0:4]
	Y=np.array(df['class'])

	x_train,x_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.2,random_state=2)
	
	log_reg=LogisticRegression()
	log_reg.fit(x_train,y_train)
 
	y_pred=log_reg.predict(x_test)

	a=np.array([s_len,s_width,p_len,p_width])
	y_tableau_data=pd.DataFrame(a)
	print(y_tableau_data.shape)
	y_tableau_data_tran=y_tableau_data.T
	print(y_tableau_data_tran.shape)
	y_p=log_reg.predict(y_tableau_data_tran)  
	return (encoder.inverse_transform(log_reg.predict(y_tableau_data_tran)).tolist())

client.deploy('IrisClassification_June15',CheckIrisClassify,'Sample python application for connecting tableau with Python',override=True)