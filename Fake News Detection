#----------------------------------------Necessary Imports--------------------------------------------------------
import numpy as np
import pandas as pd
import itertools
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#----------------------------------------Reading Of Data--------------------------------------------------------

df=pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")

#----------------------------------------Initialising Of Labels--------------------------------------------------------

labels=df.label

#----------------------------------------Spliting Of dataset--------------------------------------------------------

x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=7)

#----------------------------------------Initialising Of TfidfVectorizer--------------------------------------------------------

tfidf_vectorizer=TfidfVectorizer(stop_words="english" , max_df=0.7)

#----------------------------------------Fit And Transform Train Set,Transfer Test Set-----------------------------------------

tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#----------------------------------------Initialising Of PassiveAggressiveClassifier--------------------------------------------

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#----------------------------------------Confusion Matrix--------------------------------------------------------

confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])

joblib_LR_model = joblib.load(joblib_file)
joblib_LR_model

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
