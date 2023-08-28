import streamlit as st
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import plotly.express as px

st.title('My ML App')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  
  categorical = [var for var in df.columns if df[var].dtype=='O']
  numerical = [var for var in df.columns if df[var].dtype!='O']
  
  st.write('Categorical Variables:', categorical)
  st.write('Numerical Variables:', numerical)
    
  label_encoder = LabelEncoder()
  for var in categorical:
    df[var] = label_encoder.fit_transform(df[var])
      
  feature_options = st.multiselect('Select Feature Columns', df.columns)
  label_options = st.multiselect('Select Label Column', df.columns)

  test_size = st.slider('Test Split %', min_value=10, max_value=90, value=20)

  problem_type = st.radio('Regression or Classification', ('Regression', 'Classification'))

  if problem_type=='Regression':
    models = st.multiselect('Select Models', ['Linear Regression', 'Polynomial Regression', 'Ridge Regression',
                                              'Lasso Regression', 'SVR', 'Decision Tree Regressor', 
                                              'KNN Regressor', 'Random Forest Regressor',
                                              'Gradient Boost Regressor', 'XGBoost Regressor'])
      
    if st.button('Run Models'):
      X = df[feature_options]
      y = df[label_options]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
      reg_models = []
      model_metrics = {}
      for model in models:
        if model=='Linear Regression':
          reg = LinearRegression()
        elif model=='Polynomial Regression':
          reg = PolynomialFeatures(degree=2)
        elif model=='Ridge Regression':
          reg = Ridge() 
        elif model=='Lasso Regression':
          reg = Lasso()
        elif model=='SVR':
          reg = SVR()
        elif model=='Decision Tree Regressor':
          reg = DecisionTreeRegressor()
        elif model=='KNN Regressor':
          reg = KNeighborsRegressor()
        elif model=='Random Forest Regressor':
          reg = RandomForestRegressor(n_estimators=100)
        elif model=='Gradient Boost Regressor':
          reg = GradientBoostingRegressor(n_estimators=100)
        elif model=='XGBoost Regressor':
          reg = XGBRegressor(n_estimators=100)
          
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        model_metrics[model] = {'MSE': mse, 'R2 Score': r2}
        
      df_metrics = pd.DataFrame(model_metrics).T  
      st.write(df_metrics)
      
      fig = px.bar(df_metrics, x=df_metrics.index, y=['MSE', 'R2 Score'])
      st.plotly_chart(fig)
      
      best_model = df_metrics['MSE'].idxmin()
      st.write('Best Model:', best_model)
      
  if problem_type=='Classification':
    models = st.multiselect('Select Models', ['Logistic Regression', 'SVC','KNN Classifier', 
                                              'Decision Tree Classifier', 'Random Forest Classifier',
                                              'Naive Bayes', 'Gradient Boost Classifier',
                                              'XGBoost Classifier'])

    if st.button('Run Models'):
      X = df[feature_options]  
      y = df[label_options]

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
      
      classifiers = []
      model_metrics = {}
      for model in models:
        if model=='Logistic Regression':
          clf = LogisticRegression()
        elif model=='SVC':
          clf = SVC()
        elif model=='KNN Classifier':
          clf = KNeighborsClassifier()
        elif model=='Decision Tree Classifier':
          clf = DecisionTreeClassifier()
        elif model=='Random Forest Classifier':
          clf = RandomForestClassifier(n_estimators=100)
        elif model=='Naive Bayes':
          clf = GaussianNB()
        elif model=='Gradient Boost Classifier':
          clf = GradientBoostingClassifier(n_estimators=100)
        elif model=='XGBoost Classifier':
          clf = XGBClassifier(n_estimators=100)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')  
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
      
        model_metrics[model] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        
      df_metrics = pd.DataFrame(model_metrics).T
      st.write(df_metrics)
      
      fig = px.bar(df_metrics, x=df_metrics.index, y=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
      st.plotly_chart(fig)
      
      best_model = df_metrics['Accuracy'].idxmax()
      st.write('Best Model:', best_model)