
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Advertising App")
st.write("This app predicts the **Sales** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    tv = st.sidebar.slider('TV', 0, 300, 147)
    radio = st.sidebar.slider('Radio', 0, 50, 25)
    newspaper = st.sidebar.slider('Newspaper', 0, 115, 30)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('Advertising')
X = data.drop(['sales'],axis=1)
Y = data.sales.copy()

modelGaussianAdvertising = GaussianNB()
modelGaussianAdvertising.fit(X, Y)

prediction = modelGaussianAdvertising.predict(df)
prediction_proba = modelGaussianAdvertising.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
