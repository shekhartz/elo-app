import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import s3fs
import os

st.title("ELO Loyalty Score Prediction")
#--------------------------------------------------------------------------------------------------
DATA_URL = ('https://elo-stream.s3.us-east-2.amazonaws.com/train_FE2.csv')


def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data

data_load_state = st.text('Loading data...')
data = load_data(1000)
st.text_input(data.shape)
#--------------------------------------------------------------------------------------------------

MODEL_URL = ('lgb_kfold_model.sav')
nrows = 1000
model = pickle.load(open(MODEL_URL, 'rb'))

#--------------------------------------------------------------------------------------------------
def checkValid(cardID):
  card_details = data.loc[data['card_id'] == cardID]
  row_count = card_details.shape[0]
  return row_count

#--------------------------------------------------------------------------------------------------
def predict(cardID):
  card_details = data.loc[data['card_id'] == cardID]
  y = card_details['target'].values
  x = card_details.drop(['card_id','first_active_month','target','outliers'], axis=1)
  y_pred = model.predict(x)
  rmse = mean_squared_error(y_pred, y)**0.5
  return rmse

#--------------------------------------------------------------------------------------------------
def main():
    cardID = st.text_input("Enter Card ID")
    result = ""

    if st.button("Check Loyalty Score"):
        row_count = checkValid(cardID)
        if row_count > 0:
           result = predict(cardID)
           st.success('RMSE is {}'.format(result))
        else:
           st.error('Please enter a valid Card ID')


    if st.checkbox('Show Sample Card ID'):
        st.subheader('You can choose from below sample Card IDs')
        st.write(data['card_id'][:10])

if __name__=='__main__':
    main()
