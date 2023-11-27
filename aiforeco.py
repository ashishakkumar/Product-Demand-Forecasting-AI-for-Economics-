import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import urllib.request
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import urllib.request
import datetime
import pandas_profiling
import joblib
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from streamlit_pandas_profiling import st_profile_report

from datetime import datetime, timedelta
import wget
import time

st.set_page_config(layout="wide")
url = "/Users/ashish/Documents/power_demand/multivariate_lstm.h5"
# filename = "multivariate_lstm.h5"
# urllib.request.urlretrieve(url, filename)



with st.sidebar:
    st.title("Product Demand Forecasting (AI for Economics)")
    choice = st.radio("Navigation", ["Home Page", "About data"], index=0)
    # st.info("This application allows you to predict the total electricity demand (in Million Units) for Andhra Pradesh")




# page_bg_img = f'''
# <style>
# [data-testid="stAppViewContainer"] > .main {{ 
# background-image: url("/Users/ashish/Documents/aiforeco/demand-forecasting.jpg");
# background-size: cover;
# }}
# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}
# [data-testid="stToolbar"] {{
# right: 2rem;
# }}
# </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)
# page_bg_img = '''
# <style>
# body {
#     background-image: url("https://cdn-gakoi.nitrocdn.com/UYXSDHrYcEMbBLuUZWqNsvzHQVDdOSSE/assets/images/optimized/rev-9a7d9bf/www.nuaig.ai/wp-content/uploads/2020/10/demand-forecasting-1-2048x1070.jpg");
#     background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

# original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Streamlit CSS Styling✨ </h1>'
# st.markdown(original_title, unsafe_allow_html=True)


# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://raw.githubusercontent.com/ashishakkumar/Product-Demand-Forecasting/main/demand-forecasting.webp");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)





def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')



if choice == 'Home Page' : 

    prep = pd.read_csv('data.csv')
    prep = prep.dropna()

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    

    text = "Welcome to this interface. We have forecasted the Product Demand for particular stores based on the Poblem Statement as described"
    speed = 8
    typewriter(text=text, speed=speed)

    train_to_scale = prep[['store_id','sku_id','total_price','base_price']]
    train_to_ohe = prep[['is_featured_sku','is_display_sku']]
    prep = prep.drop(['record_ID', 'store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku'], axis=1)
    prep = prep.groupby('week').sum()


    prep.index = pd.to_datetime(prep.index, format='%d/%m/%y')
    prep = prep.sort_index()

    prep['week_1'] = prep['units_sold'].shift(-1)
    prep['week_2'] = prep['units_sold'].shift(-2)
    prep['week_3'] = prep['units_sold'].shift(-3)
    prep['week_4'] = prep['units_sold'].shift(-4)
    prep['week_5'] = prep['units_sold'].shift(-5)
    prep['week_6'] = prep['units_sold'].shift(-6)
    prep['week_7'] = prep['units_sold'].shift(-7)

    df = prep.dropna()

    x1, x2, x3, x4, x5, x6, x7, y = df['week_1'], df['week_2'], df['week_3'], df['week_4'], df['week_5'], df['week_6'], df['week_7'], df['units_sold']
    x1, x2, x3, x4, x5, x6, x7, y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6), np.array(x7), np.array(y)
    x1, x2, x3, x4, x5, x6, x7, y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), x5.reshape(-1,1), x6.reshape(-1,1), x7.reshape(-1,1), y.reshape(-1,1)

    split_percentage = 15
    test_split = int(len(df)*(split_percentage/100))
    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7), axis=1)
    X_train,X_test,y_train,y_test = x[:-test_split],x[-test_split:],y[:-test_split],y[-test_split:]
        


    
    fig_2 = go.Figure()

    fig_2.add_trace(go.Scatter(x=prep.index, y=prep['units_sold'],
                             mode='lines', name='Total SKU Units Sold with time'))
    fig_2.update_layout(margin=dict(r=25))
    st.plotly_chart(fig_2, use_container_width=True)

    best_random_loaded = joblib.load('/Users/ashish/Documents/aiforeco/best_random_model.joblib')

    y_pred = best_random_loaded.predict(X_test)

    mape_7 = mean_absolute_percentage_error(y_test.ravel(), y_pred)


    st.write(f'MAPE on the validation dataset for the Random Forest Regressor with optimal parameters: {mape_7}')


     
    y_test_list = [i[0] for i in y_test ]
    y_pred_list = [i for i in y_pred ]
    pred_df = pd.DataFrame()
    pred_df['Actual Demand'] = y_test_list
    pred_df['Predicted Demand'] = y_pred_list

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_pred_list,
                           mode='lines', name='Total SKU Units Sold with time'))
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_test_list,
                            mode='lines', name='Total SKU Units Sold with time'))
    fig_3.update_layout(margin=dict(r=25))
    st.plotly_chart(fig_3, use_container_width=True)

    st.dataframe(pred_df.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))  # Set custom background and text color
    csv = convert_df(pred_df)
    # Add a download button for the dataframe
    st.download_button(
        label="Download Prediction as CSV File",
        data=csv,
        file_name='df.csv',
        mime='text/csv',
)








    








    # merged_df["Day"] = merged_df["Date"].dt.dayofweek
    # merged_df.drop("day", axis=1, inplace=True)
    # weather_df = pd.read_csv("/Users/ashish/Documents/power_demand/finWeatherData.csv",
    #                         index_col=0, parse_dates=True)
    # weather_df = weather_df.rename(columns={"date": "Date", "tmax": "Tmax", "tmin": "Tmin", "rain": "Rain"})

    # weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    # merged_df = pd.merge(merged_df, weather_df, on='Date', how='inner')
    # merged_df.drop(["rain", "temp"], axis=1, inplace=True)

    # light_colors = ['#F5F5F5', '#F0FFFF', '#F5FFFA', '#FAFAD2', '#F0FFF0', '#FFF5EE', '#F5F5DC', '#FAF0E6', '#F0E68C', '#FFFACD']

    # fig_1 = px.line(data_frame=merged_df, x="Date", y="Energy Required (MU)", title="Energy Demand (MU) trend till Date")
    # fig_1.update_xaxes(gridcolor="lightgray", gridwidth=0.5)
    # fig_1.update_yaxes(gridcolor="lightgray", gridwidth=0.5)
    # fig_1.update_layout(plot_bgcolor=light_colors[np.random.randint(0, 9)],paper_bgcolor = '#36454F')
    # fig_1.update_layout(margin=dict(r=25))


    # st.line_chart(merged_df, x="Date", y="Energy Required (MU)",use_container_width=True)

    # columns = ['Energy Required (MU)', 'Rain', 'Tmax', 'Tmin', 'inflation', 'Day']
    # merged_df = merged_df.loc[:, columns]

    # X = merged_df.values
    # y = merged_df['Energy Required (MU)'].values
    # y = y.reshape(-1, 1)

    # #defining the end indices of train validation and test set
    # train_end_idx = 1202
    # cv_end_idx = 1367
    # test_end_idx = 1565

    # scaler_X = MinMaxScaler(feature_range=(0, 1))
    # scaler_y = MinMaxScaler(feature_range=(0, 1))

    # scaler_X.fit(X[:train_end_idx])
    # scaler_y.fit(y[:train_end_idx])

    # X_norm = scaler_X.transform(X)
    # y_norm = scaler_y.transform(y)

    # dataset_norm = np.concatenate((X_norm, y_norm), axis=1)
    # n_features = 6
    # past_history = 60
    # future_target = 30

    # X_train, y_train = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, -1],
    #                                     0, train_end_idx, past_history, 
    #                                     future_target, step=1, single_step=False)

    # X_val, y_val = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, 0],
    #                                 train_end_idx, cv_end_idx, past_history, 
    #                                 future_target, step=1, single_step=False)

    # X_test, y_test = multivariate_data(dataset_norm[:, 0:-1], dataset_norm[:, 0],
    #                                 cv_end_idx, test_end_idx, past_history, 
    #                                 future_target, step=1, single_step=False)

    # batch_size = 32
    # buffer_size = 1184

    # train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train = train.cache().shuffle(buffer_size).batch(batch_size).prefetch(1)

    # validation = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    # validation = validation.batch(batch_size).prefetch(1)



    # y_test_inv = scaler_y.inverse_transform(y_test)
    # test_forecast = multivariate_lstm.predict(X_test)
    # lstm_forecast = scaler_y.inverse_transform(test_forecast)
    # rmse_lstm = sqrt(mean_squared_error(y_test_inv, lstm_forecast))


    # train_forecast = multivariate_lstm.predict(X_train)
    # train_forecast_inverse = scaler_y.inverse_transform(train_forecast)

    # valid_forecast = multivariate_lstm.predict(X_val)
    # valid_forecast_inverse = scaler_y.inverse_transform(valid_forecast)

    # start_idx = X_train.shape[0] + X_val.shape[0] + X_test.shape[0] + 1
    # end_idx = start_idx + past_history
    # X_test2 = scaler_X.transform(X[start_idx:end_idx, :])
    # X_test2 = X_test2.reshape(1, past_history, n_features)
    # forecast_unseen = multivariate_lstm.predict(X_test2)
    # forecast_unseen_inverse = scaler_y.inverse_transform(forecast_unseen)[0]
    # date_range = pd.date_range(start=start_date, periods=(end_date -start_date).days+1, freq='D')
    # # Create an empty DataFrame
    # df = pd.DataFrame()

    # # Add the Date column to the DataFrame
    # df['Date'] = date_range
    # df["Date"] = pd.to_datetime(df["Date"])
    # df['Forecasted Demand (MU)'] = forecast_unseen_inverse[(start_date-last_date).days:(end_date-last_date).days+1]
    # fig = px.line(df, x='Date', y='Forecasted Demand (MU)',title=f"Forecasted Demand pattern for {start_date} to {end_date}")
    # fig.update_layout(paper_bgcolor = '#36454F',plot_bgcolor=light_colors[np.random.randint(0, 9)])
    # fig.update_layout(margin=dict(r=25))
    # # Render the chart using Streamlit


    
    # Convert the dataframe to CSV format

#     def convert_df(df):
#         return df.to_csv(index=False).encode('utf-8')

#     csv = convert_df(df)


#     st.markdown("_Tips : Click on the same date twice in the calendar interface to see the prediction for single day_")
#     st.markdown(f"You have selected: **{start_date}** to **{end_date}** for prediction.")
#     st.plotly_chart(fig_1, use_container_width=True) 
#     st.write(f"RMSE of model under use for LSTM forecast of next 30 days: {round(rmse_lstm, 3)}")
#     if selected_date is not None:
#         st.plotly_chart(fig ,use_container_width=True)

#     with st.container():
# # Show the dataframe in Streamlit


#         st.dataframe(df.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))  # Set custom background and text color

#     # Add a download button for the dataframe
#     st.download_button(
#         label="Download Prediction as CSV File",
#         data=csv,
#         file_name='df.csv',
#         mime='text/csv',
# )



if choice == "About data" :
    st.title("Automated Exploratory Data Analysis")
    prep = pd.read_csv('data.csv')
    prep = prep.dropna()
    profile_report = prep.profile_report()
    st_profile_report(profile_report)


# if choice == 'XGBoost' : 
#     #min_date = datetime(2015, 1, 1)
#     date_str = "2023-07-14"
#     last_date = datetime.strptime(date_str, "%Y-%m-%d").date()
#     # Get user input for date
#     selected_date = st.date_input(
#         "Select a date for prediction",value=(last_date)
        
#     )

    # if selected_date is not None:
    #     start_date = selected_date
    # selected_date = pd.to_datetime(selected_date)
    # pjme = pd.read_csv('total_demand_weather (1).csv', index_col=[0])
    # pjme['Date'] = pd.to_datetime(pjme['Date'])
    # pjme.Holiday = pjme.Holiday.apply(lambda x : np.random.randint(100,200)/1000 if x == 0 else 1)
    # pjme = pjme.set_index('Date')
    # pjme['weekend'] = (pjme.index.weekday >= 5).astype(int) 
    # lag_steps = 7

    # for i in range(1, lag_steps + 1):
    #     column_name = f'Energy Required (MU) (t-{i})'
    #     pjme[column_name] = pjme['Energy Required (MU)'].shift(i)

    # # Drop rows with NaN values resulting from shifting
    # pjme = pjme.dropna()
    # pjme = pjme.reset_index()
    # pjme = pjme[pjme['Date']< '2023-08-01']
    # pjme = pjme.set_index('Date')
    # split_date = '2022-12-16'
    # start_date = selected_date + timedelta(days=15)
    # pjme_test = pjme.loc[(pjme.index >= selected_date )& (pjme.index < start_date) ].copy()
    # pjme_train = pjme.loc[pjme.index < split_date].copy()
    # def create_features(df, label=None):
    #     """
    #     Creates time series features from datetime index
    #     """
    #     df['date'] = df.index
    #     df['hour'] = df['date'].dt.hour
    #     df['dayofweek'] = df['date'].dt.dayofweek
    #     df['quarter'] = df['date'].dt.quarter
    #     df['month'] = df['date'].dt.month
    #     df['year'] = df['date'].dt.year
    #     df['dayofyear'] = df['date'].dt.dayofyear
    #     df['dayofmonth'] = df['date'].dt.day    
    #     X = df.drop([label,'date'], axis = 1)
    #     if label:
    #         y = df[label]
    #         return X, y
    #     return X
    # X_test, y_test = create_features(pjme_test, label='Energy Required (MU)')
    # X_train, y_train = create_features(pjme_train, label='Energy Required (MU)')

    # model = joblib.load('/Users/ashish/Documents/power_demand/xgb_model.pkl')
    # feature_importance = model.feature_importances_
    # sorted_idx = np.argsort(feature_importance)
   

    # pjme_test['MW_Prediction'] = model.predict(X_test.values)

    # pjme_all = pd.concat([pjme_test, pjme_train], sort=False)

  

    # def mean_absolute_scaled_error(y_true, y_pred, y_train):
    #     e_t = y_true - y_pred
    #     scale = mean_absolute_error(y_train[1:], y_train[:-1])
    #     return np.mean(np.abs(e_t / scale))



    # def mean_absolute_percentage_error(y_true, y_pred): 
    #     """Calculates MAPE given y_true and y_pred"""
    #     y_true, y_pred = np.array(y_true), np.array(y_pred)
    #     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    # fig_2 = go.Figure()

    # fig_2.add_trace(go.Scatter(x=pjme_test.index, y=pjme_test['Energy Required (MU)'],
    #                         mode='lines', name='True Energy Demand'))
    # fig_2.add_trace(go.Scatter(x=pjme_test.index, y=pjme_test['MW_Prediction'],
    #                         mode='lines', name='Predicted Energy Demand'))

    # fig_2.update_layout(title='True Demand vs Predicted Demand (Million Units)',
    #                 xaxis_title='Index',
    #                 yaxis_title='Value')

    
    # fig_2.update_layout(margin=dict(r=25))
    # st.plotly_chart(fig_2, use_container_width=True) 
    # st.write('MSE : '+ str(mean_squared_error(y_true=pjme_test['Energy Required (MU)'],
    #                 y_pred=pjme_test['MW_Prediction'],squared = False)))
    # st.write('MASE : '+ str(mean_absolute_scaled_error(pjme_test['Energy Required (MU)'], pjme_test['MW_Prediction'], y_train)))

    # st.write('MAE : '+ str(mean_absolute_error(y_true=pjme_test['Energy Required (MU)'],
    #                 y_pred=pjme_test['MW_Prediction'])))
    
    # st.write("MAPE : "+ str(mean_absolute_percentage_error(y_true=pjme_test["Energy Required (MU)"],y_pred=pjme_test["MW_Prediction"])))
    #st.markdown(f'<h2 style="background-color:#000000;color:#999999;font-size:15px;">{"MAPE : "+ str(mean_absolute_percentage_error(y_true=pjme_test["Energy Required (MU)"],y_pred=pjme_test["MW_Prediction"]))}</h2>', unsafe_allow_html=True)
#     pjme_all = pjme_all.reset_index().dropna()
    
#     def convert_df(df):
#         return df.to_csv(index=False).encode('utf-8')
#     pjme_test = pjme_test.reset_index()
#     df_1 = pd.DataFrame(pjme_all[['Date','Energy Required (MU)','MW_Prediction']])
#     df_1.columns =['Date','True Demand','Predicted Demand'] 
#     with st.container():
# # Show the dataframe in Streamlit
#         st.dataframe(df_1.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'})) 
#     csv_1 = convert_df(df_1)
#     st.download_button(
#     label="Download Prediction as CSV File",
#     data= csv_1,
#     file_name='xgboost.csv',
#     mime='text/csv',
#     )


