import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import urllib.request
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import urllib.request
import datetime
# import pandas_profiling
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
    choice = st.radio("Navigation", ["Home Page", "Top 10 Stores","Top 10 SKUs"], index=0)
    st.info("This application allows you to visualise and predict the total product demand for different stores and SKUs")




background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://th.bing.com/th/id/OIG.wZ2quBnSMXZhFr2dbIHB?pid=ImgGn");
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

def top10(top_10_stores,column):
  top_10_stores_dfs = []
  for i in range(top_10_stores[column].shape[0]) : 
      store = prep[prep[column]==top_10_stores[column][i]]
      store['week'] = pd.to_datetime(store['week'])
      store = store.sort_values(by = 'week')
      store = store.groupby('week').sum(numeric_only=True) 
      top_10_stores_dfs.append(store)
  return top_10_stores_dfs

def new_cols(prep) :
    prep['week_1'] = prep['units_sold'].shift(-1)
    prep['week_2'] = prep['units_sold'].shift(-2)
    prep['week_3'] = prep['units_sold'].shift(-3)
    prep['week_4'] = prep['units_sold'].shift(-4)
    prep['week_5'] = prep['units_sold'].shift(-5)
    prep['week_6'] = prep['units_sold'].shift(-6)
    prep['week_7'] = prep['units_sold'].shift(-7)
    prep = prep.dropna()
    return prep

def split_data(df):
      x1, x2, x3, x4, x5, x6, x7,x8, y = df['week_1'], df['week_2'], df['week_3'], df['week_4'], df['week_5'], df['week_6'], df['week_7'],df['total_price'], df['units_sold']
      x1, x2, x3, x4, x5, x6, x7,x8, y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6), np.array(x7), np.array(x8), np.array(y)
      x1, x2, x3, x4, x5, x6, x7,x8, y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), x5.reshape(-1,1), x6.reshape(-1,1), x7.reshape(-1,1),x8.reshape(-1,1), y.reshape(-1,1)

      split_percentage = 15
      test_split = int(len(df)*(split_percentage/100))
      x = np.concatenate((x1, x2, x3, x4, x5, x6, x7,x8), axis=1)
      X_train,X_test,y_train,y_test = x[:-test_split],x[-test_split:],y[:-test_split],y[-test_split:]

      return [X_train,X_test,y_train,y_test]



if choice == 'Home Page' : 


    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    

    text = "Welcome to this interface. We have forecasted the Product Demand for Various Stores and Various SKUs"
    speed = 5
    typewriter(text=text, speed=speed)

    text_1 = "This page demonstrates Exploratory Data Analysis about the data being used"
    speed_1 = 5
    typewriter(text=text_1, speed=speed_1)


    prep = pd.read_csv('data.csv')
    prep = prep.dropna()

    total_units_sold = prep.groupby('store_id')['units_sold'].sum().reset_index()
    top_10_stores = total_units_sold.nlargest(10, 'units_sold').reset_index(drop =True)
    fig = px.pie(top_10_stores, values='units_sold', names='store_id', title='Top 10 Stores - Total Units Sold',
             template='plotly_white', hole=0.4)
    fig.update_layout(
        showlegend=True,
        annotations=[dict(text='Total Units Sold', x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    st.plotly_chart(fig)

    text_2= "The above pie chart shows the top 10 stores selling Maximum Products"
    speed_2 = 5
    typewriter(text=text_2, speed=speed_2)

    total_units_sold_sku = prep.groupby('sku_id')['units_sold'].sum().reset_index()
    total_units_sold_sku = total_units_sold_sku.sort_values(by = 'units_sold',ascending = False).reset_index(drop = True)
    fig = go.Figure()

    fig.add_trace(go.Pie(
    labels=[str(i) for i in total_units_sold_sku['sku_id'].tolist()],
    values=total_units_sold_sku['units_sold'].tolist(),
    hole=0.3,  # Hole size for the donut effect
    marker_colors=['blue', 'green', 'orange'],  # You can change the colors if needed
    ))

    fig.update_layout(title='Total Units Sold by SKU ID')
    st.plotly_chart(fig)

    text_3= "The above pie chart shows the proportions of SKUs being sold"
    speed_3 = 5
    typewriter(text=text_3, speed=speed_3)

    top_10_stores_dfs = top10(top_10_stores,'store_id')
    new_top_10_stores_dfs = [new_cols(i) for i in top_10_stores_dfs ]




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

    text_4= "The above line chart shows the total SKU Units Sold with time. We can observe that the SKUs sold are highly fluctuating, but there seems to be seasonality"
    speed_4 = 6
    typewriter(text=text_4, speed=speed_4)

    best_random_loaded = joblib.load('best_random_model.joblib')

    y_pred = best_random_loaded.predict(X_test)

    mape_7 = mean_absolute_percentage_error(y_test.ravel(), y_pred)

    text_5= "We did hyperparameter tuning on the Random Forest Model and used Mean Absolute Percentage Error (MAPE) for evaluating our results"
    speed_5 = 6
    typewriter(text=text_5, speed=speed_5)

    text_6= f'MAPE on the validation dataset for the Random Forest Regressor with optimal parameters: {mape_7}'
    speed_6 = 6
    typewriter(text=text_6, speed=speed_6)


    


     
    y_test_list = [i[0] for i in y_test ]
    y_pred_list = [int(i)for i in y_pred ]
    pred_df = pd.DataFrame()
    pred_df['Actual Demand'] = y_test_list
    pred_df['Predicted Demand'] = y_pred_list

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_pred_list,
                           mode='lines', name='Total SKU Units Sold [Predicted Values]'))
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_test_list,
                            mode='lines', name='Total SKU Units Sold [Actual Values]'))
    fig_3.update_layout(margin=dict(r=25))
    st.plotly_chart(fig_3, use_container_width=True)

    text_7= "The above plot shows the relation between the actual and predicted values. Although they are not so close, but can be improved having access to more data"
    speed_7 = 6
    typewriter(text=text_7, speed=speed_7)

    st.dataframe(pred_df.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))  # Set custom background and text color
    csv = convert_df(pred_df)
    # Add a download button for the dataframe
    st.download_button(
        label="Download Prediction as CSV File",
        data=csv,
        file_name='df.csv',
        mime='text/csv',
)





if choice == "Top 10 Stores" :

    prep = pd.read_csv('data.csv')
    prep = prep.dropna()
    total_units_sold = prep.groupby('store_id')['units_sold'].sum().reset_index()
    top_10_stores = total_units_sold.nlargest(10, 'units_sold').reset_index(drop =True)
    top_10_stores_dfs = top10(top_10_stores,'store_id')
    new_top_10_stores_dfs = [new_cols(i) for i in top_10_stores_dfs ]
    spiltted_list = [split_data(i) for  i in new_top_10_stores_dfs ]

    selected_store_index = st.selectbox('Select Store', range(1, 11)) - 1  # Subtract 1 to convert 1-index to 0-index

    # Display selected store name
    selected_store_name = f'Store {selected_store_index + 1}'

    st.write(f'You selected: {selected_store_name}')

    # Plot the selected store's data
    selected_dataframe = top_10_stores_dfs[int(selected_store_index)]
    selected_dataframe = selected_dataframe.drop(['store_id','sku_id','is_featured_sku','is_display_sku'],axis =1)

    # Show the dataframe for the selected store (optional)
    st.write(f'{selected_store_name} DataFrame:', selected_dataframe)

    fig_2 = go.Figure()

    fig_2.add_trace(go.Scatter(x=selected_dataframe.index, y=selected_dataframe['units_sold'],
                             mode='lines', name='Units Sold'))
    fig_2.update_layout(title_text = f'Units sold over time for {selected_store_name}')
    fig_2.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='#001324',  
)
    st.plotly_chart(fig_2)


    fig_3 = go.Figure()

    fig_3.add_trace(go.Scatter(x=selected_dataframe.index, y=selected_dataframe['total_price'],
                             mode='lines', name='Revenue'))
    fig_3.update_layout(title_text = f'Revenue from {selected_store_name}')
    fig_3.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='#002408',  
)
    st.plotly_chart(fig_3)

    path = f'store{selected_store_index + 1}.joblib'

    X_train = spiltted_list[selected_store_index][0]
    y_train = spiltted_list[selected_store_index][2]
    X_test = spiltted_list[selected_store_index][1]
    y_test = spiltted_list[selected_store_index][3]

    best_random_loaded = joblib.load(path)

    y_pred = best_random_loaded.predict(X_test)

    mape_7 = mean_absolute_percentage_error(y_test.ravel(), y_pred)


    st.write(f'MAPE on the validation dataset for the Random Forest Regressor with optimal parameters: {mape_7}')


     
    y_test_list = [i[0] for i in y_test ]
    y_pred_list = [int(i) for i in y_pred ]
    pred_df = pd.DataFrame()
    pred_df['Actual Demand'] = y_test_list
    pred_df['Predicted Demand'] = y_pred_list

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_pred_list,
                           mode='lines', name=f'Total SKU Units to be sold (Predicted) in {selected_store_name}'))
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_test_list,
                            mode='lines', name=f'Total SKU Units Sold with time (Actual) in {selected_store_name}'))
    fig_3.update_layout(margin=dict(r=25))

    st.plotly_chart(fig_3, use_container_width=True)

    st.dataframe(pred_df.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))  # Set custom background and text color
    csv = convert_df(pred_df)

    st.download_button(
        label="Download Prediction as CSV File",
        data=csv,
        file_name='df.csv',
        mime='text/csv',
)

if choice == "Top 10 SKUs" :

    prep = pd.read_csv('data.csv')
    prep = prep.dropna()
    total_units_sold = prep.groupby('sku_id')['units_sold'].sum().reset_index()
    top_10_stores = total_units_sold.nlargest(10, 'units_sold').reset_index(drop =True)
    top_10_stores_dfs = top10(top_10_stores,'sku_id')
    new_top_10_stores_dfs = [new_cols(i) for i in top_10_stores_dfs ]
    spiltted_list = [split_data(i) for  i in new_top_10_stores_dfs ]

    selected_store_index = st.selectbox('Select SKU', range(1, 11)) - 1  # Subtract 1 to convert 1-index to 0-index

    # Display selected store name
    selected_store_name = f'SKU {selected_store_index + 1} '

    st.write(f'You selected: {selected_store_name}')

    # Plot the selected store's data
    selected_dataframe = top_10_stores_dfs[int(selected_store_index)]
    selected_dataframe = selected_dataframe.drop(['store_id','sku_id','is_featured_sku','is_display_sku'],axis =1)
    

    # Show the dataframe for the selected store (optional)
    st.write(f'{selected_store_name} DataFrame:', selected_dataframe)

    fig_2 = go.Figure()

    fig_2.add_trace(go.Scatter(x=selected_dataframe.index, y=selected_dataframe['units_sold'],
                             mode='lines', name='Units Sold'))
    fig_2.update_layout(title_text = f'Units sold over time for {selected_store_name}')
    fig_2.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='#001324',  
)
    st.plotly_chart(fig_2)


    fig_3 = go.Figure()

    fig_3.add_trace(go.Scatter(x=selected_dataframe.index, y=selected_dataframe['total_price'],
                             mode='lines', name='Revenue'))
    fig_3.update_layout(title_text = f'Revenue from {selected_store_name}')
    fig_3.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='#002408',  
)
    st.plotly_chart(fig_3)

    path = f'sku{selected_store_index + 1}.joblib'

    X_train = spiltted_list[selected_store_index][0]
    y_train = spiltted_list[selected_store_index][2]
    X_test = spiltted_list[selected_store_index][1]
    y_test = spiltted_list[selected_store_index][3]

    best_random_loaded = joblib.load(path)

    y_pred = best_random_loaded.predict(X_test)

    mape_7 = mean_absolute_percentage_error(y_test.ravel(), y_pred)


    st.write(f'MAPE on the validation dataset for the Random Forest Regressor with optimal parameters: {mape_7}')


     
    y_test_list = [i[0] for i in y_test ]
    y_pred_list = [int(i) for i in y_pred ]
    pred_df = pd.DataFrame()
    pred_df['Actual Demand'] = y_test_list
    pred_df['Predicted Demand'] = y_pred_list

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_pred_list,
                           mode='lines', name=f'Total SKU Units to be sold (Predicted) in {selected_store_name}'))
    fig_3.add_trace(go.Scatter(x=prep.index, y=y_test_list,
                            mode='lines', name=f'Total SKU Units Sold with time (Actual) in {selected_store_name}'))
    fig_3.update_layout(margin=dict(r=25))

    st.plotly_chart(fig_3, use_container_width=True)

    st.dataframe(pred_df.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))  # Set custom background and text color
    csv = convert_df(pred_df)

    st.download_button(
        label="Download Prediction as CSV File",
        data=csv,
        file_name='df.csv',
        mime='text/csv',
)



