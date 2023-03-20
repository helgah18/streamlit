#importing necessary packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.cluster import KMeans
import matplotlib.colors as colors
from sklearn.datasets import make_classification
import snowflake.connector 
import plotly.express as px
import plotly.colors as pc
import streamlit as st

st.sidebar.header("Clustering")

st.title('Clustering Airbnb listings in Copenhagen')


# Connect to Snowflake
def init_connection():
    return snowflake.connector.connect(
        **st.secrets["snowflake"],
        client_session_keep_alive=True
    )

conn = init_connection()

# Retrieve data from Snowflake table
query = 'SELECT * FROM AIRBNB_DB.PUBLIC.AIRBNB'
data = pd.read_sql_query(query, conn)

# Close the Snowflake connection
conn.close()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data.T)

#data=data.dropna()


#Data cleaning
# Making the target data with low, medium and high
percentile_33 = np.percentile(data['PRICE'], 33)
percentile_66 = np.percentile(data['PRICE'], 66)


data['PRICE_CLASS'] = data['PRICE'].apply(lambda x: 2 if x > percentile_66 else (0 if x < percentile_33 else 1))



cols = ['PRICE_CLASS', 'LATITUDE', 'LONGITUDE', 
         'CALCULATED_HOST_LISTINGS_COUNT', 'AVAILABILITY_365', 'NUMBER_OF_REVIEWS_LTM', 'NUMBER_OF_REVIEWS']
cluster_col = st.multiselect(
    "Pick the columns you want to cluster by",
    cols)

number = st.number_input('Input number of clusters', min_value=1, max_value=10, value=5, step=1)
st.write('The current number is ', number)

#data = pd.get_dummies(data, columns=['NEIGHBOURHOOD', 'ROOM_TYPE'])

#if 'NEIGHBOURHOOD' in cluster_col:
#    cluster_col.remove('NEIGHBOURHOOD')
#    neigh_cols = [col for col in data if col.startswith('NEIGHBOURHOOD')]
#    cluster_col = pd.concat([cluster_col, neigh_cols])



if cluster_col: 
        # Generate a list of colors with length equal to the number of clusters
    colors = pc.qualitative.Plotly * ((number // len(pc.qualitative.Plotly)) + 1)

    # Create a dictionary that maps each cluster to a color
    color_map = {f"cluster{i}": colors[i] for i in range(number)}
    km = KMeans(n_clusters=number) 

    km.fit(data[cluster_col])
    # Create a new column with the cluster labels
    data['cluster'] = km.labels_

    # Create the scatter plot
    fig = px.scatter_mapbox(data, lat="LATITUDE", lon="LONGITUDE", color="cluster",
                            hover_data=["NAME", "PRICE"],
                            mapbox_style="carto-positron", zoom=10, height=500,
                            color_discrete_map=color_map,
                            color_discrete_sequence=colors)

    # Add a color scale
    fig.update_layout(coloraxis=dict(colorbar=dict(title='Clusters')))#, margin={"r":0,"t":0,"l":0,"b":0}))
    st.subheader('Clustering')
    st.write('Cluster by: ' + ', '.join(cluster_col))
    st.plotly_chart(fig)

    

    
