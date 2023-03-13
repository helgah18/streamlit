import pages.Classification
import pages.Clustering
import pages.ConnectToSnowflake
import streamlit as st

def snowflake_page():
    import snowflake.connector 
    import streamlit as st
    import pandas as pd

    # Connect to Snowflake
    def init_connection():
        return snowflake.connector.connect(
            **st.secrets["snowflake"],
            client_session_keep_alive=True
        )

    conn = init_connection()

    # Retrieve data from Snowflake table
    query = 'SELECT * FROM AIRBNB_DB.PUBLIC.AIRBNB'
    df = pd.read_sql_query(query, conn)

    # Close the Snowflake connection
    conn.close()
    st.title(':snowflake: Connecting Streamlit to Snowflake :snowflake:')
    st.write("Guide to access a Snowflake database from Streamlit Community Cloud. Using the snowflake-connector-python library and Streamlit's secrets management.")


    st.title('Necessary items created in Snowflake')

    st.write('Database:')
    st.code("""
    CREATE DATABASE airbnb_db;
    """)

    st.write('Staging table:')
    st.code("""
    CREATE STAGE airbnb_stage;
    """)

    st.write('Destination table in Snowflake:')
    st.code("""
    CREATE  or replace TABLE airbnb (
    id INT,
    name VARCHAR,
    host_id VARCHAR,
    host_name VARCHAR,
    neighbourhood_group VARCHAR,
    neighbourhood VARCHAR,
    latitude FLOAT,
    longitude FLOAT,
    room_type VARCHAR,
    price FLOAT,
    minimum_nights VARCHAR,
    number_of_reviews VARCHAR,
    last_review VARCHAR,
    reviews_per_month FLOAT,
    calculated_host_listings_count FLOAT,
    availability_365 FLOAT,
    number_of_reviews_ltm FLOAT
    );
    """)

    st.write('File format:')
    st.code("""
    CREATE OR REPLACE FILE FORMAT my_csv_format 
    TYPE = 'CSV' 
    RECORD_DELIMITER = '\r\n'
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1 
    NULL_IF = ('NULL', 'null') 
    EMPTY_FIELD_AS_NULL = TRUE 
    COMPRESSION = 'GZIP';
    """)


    # secrets
    st.title('Add username and password to your local app secrets')
    st.write('Your local Streamlit app will read secrets from a file .streamlit/secrets.toml in your app’s root directory. Create this file if it doesn’t exist yet and add your Snowflake username, password, account identifier, and the name of your warehouse, database, and schema as shown below:')
    st.code(' #.streamlit/secrets.toml \n user = "xxx" \n password = "xxx" \n account = "xxx" \n warehouse = "xxx" \n database = "xxx" \n schema = "xxx"')
    st.subheader('Copy your app secrets to the cloud')
    st.write("As the secrets.toml file above is not committed to GitHub, you need to pass its content to your deployed app (on Streamlit Community Cloud) separately. Go to the app dashboard and in the app's dropdown menu, click on Edit Secrets. Copy the content of secrets.toml into the text area.")


    # Display information on how the data was loaded to Snowflake
    st.title('Loading Data to Snowflake')
    st.write('The data was loaded to Snowflake using SnowSQL in the command prompt:')
    st.write('1. Connect to the Snowflake account:')
    st.code("snowsql -a sf89872.west-europe.azure -u helhal")
    st.write('2. Specify the database:')
    st.code("USE DATABASE AIRBNB_DB")
    st.write('3. Load data to staging table:')
    st.code("PUT 'file:///C:/Users/HelgaHalldórdóttir/OneDrive - Intellishore/Documents/StreamLit/Listings_clean.csv' @airbnb_stage")
    st.write('4. Copy the data from the staging table to the destination table:')
    st.code("""
    COPY INTO airbnb
    FROM @airbnb_stage/Listings_clean.csv
    FILE_FORMAT = (TYPE = CSV
                FIELD_DELIMITER = ',' 
                SKIP_HEADER = 1)
    ON_ERROR = 'CONTINUE';
    """)




    # Display information on how the data was loaded from Snowflake
    st.title('Loading Data from Snowflake')
    st.write("Install the Snowflake Connector for Python, making sure the correct requirements are installed [Instructions](https://docs.snowflake.com/en/user-guide/python-connector-install)")
    st.write('To retrieve data from the Snowflake table, the following code was used:')
    st.code("""
    import snowflake.connector 
    import streamlit as st
    import pandas as pd

    # Connect to Snowflake
    def init_connection():
        return snowflake.connector.connect(
            **st.secrets["snowflake"],
            client_session_keep_alive=True
        )

    conn = init_connection()

    # Retrieve data from Snowflake table
    query = 'SELECT * FROM AIRBNB_DB.PUBLIC.AIRBNB'
    df = pd.read_sql_query(query, conn)

    # Close the Snowflake connection
    conn.close()
    """)

    st.title('The data')
    st.write('Now we have the following dataframe that will be used for the other pages :pencil:')
    st.write(df)


    # Display basic information on the data

    st.write(f"Number of rows: {len(df)}")
    st.write(f"Number of columns: {len(df.columns)}")
    st.write("Column data types:")
    st.write(df.dtypes)


def clustering_page():
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

    if st.checkbox('Show dataset'):
        st.subheader('Data')
        st.write(data.T)

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
    number = st.number_input('Input number of clusters')
    st.write('The current number is ')

    if cluster_col: 
        number = 5
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

    


def classification_page():
    #importing necessary packages
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    import streamlit as st


    st.sidebar.header("AirBnB data")
    st.title('Classification for low or high prices')
    st.markdown('The features were chosen based in relevancy and correlation to the target price. The following features utilized were: \n - Minimum nights \n - Availability over the year \n - Room type \n - Neighbourhood \n  - Longitude \n - Latidude \n - The test set was 30% of the data and the remaining 70% was the training data.')
    data = pd.read_csv('Listings_clean.csv')

    # function to evaluate predictions
    def evaluate(y_true, y_pred, classifier):
        # calculate and display confusion matrix
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
        print(cm)

        # calculate precision, recall, and F1 score
        accuracy = float(np.trace(cm)) / np.sum(cm)
        if classifier == 'binary':
            precision = precision_score(y_true, y_pred, average='binary', labels=labels)
            recall = recall_score(y_true, y_pred, average='binary' , labels=labels)
            f1 = f1_score(y_true,y_pred, average = 'binary')
        else:
            precision = precision_score(y_true, y_pred, average='weighted', labels=labels)
            recall = recall_score(y_true, y_pred, average='weighted' , labels=labels)
            f1 = 2 * precision * recall / (precision + recall)

        st.write("accuracy:", accuracy)
        st.write("precision:", precision)
        st.write("recall:", recall)
        st.write("f1 score:", f1)

        #Making a target variable based on price (the mean is the threshold).
    price_threshold = data['price'].median()
    data['rental_price_binary'] = data['price'].apply(lambda x: 1 if x > price_threshold else 0)

    features = data[['minimum_nights', 'availability_365', 'room_type','neighbourhood', 'longitude', 'latitude']]

    target = data['rental_price_binary']

    #Splitting to test and train
    split=int(len(data)*0.7)
    x_train = features[:split]
    x_test = features[split:]
    y_train = target[:split]
    y_test = target[split:]

    #Making dummy variables
    x_train = pd.get_dummies(x_train, columns=['neighbourhood', 'room_type'])
    x_test = pd.get_dummies(x_test, columns=['neighbourhood', 'room_type'])


    #Standardizing the data
    x_train_std = x_train.copy()
    x_test_std = x_test.copy()

    features_std =['minimum_nights', 'availability_365','longitude', 'latitude']

    x_train_std[features_std] = (x_train[features_std] - x_train[features_std].mean()) / x_train[features_std].std()
    x_test_std[features_std] = (x_test[features_std] - x_train[features_std].mean()) / x_train[features_std].std()

    model = st.selectbox(
        'Pick a classification model',
        ('Logistic Regression', 'Random Forrest', 'Support Vector Machine'))


    #Making a logistic regression binary classifier
    if model == 'Logistic Regression':
        lr = LogisticRegression(max_iter = 10000)
        lr.fit(x_train_std,y_train)
        y_pred = lr.predict(x_test_std)
    elif model == 'Random Forrest':
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(x_train_std,y_train)
        y_pred=clf.predict(x_test_std)
    elif model == 'Support Vector Machine': 
        svc =SVC()
        svc.fit(x_train_std, y_train)
        y_pred = svc.predict(x_test_std)

    code = '''    #Making a target variable based on price (the mean is the threshold).
    price_threshold = data['price'].median()
    data['rental_price_binary'] = data['price'].apply(lambda x: 1 if x > price_threshold else 0)

    features = data[['minimum_nights', 'availability_365', 'room_type','neighbourhood', 'longitude', 'latitude']]

    target = data['rental_price_binary']

    #Splitting to test and train
    split=int(len(data)*0.7)
    x_train = features[:split]
    x_test = features[split:]
    y_train = target[:split]
    y_test = target[split:]

    #Making dummy variables
    x_train = pd.get_dummies(x_train, columns=['neighbourhood', 'room_type'])
    x_test = pd.get_dummies(x_test, columns=['neighbourhood', 'room_type'])


    #Standardizing the data
    x_train_std = x_train.copy()
    x_test_std = x_test.copy()

    features_std =['minimum_nights', 'availability_365','longitude', 'latitude']

    x_train_std[features_std] = (x_train[features_std] - x_train[features_std].mean()) / x_train[features_std].std()
    x_test_std[features_std] = (x_test[features_std] - x_train[features_std].mean()) / x_train[features_std].std()


    #Making a logistic regression binary classifier
    lr = LogisticRegression(max_iter = 10000)
    lr.fit(x_train_std,y_train)

    # make predictions from Logistic regression model
    y_pred_lr = lr.predict(x_test_std)'''

    # Evaluation the binary classifier
    st.subheader('Evaluation from the model')
    evaluate(y_test, y_pred, 'binary')

    labels = np.unique(y_test)

    # Create a heatmap from the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        xlabel='Predicted label', ylabel='True label',
        title='Confusion matrix',
        aspect='equal')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    st.pyplot(fig)

    if st.checkbox('Show binary prediction code'):
        st.subheader('binary prediction code')
        st.code(code, language='python')

    if st.checkbox('Show coefficient from the model'):
        st.subheader('Coefficients:')
        for colname, val in zip(x_train.columns, lr.coef_.tolist()[0]):
            st.write("%s=%.3f"%(colname, val))


PAGES = {
    "Classification": classification_page,
    "Clustering": clustering_page,
    "Connect to Snowflake": snowflake_page,
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()

