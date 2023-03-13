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
