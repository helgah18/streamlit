#importing necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from app import main


st.set_page_config(page_title="AirBnB data", page_icon="🌍")

st.sidebar.header("AirBnB data")
st.title('Airbnb listings in Copenhagen')


data = pd.read_csv('listings_CPH.csv')


st.markdown("""
 Airbnb or it's original name AirBedandBreakfast operates an online marketplace that offers short-term rentals. 
 The company charges a comission for each booking acting as a broker. The main dataset used in this project is called "listings_CPH.csv" and contains information on Airbnb rentals in Copenhagen. It consists of 13815 observations and 18 features that can be seen in the table below. 
 The features include information about the host , type of apartment/rental, reviews, neighbourhood and location.
""")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data.T)

st.write('The highest price in the dataset is', data['price'].max(), ' and the lowest price in the dataset is ',data['price'].min(),
      '\nThe mean is', round(data['price'].mean(),2) , 'and the median is ',data['price'].median())

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey='row')
fig.suptitle('Price column')
# Set the Seaborn style
sns.set_style('darkgrid')

#Original dataset
axes[0].hist(data['price'], bins=100, color="purple")
axes[0].set_title("Original dataset")

#After removing price outliers
axes[1].hist(data[(data['price']>0) & (data['price']<10000)]['price'], bins=100, color='thistle')
axes[1].set_title("After removing price outliers")


# Add a common title for the figure
fig.suptitle('Price column')

st.pyplot(fig)



st.markdown("Outliers are obviously having a significant impact on the data because the later plot clearly depicts the distribution much better. Next it will be investigated whether prices vary between Copenhagen's various neighborhoods.")



#Data cleaning

#License 
data_clean = data.copy()
np.unique(data_clean['license']) #The column is only nan so dropping.
data_clean = data_clean.drop(['license'], axis=1)

#Neighbourhood group
np.unique(data_clean['neighbourhood_group'])  #The column is only nan so dropping.
data_clean = data_clean.drop(['neighbourhood_group'], axis=1)

#Making relevant columns categorical
data_clean["room_type"] = data["room_type"].astype("category")
data_clean["neighbourhood"] = data["neighbourhood"].astype("category")

data_clean.to_csv()
#Making datetime data for dates
data_clean['last_review'] = pd.to_datetime(data['last_review'])

#drop price = 0
data_clean = data_clean[data_clean['price'] != 0]

data_clean.to_csv('Listings_clean.csv')


code = '''#License 
data_clean = data.copy()
np.unique(data_clean['license']) #The column is only nan so dropping.
data_clean = data_clean.drop(['license'], axis=1)

#Neighbourhood group
np.unique(data_clean['neighbourhood_group'])  #The column is only nan so dropping.
data_clean = data_clean.drop(['neighbourhood_group'], axis=1)

#Making relevant columns categorical
data_clean["room_type"] = data["room_type"].astype("category")
data_clean["neighbourhood"] = data["neighbourhood"].astype("category")

#Making datetime data for dates
data_clean['last_review'] = pd.to_datetime(data['last_review'])

#drop price = 0
data_clean = data_clean[data_clean['price'] != 0]'''

if st.checkbox('Show cleaning code'):
    st.subheader('Cleaning code')
    st.markdown("Let's clean the data. \n - Remove columns with only NaN values \n - Categorize the relevant columns. \n - Make columns including dates on relevant form of datetime")
    st.code(code, language='python')


st.title('Mapping the listings')

location = data_clean[['latitude',
       'longitude']]

st.map(location)


def switch_page(page_name: str):
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    from streamlit.source_util import get_pages

    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    pages = get_pages("AirBnBData.py")  # OR whatever your main page is called

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name,
                )
            )

    page_names = [standardize_name(config["page_name"]) for config in pages.values()]

    raise ValueError(f"Could not find page {page_name}. Must be one of {page_names}")

classifications = st.button("Look at classification examples")
if classifications:
    switch_page('classifications')

clustering = st.button("Look at clustering examples")
if classifications:
    switch_page('clustering')

if __name__ == "__main__":
    main()
