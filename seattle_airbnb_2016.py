import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
Question 1)
What percentage of hosts make 40K$+ a year from their listings on airbnb in seattle ? 
Question 2) 
Do properties with a superhost have higher booking rate  ?
Question 3) 
Do certain neighbrhoods have higher booking rate than others or is the number evenly spread out ? 
Question 4) 
If a friend were to list a property on airbnb in seattle, what property type, price, amenities and neighborhood would we advise him to pick ? 
Question 4)
Predict is host superhost 
Question 5)
Predict how many nights a property is booked through the year 
Question 6) 
Predict rating 
'''


def clean_numeric(num_text, is_money):
    try:
        if is_money:
            num = num_text
            if ',' in num:
                num = num.replace(',', '')

            return num.replace('$', '')

        else:
            return num_text.replace('%', '')

    except:
        # some rows have nan stored as string. We write this exception block to get around it return an actual None value
        return None


def clean_boolean(bol_text):
    try:
        if bol_text == 't':
            return '1'
        else:
            return '0'

    except:
        # some rows have nan stored as string. We write this exception block to get around it return an actual None value
        return None

unique_amenities = ['']
def clean_amenities(all_ams):
        amns_list = ['']
        amns = all_ams.replace('{', '').replace('}', '')

        try:
            amns = amns.replace('"', '').split(',')
        except:
            amns = amns.split(',')

        for item in amns:
            if item not in unique_amenities and item not in ['Dog(s)', 'Cat(s)', 'Pets live on this property', 'Other pet(s)', 'Wireless Internet', 'Washer / Dryer', 'Carbon Monoxide Detector']:
                unique_amenities.append(item)
            if item in amns_list:
                continue
            else:
                if item in ['Dog(s)', 'Cat(s)', 'Pets live on this property', 'Other pet(s)']:
                    if 'Pets Allowed' not in amns_list:
                        amns_list.append('Pets Allowed')
                elif item == 'Wireless Internet':
                    if 'Internet' not in amns_list:
                        amns_list.append('Internet')
                elif item == 'Washer / Dryer':
                    if 'Washer' not in amns_list:
                        amns_list.append('Washer')
                    if 'Dryer' not in amns_list:
                        amns_list.append('Dryer')
                elif item == 'Carbon Monoxide Detector':
                    if 'Smoke Detector' not in amns_list:
                        amns_list.append('Smoke Detector')
                elif item == 'Essentials':
                    continue
                else:
                    amns_list.append(item)
        return str(amns_list[1:])


def dummies_amenities(df):
    matrix = []
    for i in range(0, 3818):
        matrix.append(np.zeros(len(unique_amenities) + 1).tolist())

    for index_x, row in df.iterrows():
        for index_j in range(0, len(unique_amenities)):
            if unique_amenities[index_j] in row['amenities']:
                matrix[index_x][index_j] = 1
        matrix[index_x][-1] = row['id']

    unique_amenities.append('id')
    return pd.DataFrame(data=matrix, columns=unique_amenities).astype(int).copy()

def clean_calendar_updated(updated_at):
    time = 0
    if updated_at == 'a week ago':
        time = 7
    elif updated_at == 'today':
        time = 1
    elif updated_at == 'yesterday':
        time = 2
    elif updated_at == 'never':
        time = 0
    elif 'week' in updated_at:
        time = int(re.search(r'\d+', updated_at).group()) * 7
    elif 'month' in updated_at:
        time = int(re.search(r'\d+', updated_at).group()) * 30
    else:
        time = int(re.search(r'\d+', updated_at).group())

    return time


def hyp_test(df):
    normalized = [0]
    for index in range(0, 600):
        iteration = np.array([])
        for j in range(0, 300):
            pos = random.randint(0, df.shape[0]-1)
            iteration = np.insert(iteration, j, df.iloc[pos, 48])
        normalized.append(iteration.mean())

    return pd.DataFrame(data={'normalized': normalized[1: ]})




print('Step one : Load the data')
print('Success !!!\n')
df_listing = pd.read_csv('archive/listings.csv')
df_calendar = pd.read_csv('archive/calendar.csv')
df_reviews = pd.read_csv('archive/reviews.csv')

print('Step Two : explore the data')
print('Let\'s take a look at the columns and the data types in each table')
print("2.1) We start with the reviews df")
print(df_reviews.info())
print('\nThis df contains written reviews (text) submitted by Airbnb users after staying at one of the listed properties in df_listing')
print('We are going to keep all of the columns for now')
print('We might think of dropping the rows with null values in the comments column If we are going to analyze comments\' sentiment\n')
print('2.2) Calendar df')
print(df_calendar.info())
print('\nThis df shows the days in which each listing was available and occupied through out the year (365 days)')
print('It also shows the listing price only on days when status is available')
print('We are going to keep all of the columns and rows for now\n')
print('2.3) Walkthrough The listings df')
print(df_listing.info())
print('\n')
print('This is the main df')
print('It contains all of the information for each listing')
print('It consists of {} listings and {} columns'.format(df_listing.shape[0], df_listing.shape[1]))
print('We can notice that there are many important numeric columns with object data type like price, cleaning fee .. etc')
print('We will clean the data before plotting and analyzing numeric variables')
print('\n#################################\nCleaning Phase : ')
print('We will inspect the data for the following issues')
print('1- Make sure there are no duplicate rows')
df_listing_clean = df_listing.drop_duplicates().copy()
print('\nSize before dropping duplicates {} and after {}\n'.format(df_listing.shape[0], df_listing_clean.shape[0]))
print('2- Drop columns that are :')
print(' 2.1) Redundant\n  neighbourhood, smart_location, host_total_listings_count and calculated_host_listings_count')
print('\n 2.2) Irrelevant for analysis or modeling\n  latitude, longitude, listing_url, scrape_id, last_scraped, picture_url, xl_picture_url, medium_url, thumbnail_url, host_url, host_thumbnail_url, host_picture_url and calendar_last_scraped')
print('\n 2.3) Have all or the vast majority of their values as Null or one value across the whole df\n  experiences_offered, city, country, country_code, state, market, square_feet, license, has_availability, jurisdiction_names and requires_license')

df_listing_clean.drop(['neighbourhood', 'smart_location', 'host_total_listings_count', 'calculated_host_listings_count', 'latitude', 'longitude', 'listing_url', 'scrape_id', 'last_scraped', 'picture_url', 'xl_picture_url', 'medium_url', 'thumbnail_url', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped',
                 'experiences_offered', 'city', 'country', 'country_code', 'state', 'market', 'square_feet', 'square_feet', 'license', 'has_availability', 'jurisdiction_names', 'requires_license'], axis=1, inplace=True)
print('\nSize before dropping columns {} and after {}'.format(df_listing.shape[1], df_listing_clean.shape[1]))

print('\n3- Make sure All columns have an appropriate data type')
print(' 3.1) Some columns must have a numeric data type. However, the columns are of object data type. we are going to remove $ and % and convert the columns into floats or integers')
print('  columns : price, weekly_price, monthly_price, security_deposit, cleaning_fee, extra_people, host_acceptance_rate and host_response_rate')
df_listing_clean['price'] = df_listing_clean['price'].apply(lambda x: clean_numeric(x, True))
df_listing_clean['weekly_price'] = df_listing_clean['weekly_price'].apply(lambda x: clean_numeric(x, True))
df_listing_clean['monthly_price'] = df_listing_clean['monthly_price'].apply(lambda x: clean_numeric(x, True))
df_listing_clean['security_deposit'] = df_listing_clean['security_deposit'].apply(lambda x: clean_numeric(x, True))
df_listing_clean['cleaning_fee'] = df_listing_clean['cleaning_fee'].apply(lambda x: clean_numeric(x, True))
df_listing_clean['extra_people'] = df_listing_clean['extra_people'].apply(lambda x: clean_numeric(x, True))
df_listing_clean['host_acceptance_rate'] = df_listing_clean['host_acceptance_rate'].apply(lambda x: clean_numeric(x, False))
df_listing_clean['host_response_rate'] = df_listing_clean['host_response_rate'].apply(lambda x: clean_numeric(x, False))
# convert to float data type
df_listing_clean['price'] = df_listing_clean['price'].astype(float)
df_listing_clean['weekly_price'] = df_listing_clean['weekly_price'].astype(float)
df_listing_clean['monthly_price'] = df_listing_clean['monthly_price'].astype(float)
df_listing_clean['security_deposit'] = df_listing_clean['security_deposit'].astype(float)
df_listing_clean['cleaning_fee'] = df_listing_clean['cleaning_fee'].astype(float)
df_listing_clean['extra_people'] = df_listing_clean['extra_people'].astype(float)
df_listing_clean['host_acceptance_rate'] = df_listing_clean['host_acceptance_rate'].astype(float)
df_listing_clean['host_response_rate'] = df_listing_clean['host_response_rate'].astype(float)
print('\ncheck columns data types after cleaning')
print(df_listing_clean.iloc[:, [14, 15, 35, 36, 37, 38, 39, 41]].dtypes)
print('\n 3.2) Some columns must have a boolean or 0/1 values. However, the columns are stored as string and have only two unique values (t and f)')
print('  columns : host_is_superhost, host_identity_verified, host_has_profile_pic, instant_bookable, require_guest_profile_picture, require_guest_phone_verification and is_location_exact')
df_listing_clean['host_is_superhost'] = df_listing_clean['host_is_superhost'].apply(lambda x: clean_boolean(x))
df_listing_clean['host_identity_verified'] = df_listing_clean['host_identity_verified'].apply(lambda x: clean_boolean(x))
df_listing_clean['host_has_profile_pic'] = df_listing_clean['host_has_profile_pic'].apply(lambda x: clean_boolean(x))
df_listing_clean['instant_bookable'] = df_listing_clean['instant_bookable'].apply(lambda x: clean_boolean(x))
df_listing_clean['require_guest_profile_picture'] = df_listing_clean['require_guest_profile_picture'].apply(lambda x: clean_boolean(x))
df_listing_clean['require_guest_phone_verification'] = df_listing_clean['require_guest_phone_verification'].apply(lambda x: clean_boolean(x))
df_listing_clean['is_location_exact'] = df_listing_clean['is_location_exact'].apply(lambda x: clean_boolean(x))

df_listing_clean['host_is_superhost'] = df_listing_clean['host_is_superhost'].astype(int)
df_listing_clean['host_identity_verified'] = df_listing_clean['host_identity_verified'].astype(int)
df_listing_clean['host_has_profile_pic'] = df_listing_clean['host_has_profile_pic'].astype(int)
df_listing_clean['instant_bookable'] = df_listing_clean['instant_bookable'].astype(int)
df_listing_clean['require_guest_profile_picture'] = df_listing_clean['require_guest_profile_picture'].astype(int)
df_listing_clean['require_guest_phone_verification'] = df_listing_clean['require_guest_phone_verification'].astype(int)
df_listing_clean['is_location_exact'] = df_listing_clean['is_location_exact'].astype(int)
print('\ncheck columns data types after cleaning')
print(df_listing_clean.iloc[:, [16, 20, 21, 26, 59, 61, 62]].dtypes)
print('\n4- Clean amenities column')
print(' Remove redundant amenities and return a clean list')
df_listing_clean['amenities'] = df_listing_clean.amenities.apply(lambda x: clean_amenities(x))
print('\n5- unify units in calendar_updated column')
print('\n We will convert weeks and months to days')
df_listing_clean['calendar_updated_numeric'] = df_listing_clean['calendar_updated'].apply(lambda x: clean_calendar_updated(x))
df_listing_clean['calendar_updated_numeric'] = df_listing_clean['calendar_updated_numeric'].astype(int)
print('\n6- Handle null values and categorical columns (will perform this step later before modelling)')
print('\n#################################\nLet\'s briefly look at descriptive statistics and plot numeric data : ')
numeric_df = df_listing_clean.iloc[:, [14, 15, 18, 29, 30, 31, 32, 35, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 63]].copy()
print(numeric_df.describe().to_string())
print('\n Percentage of listings that require a secuirity deposit = {}%'.format(round(numeric_df[numeric_df.security_deposit.isnull()==False].shape[0]/numeric_df.shape[0]*100)))
print('\n Percentage of listings that require a cleaning fee = {}% and 75% ask for less than 83$'.format(round(numeric_df[numeric_df.cleaning_fee.isnull()==False].shape[0]/numeric_df.shape[0]*100)))
print('\n Percentage of listings that allow one guest only = {}%'.format(round(numeric_df.query('guests_included == 1.0').shape[0]/numeric_df.shape[0]*100)))
print('\n Percentage of listings that are booked more than 50% of the year = {}%'.format(round(numeric_df.query('availability_365 <= 182').shape[0]/numeric_df.shape[0]*100)))
'''
1) 75% of hosts have a response rate greater than or equal to 98% and an acceptance rate of 100%
2) 50% of hosts have 1 property on airbnb and 75% have 2 or less 
3) the majority of listings have 1 bathroom, 1 bedroom and 1 bed 
4) 75% of listings are priced below 150$. The most expensive listing is priced at 1000$ 
5) half of the listing allow bookings with more than one night
6) 75% of listing get less than 3 reviews per month 
'''

print('\nlet\'s look at a couple of plots explain what we see')
numeric_df.loc[:, ['availability_365', 'number_of_reviews', 'review_scores_rating', 'reviews_per_month']].hist()
plt.show()
'''
7) Approximately half of listings are almost available the whole year  
8) Tha vast majority of listings get less than 4 reviews a month 
9) Tha vast majority of listings have a review score rating of 90-100%. There a a few exceptions on the other end of the spectrum 
'''
print('\nNext, We will plot a heatmap find  out if some variables are correlated to some target variables (price, review_scores_rating and availability_365')
#sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f");
#plt.show()
# there are no interesting linear relationships in the plot above
print('\n#################################\nWe move on to the next step to answer the first two questions')
print('\nQuestion 1: What percentage of hosts that make 35K$+ a year from their listings on airbnb in seattle ?')
df_listing_clean['year_revenue'] = (365-df_listing_clean['availability_365'])*df_listing_clean['price']
print('\n {}% of hosts make 35K$+ from their listings in seattle'.format(round(df_listing_clean.groupby('host_id').year_revenue.sum().reset_index(name='rev').query('rev >= 35000.0').shape[0]/df_listing_clean.host_id.nunique()*100, 2)))
print(' {}% of hosts make 7500$ or less from their listings in seattle'.format(round(df_listing_clean.groupby('host_id').year_revenue.sum().reset_index(name='rev').query('rev <= 7500.0').shape[0]/df_listing_clean.host_id.nunique()*100, 2)))
print(' {}% of hosts make nothing from their listings in seattle'.format(round(df_listing_clean.groupby('host_id').year_revenue.sum().reset_index(name='rev').query('rev == 0.0').shape[0]/df_listing_clean.host_id.nunique()*100, 2)))
print('\nQuestion2: Do properties with a superhost have higher booking rates ?')
print('\n H0: superhosts have the same or higher availability_365')
print(' H1: superhosts have a lower availability_365')
print(' alpha = 0.05')
print('\n First of all, we normalize availability_365 by taking (N) random values and finding their means and repeating this process many times then plotting the means')
normalized_365 = hyp_test(df_listing_clean)
# normalized_365.hist()
# plt.show()

print('\n Split the data into our two segments, superhosts and hosts. Find their availability_365 means')
superhost_mean, host_mean = df_listing_clean.query('host_is_superhost == 1').availability_365.mean(), df_listing_clean.query('host_is_superhost == 0').availability_365.mean()
superhost_var, host_var = df_listing_clean.query('host_is_superhost == 1').availability_365.var(), df_listing_clean.query('host_is_superhost == 0').availability_365.var()
print(' Mean availability_365 : superhosts = {} and hosts = {} days a year'.format(superhost_mean, host_mean))
print(' Variance of availability_365 : superhosts = {} and hosts = {} and ratio = {}'.format(superhost_var, host_var, superhost_var/host_var))
_, p_value = stats.ttest_ind(a=df_listing_clean.query('host_is_superhost == 1').availability_365, b=df_listing_clean.query('host_is_superhost == 0').availability_365, equal_var=True)
if superhost_mean < host_mean:
    p = p_value/2
else:
    p = 1.0 - p_value/2

print(' Is p_value {} less than alpha = 0.05 ? {}'.format(p, p < 0.05))
print('\n Conclusion : We reject the alternative hypothesis')


print('\nQuestion 3) Predict If host is superhost from a list of available features')
print('\n We will build a model that will help us predict wether a new host is a superhost or not based')
print(' We will single out a number of features that might help us to accurately predict/classify the host')
print(' Some of the features that are available when a listing is newly published : \n price - neighborhood - amenities - cleaning fee - secuirity deposit - guests_included - calendar_updated_numeric - extra_people\n bed_type - room_type - property_type - cancellation_policy - require_guest_phone_verification - require_guest_profile_picture - instant_bookable\n')

# handling null values
# We are going to consider one of two strategies when working with null values
# 1- drop
# 2- Impute
svm_df = df_listing_clean.loc[:, ['host_id', 'id', 'host_is_superhost', 'price', 'neighbourhood_cleansed', 'cleaning_fee', 'security_deposit', 'guests_included', 'extra_people', 'calendar_updated_numeric', 'bed_type', 'room_type', 'property_type', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture', 'require_guest_phone_verification']].copy()
# our target/response variable host_is_superhost doesn't have any null values
# price also doesn't have any null values
# bed_type, room_type, neighbourhood_cleansed, instant_bookable, cancellation_policy, require_guest_profile_picture, calendar_updated_numeric and require_guest_phone_verification all don't have any null values
# cleaning_fee, security_deposit have null values. We are going to assume that a null value in both of these columns means a fee is not required => fill na with 0
svm_df['cleaning_fee'] = svm_df['cleaning_fee'].fillna(0)
svm_df['security_deposit'] = svm_df['security_deposit'].fillna(0)
# property_type has one null value, drop the row
svm_df.dropna(subset=['property_type'], inplace=True)
print(svm_df.info())
# handle categorical variables
# neighbourhood_cleansed, bed_type, room_type, property_type, cancellation_policy
svm_df = pd.get_dummies(svm_df, columns=['neighbourhood_cleansed', 'bed_type', 'room_type', 'property_type', 'cancellation_policy'], dtype=int, prefix=['neighbourhood', 'bed', 'room', 'property', 'cancel'])
# amenities
unique_amenities = unique_amenities[1:]
svm_df = svm_df.merge(dummies_amenities(df_listing_clean), on='id', how='inner')
print(svm_df.dtypes)

# our target/response variable host_is_superhost is a binary variable
# which means where are going to select a classification algorithm for our model
# SVM
print('\n\n')
X_train, X_test, y_train, y_test = train_test_split(svm_df.iloc[:, 3:], svm_df.iloc[:, 2], test_size=0.33, random_state=42)
superhostModel = svm.SVC()
superhostModel.fit(X_train, y_train)
predicted = superhostModel.predict(X_test)
print(accuracy_score(y_test, predicted))

