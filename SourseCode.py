"""
ITC 686 Project

Group Members: 
1. Riad Hossain - hossa1r
2. Mahmod Besher Alrez - alrez1m

Original file located: 
https://colab.research.google.com/drive/1OQ5lB2hA5lbHJlbfNgYXPGCRcBPsk0Fx?authuser=2#scrollTo=lRzQap0F646g

"""

import pandas as pd

# Load data
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# Load the dataset
crime_data = pd.read_csv('/content/drive/MyDrive/itc686/Project/Crimes_-_2001_to_Present_20240405.csv')

crime_data.describe()

crime_data.head(10)

# Query 1
# Total Number of Incidents by Year
total_incidents_by_year = crime_data['Year'].value_counts().sort_index().reset_index()
total_incidents_by_year.columns = ['Year', 'TotalIncidents']
print(total_incidents_by_year)

# Query 2
# Number of Arrests vs. Non-Arrests
arrest_count = crime_data['Arrest'].value_counts().reset_index()
arrest_count.columns = ['ArrestStatus', 'IncidentCount']
print(arrest_count)

# Query 3
# Distribution of Crime Types
crime_type_distribution = crime_data['Primary Type'].value_counts().reset_index()
crime_type_distribution.columns = ['PrimaryType', 'IncidentCount']
print(crime_type_distribution)

# Query 4
# Arrest Rate by Crime Type
arrest_rate_by_type = crime_data.groupby('Primary Type')['Arrest'].apply(lambda x: (x == True).mean() * 100).reset_index()
arrest_rate_by_type.columns = ['PrimaryType', 'ArrestRate']
print(arrest_rate_by_type.sort_values(by='ArrestRate', ascending=False))

# Query 5
# Most Common Crime Location by Year
common_location_by_year = crime_data.groupby(['Year', 'Location Description']).size().reset_index(name='IncidentCount')
print(common_location_by_year.sort_values(by=['Year', 'IncidentCount'], ascending=[True, False]).groupby('Year').head(1))

# Query 6
# Crime Trend Analysis
crime_trends = crime_data.groupby(['Year', 'Primary Type']).size().reset_index(name='IncidentCount')

# Shift the data to compare with previous year
crime_trends['IncidentCount_prev'] = crime_trends.groupby('Primary Type')['IncidentCount'].shift(1)

# Calculate the percentage change
crime_trends['PercentageChange'] = ((crime_trends['IncidentCount'] - crime_trends['IncidentCount_prev']) / crime_trends['IncidentCount_prev'].abs()) * 100

# Drop NaN values
crime_trends.dropna(subset=['PercentageChange'], inplace=True)

# Print the DataFrame with percentage change
print(crime_trends)

# Query 7
# Convert the 'Date' column to datetime if it's not already
crime_data['Date'] = pd.to_datetime(crime_data['Date'])

# Filter the data for the crime type "Robbery"
robbery_data = crime_data[crime_data['Primary Type'] == 'ROBBERY'].copy()  # Make a copy to avoid the warning

# Create a new column to extract the day of the week
robbery_data.loc[:, 'DayOfWeek'] = robbery_data['Date'].dt.day_name()

# Group the data by day of the week and count the number of incidents
weekly_pattern = robbery_data.groupby('DayOfWeek').size().reset_index(name='IncidentCount')

# Sort the data by the day of the week
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern['DayOfWeek'] = pd.Categorical(weekly_pattern['DayOfWeek'], categories=days_order, ordered=True)
weekly_pattern = weekly_pattern.sort_values('DayOfWeek')

# Print or visualize the weekly pattern for robbery incidents
print(weekly_pattern)

# Query 9
# Hotspots for Narcotics Crimes
narcotics_hotspots = crime_data[crime_data['Primary Type'] == 'NARCOTICS'].groupby('Block').size().reset_index(name='NarcoticsCount').nlargest(10, 'NarcoticsCount')
print(narcotics_hotspots)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Overview
print(crime_data.head())
print(crime_data.info())

# 2. Crime Trends Over the Years
crime_trends_over_years = crime_data['Year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x=crime_trends_over_years.index, y=crime_trends_over_years.values)

plt.xticks(crime_trends_over_years.index, rotation=45)

plt.title('Total Reported Crimes Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.show()

# 3. Crime Distribution by Type
plt.figure(figsize=(15, 8))
sns.countplot(data=crime_data, y='Primary Type', order=crime_data['Primary Type'].value_counts().index)
plt.title('Distribution of Crime Types')
plt.xlabel('Number of Crimes')
plt.ylabel('Crime Type')
plt.show()

# 4. Arrest Rate Analysis
arrest_rate_by_type = crime_data.groupby('Primary Type')['Arrest'].apply(lambda x: (x == True).mean() * 100).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=arrest_rate_by_type.values, y=arrest_rate_by_type.index, palette='viridis')
plt.title('Arrest Rate by Crime Type')
plt.xlabel('Arrest Rate (%)')
plt.ylabel('Crime Type')
plt.show()

# 5. Crime Distribution by Location
plt.figure(figsize=(15, 8))
sns.countplot(data=crime_data, y='Location Description', order=crime_data['Location Description'].value_counts().iloc[:15].index)
plt.title('Top 15 Locations with Most Reported Crimes')
plt.xlabel('Number of Crimes')
plt.ylabel('Location Description')
plt.show()

# 6
# Filter the dataset for narcotics-related crimes
narcotics_data = crime_data[crime_data['Primary Type'] == 'NARCOTICS']

# Group by Year and calculate the mean arrest rate
arrest_rate_by_year_narcotics = narcotics_data.groupby('Year')['Arrest'].mean().reset_index()

# Calculate correlation
correlation_year_arrest_rate = arrest_rate_by_year_narcotics['Year'].corr(arrest_rate_by_year_narcotics['Arrest'])

print(f"Correlation between Year and Arrest Rate for Narcotics Crimes: {correlation_year_arrest_rate}")

# 7
# Scatter plot of Arrest Rate for Narcotics crimes Over the Years
plt.figure(figsize=(12, 6))
sns.scatterplot(data=arrest_rate_by_year_narcotics, x='Year', y='Arrest', color='blue')
plt.title('Arrest Rate for Narcotics Crimes Over the Years')
plt.xlabel('Year')
plt.ylabel('Arrest Rate')
plt.grid(True)
plt.show()

# 8
# Filter the dataset for assault-related crimes
assault_data = crime_data[crime_data['Primary Type'] == 'ASSAULT']

# Group by Community Area and calculate the mean arrest rate
arrest_rate_by_community_area_assault = assault_data.groupby('Community Area')['Arrest'].mean().reset_index()

# Calculate correlation
correlation_community_area_arrest_rate = arrest_rate_by_community_area_assault['Community Area'].corr(arrest_rate_by_community_area_assault['Arrest'])

print(f"Correlation between Community Area and Arrest Rate for Assault Crimes: {correlation_community_area_arrest_rate}")

# 9 
# Scatter plor for Arrest Rate for Assault Crimes by Community Area
sns.scatterplot(data=arrest_rate_by_community_area_assault, x='Community Area', y='Arrest', color='red')
plt.title('Arrest Rate for Assault Crimes by Community Area')
plt.xlabel('Community Area')
plt.ylabel('Arrest Rate (%)')
plt.grid(True)
plt.show()
