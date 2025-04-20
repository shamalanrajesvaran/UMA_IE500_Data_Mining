#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def feature_engineering(df):
    """
    Adds custom features to the given DataFrame.

    Parameters:
    df (pd.DataFrame): Input dataframe containing hotel booking data.

    Returns:
    pd.DataFrame: Dataframe with new features added.
    """
    
    # Check if the reserved room type matches the assigned room type
    df['is_reserved_room_same_with_assigned_room'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)

    # Calculate the ratio of days in the waiting list to the average daily rate (adr)
    # Add a small value (1e-10) to adr to avoid division by zero errors
    df['waiting_list_to_adr_ratio'] = df['days_in_waiting_list'] / (df['adr'] + 1e-10)
    
    # Calculate the lead time to days in waiting list ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['lead_time_to_waiting_list_ratio'] = df['lead_time'] / (df['days_in_waiting_list'] + 1e-10)
    
    # Calculate the total_special_requests to adr ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['total_special_requests_to_adr_ratio'] = df['total_of_special_requests'] / (df['adr'] + 1e-10)
    
    # Calculate the adr to lead time ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['adr_to_lead_time_ratio'] = df['adr'] / (df['lead_time'] + 1e-10)
    
    # Calculate total stay length as the sum of stays in weekend nights and stays in week nights
    df['total_stay_length'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    # Calculate weekend to weekday ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['weekend_weekday_ratio'] = df['stays_in_weekend_nights'] / (df['stays_in_week_nights'] + 1e-10)
    
    # Create is_only_children_booking feature
    df['is_only_children_booking'] = ((df['adults'] == 0) & (df['children'] > 0)).astype(int)
    
    # Create is_only_adult_booking feature
    df['is_only_adult_booking'] = ((df['adults'] > 0) & (df['children'] == 0) & (df['babies'] == 0)).astype(int)

    # Create the arrival_date as a full datetime object from year, month, and day
    df['arrival_date_full'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-' + df['arrival_date_day_of_month'].astype(str)
    )

    # Get the weekday name for the arrival date (e.g., 'Monday', 'Tuesday', etc.)
    df['arrival_weekday'] = df['arrival_date_full'].dt.day_name()
    
    #A binary feature indicating if a booking includes only adults and is on weekdays—possibly suggesting a business trip, 
    # which may be less prone to cancellation.
    df['is_only_adults_on_weekday'] = (
        (df['adults'] > 0) &
        (df['children'] == 0) &
        (df['babies'] == 0) &
        (df['arrival_weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
    ).astype(int)
    
    # Calculate booking date as arrival_date_full minus lead_time days
    df['booking_date_full'] = df['arrival_date_full'] - pd.to_timedelta(df['lead_time'], unit='d')
    
    # Extract year, month, week number, and day of the month from 'booking_date_full'
    df['booking_date_year'] = pd.to_datetime(df['booking_date_full']).dt.year
    df['booking_date_month'] = pd.to_datetime(df['booking_date_full']).dt.month_name()
    df['booking_date_week_number'] = pd.to_datetime(df['booking_date_full']).dt.isocalendar().week.astype('int32')
    df['booking_date_day_of_month'] = pd.to_datetime(df['booking_date_full']).dt.day

    # Create season-related features for booking and arrival dates
    season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['booking_season'] = df['booking_date_full'].dt.month.map(season_mapping)
    df['arrival_season'] = df['arrival_date_full'].dt.month.map(season_mapping)
    
    # Create integer versions of booking_date_month and arrival_date_month columns
    month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['booking_date_month_integer_version'] = df['booking_date_month'].map(month_mapping)
    df['arrival_date_month_integer_version'] = df['arrival_date_month'].map(month_mapping)
    
    # Create special day-related features for booking and arrival dates
    special_days = [
    (1, 1),   # New Year's Day
    (12, 31),  # New Year's Eve
    (12, 25), # Christmas Day
    (2, 14),  # Valentine's Day
    (4, 1),   # Easter Sunday (example)
    (10, 31), # Halloween
    ]
    df['is_booking_on_special_day'] = df.apply(
        lambda x: int((x['booking_date_month_integer_version'], x['booking_date_day_of_month']) in special_days), axis=1)
    df['is_arrival_on_special_day'] = df.apply(
        lambda x: int((x['arrival_date_month_integer_version'], x['arrival_date_day_of_month']) in special_days), axis=1)
    
    # Check if the arrival and booking date is a weekend (Saturday or Sunday)
    df['is_arrival_on_weekend'] = df['arrival_date_full'].dt.weekday.isin([5, 6]).astype(int)
    df['is_booking_on_weekend'] = df['booking_date_full'].dt.weekday.isin([5, 6]).astype(int)

    return df

