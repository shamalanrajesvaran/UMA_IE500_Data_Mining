import pandas as pd

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
    # Add a small value (1e-10) to avoid division by zero errors # REMOVED THIS AS IT BECOMES TOO CORRELATED WITH lead_time itself
    #df['lead_time_to_waiting_list_ratio'] = df['lead_time'] / (df['days_in_waiting_list'] + 1e-10)
    
    # Calculate the total_special_requests to adr ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['total_special_requests_to_adr_ratio'] = df['total_of_special_requests'] / (df['adr'] + 1e-10)
    
    # Calculate lead_time_to_adr_ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['lead_time_to_adr_ratio'] = df['lead_time'] / (df['adr'] + 1e-10)

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
    df['booking_weekday'] = df['booking_date_full'].dt.day_name()
    
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
    (12, 24), # Christmas's Eve
    (2, 14),  # Valentine's Day
    (10, 31), # Halloween
    (5, 1)  # Labor Day
    ]
    df['is_booking_on_special_day'] = df.apply(
        lambda x: int((x['booking_date_month_integer_version'], x['booking_date_day_of_month']) in special_days), axis=1)
    df['is_arrival_on_special_day'] = df.apply(
        lambda x: int((x['arrival_date_month_integer_version'], x['arrival_date_day_of_month']) in special_days), axis=1)
    
    # Check if the arrival and booking date is a weekend (Saturday or Sunday)
    df['is_arrival_on_weekend'] = df['arrival_date_full'].dt.weekday.isin([5, 6]).astype(int)
    df['is_booking_on_weekend'] = df['booking_date_full'].dt.weekday.isin([5, 6]).astype(int)
    
    # Calculate number of bookings before a booking 
    df['total_previous_bookings'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']
    
    # Calculate the percentage of cancellations before a booking (if total_previous_bookings is 0, this also becomes 0)
    df['previous_cancellation_percentage'] = df.apply(
    lambda x: (x['previous_cancellations'] / x['total_previous_bookings'] * 100) 
    if x['total_previous_bookings'] > 0 else 0,
    axis=1
    )
    
    # Calculate ratio of the number of changes about a booking and lead time
    # Add a small value (1e-10) to avoid division by zero errors
    df['booking_changes_to_lead_time_ratio'] = df['booking_changes'] / (df['lead_time'] + 1e-10)
    
    # Calculate ratio of the number of changes about a booking and days_in_waiting_list
    # Add a small value (1e-10) to avoid division by zero errors # REMOVED THIS AS IT BECOMES TOO CORRELATED WITH booking_changes itself
    #df['booking_changes_to_days_in_waiting_list_ratio'] = df['booking_changes'] / (df['days_in_waiting_list'] + 1e-10)

    # Calculate ratio of the number of changes about a booking and adr
    # Add a small value (1e-10) to avoid division by zero errors
    df['booking_changes_to_adr_ratio'] = df['booking_changes'] / (df['adr'] + 1e-10)

    # Calculate total_special_requests_to_lead_time_ratio
    # Add a small value (1e-10) to avoid division by zero errors
    df['total_special_requests_to_lead_time_ratio'] = df['total_of_special_requests'] / (df['lead_time'] + 1e-10)

    # Calculate ratio of total_previous_bookings and total_of_special_requests
    # Add a small value (1e-10) to avoid division by zero errors
    df['total_previous_bookings_to_total_special_requests_ratio'] = df['total_previous_bookings'] / (df['total_of_special_requests'] + 1e-10)

    # Calculate ratio of days_in_waiting_list and total_of_special_requests
    # Add a small value (1e-10) to avoid division by zero errors # REMOVED THIS AS IT BECOMES TOO CORRELATED WITH days_in_waiting_list itself
    #df['days_in_waiting_list_to_total_special_requests_ratio'] = df['days_in_waiting_list'] / (df['total_of_special_requests'] + 1e-10)

    # Create a binary feature indicating if a guest's reserved premium room was downgraded to a non-premium room
    # Room types H, G, F, and C are identified as premium due to their higher average ADR (Average Daily Rate)
    # compared to other room types in the dataset
    premium_rooms = ['H', 'G', 'F', 'C']

    df['is_premium_room_downgraded'] = (
        df['reserved_room_type'].isin(premium_rooms) &
        ~df['assigned_room_type'].isin(premium_rooms)
    ).astype(int)

    # Create a new binary feature indicating whether the room was upgraded to a premium room
    df['is_room_upgraded_to_premium'] = (
    ~df['reserved_room_type'].isin(premium_rooms) &  # Reserved room is not premium
     df['assigned_room_type'].isin(premium_rooms)   # But assigned room is premium
    ).astype(int)
    
    # Early booking with refundable deposit — could be cancellation prone
    df['is_early_refundable_booking'] = ((df['lead_time'] > 90) & (df['deposit_type'] == 'Refundable')).astype(int)

    # Late booking with non-refundable deposit — likely to show up
    df['is_late_non_refundable_booking'] = ((df['lead_time'] < 30) & (df['deposit_type'] == 'Non Refund')).astype(int)
    
    # Ratio of lead_time to total stay length — shows how early the booking was made relative to stay duration
    # Add a small value to avoid division by zero
    df['lead_time_to_total_stay_ratio'] = df['lead_time'] / (df['stays_in_weekend_nights'] + df['stays_in_week_nights'] + 1e-10)

    # Create binary feature for reservations with meal type 'Undefined' or 'SC', and total stay duration (weekend + weekday nights) > 3
    df['is_long_stay_no_meal'] = df.apply(
        lambda x: (x['meal'] in ['Undefined', 'SC']) and ((x['stays_in_weekend_nights'] + x['stays_in_week_nights']) > 3),
        axis=1
    ).astype(int)

    # Create binary feature for bookings with children or babies but meal type is 'Undefined' or 'SC'
    # This may indicate families who might be more likely to cancel due to lack of included meals
    df['has_kids_but_no_meal'] = df.apply(
        lambda x: ((x['children'] + x['babies']) > 0) and (x['meal'] in ['Undefined', 'SC']),
        axis=1
    ).astype(int)
    
    #fillna for country
    df['country'] = df['country'].fillna('unknown')
    
    # create new column "is_low_activity_agent"
    df['agent'] = df['agent'].fillna('unknown')
    df['agent'] = df['agent'].astype(str) #agent column is logically categorical
    # Count agent IDs
    agent_counts = df['agent'].value_counts()
    # Identify agents with fewer than 1000 bookings — considered low-activity agents
    rare_agents = agent_counts[agent_counts < 1000].index
    # create the column
    df['is_low_activity_agent'] = df['agent'].isin(rare_agents).astype(int)
    
    # Calculate group size
    df["children"] = df["children"].astype("Int64") #first fix dtype of children column
    df["total_people"] = df["adults"] + df["children"] + df["babies"]
    
    # Hotel is probably in Portugal bcs most of the bookings are made from Portugal, so let's check if booking was made by a foreigner
    #df["is_booking_foreign"] = (df["country"] != "PRT").astype(int) # REMOVED AS THIS BECOMES 0.98 CORRELATED WITH country_frequency_encoded COLUMN

    # Check if they changed the room of a loyal customer
    df["is_repeated_guest_but_changed_room"] = ((df["is_repeated_guest"] == 1) & (df["reserved_room_type"] != df["assigned_room_type"])).astype(int)

    # Check if the booking was made late
    df["is_late_booking"] = (df["lead_time"] < 7).astype(int)

    # Check if it is a solo traveler
    df["is_solo_traveler"] = ((df["adults"] + df["children"] + df["babies"]) == 1).astype("Int64")

    #drop unnecessary or unusable columns for modeling
    df = df.drop(
        columns=[
            'arrival_date_year',
            'arrival_date_day_of_month',
            'company',
            'reservation_status',
            'reservation_status_date',
            'arrival_date_full',
            'booking_date_full',
            'booking_date_year',
            'booking_date_day_of_month',
            'booking_date_month_integer_version',
            'arrival_date_month_integer_version'
        ]
    )


    return df