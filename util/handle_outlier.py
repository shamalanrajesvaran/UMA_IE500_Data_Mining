def handle_outlier(df):
    df = df[df['adr'] >= 0]
    df = df[df['stays_in_weekend_nights'] <= 8]
    df = df[df['stays_in_week_nights'] <= 20]
    df = df[df['adults'] <= 6]
    df = df[df['babies'] <= 5]
    df = df[df['previous_bookings_not_canceled'] <= 20]
    df = df[df['booking_changes'] <= 10]
    df = df[df['adr'] != 5400]  # drop that one line with 5400
    df = df[df['required_car_parking_spaces'] <= 4]
    df = df[df['total_people'] <= 6]
    
    return df