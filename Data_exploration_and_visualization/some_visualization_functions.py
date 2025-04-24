import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import calendar
from scipy.stats import pointbiserialr
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


## ------------------------------------------- Summary of the functions -------------------------------------------------------- ## 

# 1. General functions for first overviews are: 
    # - plot_cancellation_distribution_general_overview_bar
    # - plot_cancellation_by_category_general_for_categorical_data
    # - plot_cancellation_by_hotel_for_categorical_data

# 2. Functions for temporal trends are: 
    # - plot_temporal_trends_smoothed_14daysavg_bookings_cancellations_both_hotels
    # - plot_temporal_trends_monthly_bookings_cancellations_both_hotels
    # - plot_temporal_trends_smoothed_14daysavg_bookings_cancellations_split_by_hotels
    # - plot_temporal_trend_monthly_bookings_cancellations_split_by_hotel
    # - plot_development_bookings_by_month_by_hotels
    # - plot_development_cancellation_rate_by_month_by_hotel
    # - plot_development_bookings_by_weekday_by_hotel
    # - plot_development_cancellation_rate_by_weekday_by_hotel

# 3. Functions for displaying guest level information are:
    # - plot_guest_information_patterns_in_respect_to_cancellation_both_hotels
    # - plot_guest_information_patterns_in_respect_to_cancellation_split_by_hotels
    # - plot_stays_in_week_nights_of_guest_in_respect_to_cancellation_split_by_hotels
    # - plot_lead_time_and_adr_relationship_with_cancellation_both_hotels
    # - plot_specific_guest_needs_in_respect_to_cancellation_both_hotels
    # - plot_lead_time_and_adr_relationship_with_cancellation_split_by_hotels
    # - plot_specific_guest_needs_in_respect_to_cancellation_split_by_hotels



## ------------------------------------------- General functions for first overviews -------------------------------------------------------- ## 


# need to be imported as well
custom_palettes = {
        'hotel': {
            'Resort Hotel': '#99badf',
            'City Hotel': '#29a15c'
        },
        'deposit_type': {
            'No Deposit': '#99badf',
            'Refundable': '#29a15c',
            'Non Refund': '#073E7F'
        }
    }


# example usage: plot_cancellation_distribution_general_overview_bar(data)
def plot_cancellation_distribution_general_overview_bar(data):
    counts = data['is_canceled'].value_counts()
    labels = ['Not Canceled', 'Canceled']
    total = counts.sum()
    percentages = [(count / total) * 100 for count in counts]
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='is_canceled', data=data, palette=['#99badf', '#29a15c'])
    plt.title('Booking Cancellation Distribution')
    plt.xlabel('Is Canceled')
    plt.ylabel('Count')
    plt.xticks([0, 1], labels)
    for p, percent in zip(ax.patches, percentages):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 100,
                f'{percent:.1f}%', ha="center", fontsize=11)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#99badf', '#29a15c'])
    plt.title('Cancellation Ratio')
    plt.tight_layout()
    plt.show()
    print(f"Canceled: {counts[1]} bookings ({percentages[1]:.2f}%)")
    print(f"Not Canceled: {counts[0]} bookings ({percentages[0]:.2f}%)")





# Example usage: plot_cancellation_by_category_general_for_categorical_data(data, 'market_segment')
def plot_cancellation_by_category_general_for_categorical_data(data, column, palette='pastel'):
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=data, x=column, y='is_canceled', estimator='mean', palette=palette)
    plt.title(f'Cancellation Rate by {column.replace("_", " ").title()}')
    plt.ylabel('Cancellation Rate')
    plt.xticks(rotation=45)
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.text(p.get_x() + p.get_width() / 2., height + 0.01,
                    f'{height * 100:.1f}%', ha="center", fontsize=10)
    plt.tight_layout()
    plt.show()





# Example usage: plot_cancellation_by_hotel_for_categorical_data('deposit_type', 'hotel', 'is_canceled', 'Hotel vs. Deposit Type vs. Cancellation', data, custom_palettes['hotel'])
def plot_cancellation_by_hotel_for_categorical_data(x, hue, col, title, data, palette):
    subset_data = data[[x, hue, col]].dropna()
    # Create the FacetGrid
    g = sns.catplot(
        data=subset_data,
        x=x, hue=hue, col=col,
        kind='count', height=4, aspect=1.6,
        palette=palette,
        legend=False
    )
    # Titles and labels
    g.set_axis_labels(x.replace('_', ' ').title(), "Count")
    g.set_titles(col_template="{col_name} Cancellation")
    g.fig.suptitle(title, y=1.08)
    # Rotate x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    # Manual legend creation
    hue_order = subset_data[hue].unique()
    handles = [mpatches.Patch(color=palette[h], label=h) for h in hue_order]
    g.fig.legend(
        handles, [h.replace('_', ' ').title() for h in hue_order],
        title=hue.replace('_', ' ').title(),
        loc='center left',
        bbox_to_anchor=(1, 0.5),  # push legend further right
        frameon=True
    )
    plt.subplots_adjust(right=0.8, top=0.88)  # more space for legend
    plt.tight_layout()
    plt.show()





## ------------------------------------------- Functions for temporal trends -------------------------------------------------------- ## 


# Example usage: plot_temporal_trends_smoothed_14daysavg_bookings_cancellations_both_hotels(data)
def plot_temporal_trends_smoothed_14daysavg_bookings_cancellations_both_hotels(data, window=14):
    # Daily aggregation
    daily = data.groupby('arrival_date')['is_canceled'].agg(
        total_bookings='count',
        cancellations='sum'
    )
    daily['cancellation_rate'] = daily['cancellations'] / daily['total_bookings']
    daily_rolling = daily.rolling(window=window).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily_rolling.index, daily_rolling['total_bookings'], label='Total Bookings (Smoothed)', color='#073E7F')
    ax1.plot(daily_rolling.index, daily_rolling['cancellations'], label='Cancellations (Smoothed)', color='#29A15C')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Arrival Date')
    ax1.set_title(f'Bookings and Cancellations (Rolling {window}-Day Avg)')
    ax2 = ax1.twinx()
    ax2.plot(daily_rolling.index, daily_rolling['cancellation_rate'], label='Cancellation Rate', color='black', linestyle='--')
    ax2.set_ylabel('Cancellation Rate')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='center left', bbox_to_anchor=(1.08, 0.5),
        fontsize=10, title='Legend', frameon=True
    )
    plt.tight_layout()
    plt.show()





# Example usage: plot_temporal_trends_monthly_bookings_cancellations_both_hotels(data)
def plot_temporal_trends_monthly_bookings_cancellations_both_hotels(data):
    data['arrival_month'] = data['arrival_date'].dt.to_period('M')
    monthly = data.groupby('arrival_month')['is_canceled'].agg(
        total_bookings='count',
        cancellations='sum'
    ).astype(int)
    monthly['cancellation_rate'] = monthly['cancellations'] / monthly['total_bookings']

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(monthly.index.to_timestamp(), monthly['total_bookings'], label='Total Bookings', color='#073E7F', marker='o')
    ax1.plot(monthly.index.to_timestamp(), monthly['cancellations'], label='Cancellations', color='#29A15C', marker='o')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Arrival Month')
    ax1.set_title('Monthly Bookings and Cancellations')
    ax2 = ax1.twinx()
    ax2.plot(monthly.index.to_timestamp(), monthly['cancellation_rate'], label='Cancellation Rate', color='black', linestyle='--', marker='x')
    ax2.set_ylabel('Cancellation Rate')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='center left', bbox_to_anchor=(1.08, 0.5),
        fontsize=10, title='Legend', frameon=True
    )
    plt.tight_layout()
    plt.show()





# Example usage: plot_temporal_trends_smoothed_14daysavg_bookings_cancellations_split_by_hotels(data)
def plot_temporal_trends_smoothed_14daysavg_bookings_cancellations_split_by_hotels(data, window=14):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    y1_max, y2_max = 0, 0
    for hotel in ['Resort Hotel', 'City Hotel']:
        hotel_data = data[data['hotel'] == hotel]
        daily = hotel_data.groupby('arrival_date')['is_canceled'].agg(
            total_bookings='count',
            cancellations='sum'
        )
        daily['cancellation_rate'] = daily['cancellations'] / daily['total_bookings']
        rolling = daily.rolling(window=window).mean()
        y1_max = max(y1_max, rolling[['total_bookings', 'cancellations']].max().max())
        y2_max = max(y2_max, rolling['cancellation_rate'].max())
    y1_max = int((y1_max + 10) // 10 * 10)
    y2_max = round(y2_max + 0.05, 2)

    for i, hotel in enumerate(['Resort Hotel', 'City Hotel']):
        hotel_data = data[data['hotel'] == hotel]
        daily = hotel_data.groupby('arrival_date')['is_canceled'].agg(
            total_bookings='count',
            cancellations='sum'
        )
        daily['cancellation_rate'] = daily['cancellations'] / daily['total_bookings']
        rolling = daily.rolling(window=window).mean()
        ax = axes[i]
        ax.plot(rolling.index, rolling['total_bookings'], label='Total Bookings', color='#073E7F')
        ax.plot(rolling.index, rolling['cancellations'], label='Cancellations', color='#29A15C')
        ax.set_ylabel('Count')
        ax.set_ylim(0, y1_max)
        ax.set_title(f'{hotel} – Rolling {window}-Day Avg')
        ax2 = ax.twinx()
        ax2.plot(rolling.index, rolling['cancellation_rate'], label='Cancellation Rate', color='black', linestyle='--')
        ax2.set_ylabel('Cancellation Rate')
        ax2.set_ylim(0, y2_max)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc='center left', bbox_to_anchor=(1.08, 0.5),
            fontsize=10, title='Legend', frameon=True
        )
    plt.suptitle('Rolling Bookings & Cancellations by Hotel Type (Uniform Scale)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





# Example usage: plot_temporal_trend_monthly_bookings_cancellations_split_by_hotel(data)
def plot_temporal_trend_monthly_bookings_cancellations_split_by_hotel(data):
    data['arrival_month'] = data['arrival_date'].dt.to_period('M')
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    y1_max, y2_max = 0, 0

    for hotel in ['Resort Hotel', 'City Hotel']:
        hotel_data = data[data['hotel'] == hotel]
        monthly = hotel_data.groupby('arrival_month')['is_canceled'].agg(
            total_bookings='count',
            cancellations='sum'
        ).astype(int)
        monthly['cancellation_rate'] = monthly['cancellations'] / monthly['total_bookings']
        y1_max = max(y1_max, monthly[['total_bookings', 'cancellations']].max().max())
        y2_max = max(y2_max, monthly['cancellation_rate'].max())
    y1_max = int((y1_max + 100) // 100 * 100)
    y2_max = round(y2_max + 0.05, 2)

    for i, hotel in enumerate(['Resort Hotel', 'City Hotel']):
        hotel_data = data[data['hotel'] == hotel]
        monthly = hotel_data.groupby('arrival_month')['is_canceled'].agg(
            total_bookings='count',
            cancellations='sum'
        ).astype(int)
        monthly['cancellation_rate'] = monthly['cancellations'] / monthly['total_bookings']
        ax = axes[i]
        ax.plot(monthly.index.to_timestamp(), monthly['total_bookings'], label='Total Bookings', color='#073E7F', marker='o')
        ax.plot(monthly.index.to_timestamp(), monthly['cancellations'], label='Cancellations', color='#29A15C', marker='o')
        ax.set_ylabel('Count')
        ax.set_ylim(0, y1_max)
        ax.set_title(f'{hotel} – Monthly Bookings and Cancellations')
        ax2 = ax.twinx()
        ax2.plot(monthly.index.to_timestamp(), monthly['cancellation_rate'], label='Cancellation Rate', color='black', linestyle='--', marker='x')
        ax2.set_ylabel('Cancellation Rate')
        ax2.set_ylim(0, y2_max)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc='center left', bbox_to_anchor=(1.08, 0.5),
            fontsize=10, title='Legend', frameon=True
        )
    plt.suptitle('Monthly Bookings & Cancellations by Hotel Type (Uniform Scale)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





# def plot_bookings_over_time(data):
#     bookings_by_date = data['arrival_date'].value_counts().sort_index()
#     plt.figure(figsize=(12, 4))
#     bookings_by_date.plot()
#     plt.title('Number of Bookings Over Time')
#     plt.xlabel('Arrival Date')
#     plt.ylabel('Number of Bookings')
#     plt.tight_layout()
#     plt.show()





# def plot_bookings_over_time_by_hotel(data):
#     bookings = data.groupby(['arrival_date', 'hotel']).size().unstack(fill_value=0)
#     fig, ax = plt.subplots(figsize=(12, 4))
#     bookings.plot(ax=ax)
#     ax.set_title('Bookings Over Time by Hotel')
#     ax.set_xlabel('Arrival Date')
#     ax.set_ylabel('Number of Bookings')
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Hotel')
#     plt.tight_layout()
#     plt.show()






# def plot_bookings_by_month(data):
#     month_order = list(calendar.month_name)[1:]  # Jan to Dec
#     monthly = data['arrival_date_month'].value_counts().reindex(month_order)
#     plt.figure(figsize=(10, 4))
#     sns.barplot(x=monthly.index, y=monthly.values, palette='pastel')
#     plt.title('Total Bookings per Month')
#     plt.ylabel('Number of Bookings')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()





def plot_development_bookings_by_month_by_hotels(data):
    month_order = list(calendar.month_name)[1:]
    booking_counts = data.groupby(['arrival_date_month', 'hotel']).size().unstack().reindex(month_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    booking_counts.plot(kind='bar', width=0.8, ax=ax, color=['#99badf', '#29a15c'])
    booking_counts.plot(marker='o', linewidth=2, ax=ax)
    ax.set_title('Monthly Bookings by Hotel (Bar + Line)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Bookings')
    ax.set_xticklabels(booking_counts.index, rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Hotel')
    plt.tight_layout()
    plt.show()






# def plot_cancellation_rate_by_month(data):
#     # Use month number for sorting
#     cancel_rate = data.groupby(['arrival_date_month'])['is_canceled'].mean()
#     cancel_rate = cancel_rate.reindex(list(calendar.month_name)[1:])  # Ordered
#     plt.figure(figsize=(10, 4))
#     sns.barplot(x=cancel_rate.index, y=cancel_rate.values, palette='Set2')
#     plt.title('Cancellation Rate per Month')
#     plt.ylabel('Cancellation Rate')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()





def plot_development_cancellation_rate_by_month_by_hotel(data):
    cancel_rate = data.groupby(['arrival_date_month', 'hotel'])['is_canceled'].mean().unstack().reindex(list(calendar.month_name)[1:])
    fig, ax = plt.subplots(figsize=(10, 5))
    cancel_rate.plot(kind='bar', width=0.8, ax=ax, color=['#99badf', '#29a15c'])
    cancel_rate.plot(marker='o', linewidth=2, ax=ax)
    ax.set_title('Monthly Cancellation Rate by Hotel')
    ax.set_ylabel('Cancellation Rate')
    ax.set_xlabel('Month')
    ax.set_xticklabels(cancel_rate.index, rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Hotel')
    plt.tight_layout()
    plt.show()





# def plot_bookings_by_weekday(data):
#     data['weekday'] = data['arrival_date'].dt.day_name()
#     weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#     counts = data['weekday'].value_counts().reindex(weekday_order)
#     plt.figure(figsize=(10, 4))
#     sns.barplot(x=counts.index, y=counts.values, palette='pastel')
#     plt.title('Total Bookings by Day of the Week')
#     plt.ylabel('Number of Bookings')
#     plt.tight_layout()
#     plt.show()




def plot_development_bookings_by_weekday_by_hotel(data):
    data['weekday'] = data['arrival_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = data.groupby(['weekday', 'hotel']).size().unstack().reindex(weekday_order)
    fig, ax = plt.subplots(figsize=(10, 4))
    weekday_counts.plot(kind='bar', ax=ax, color=['#99badf', '#29a15c'])
    weekday_counts.plot(marker='o', ax=ax, linewidth=2)
    ax.set_title('Bookings by Weekday and Hotel')
    ax.set_ylabel('Number of Bookings')
    ax.set_xlabel('Weekday')
    ax.set_xticklabels(weekday_counts.index)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Hotel')
    plt.tight_layout()
    plt.show()





def plot_development_cancellation_rate_by_weekday_by_hotel(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    data['weekday'] = data['arrival_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cancel_rate = (
        data.groupby(['weekday', 'hotel'])['is_canceled']
        .mean()
        .unstack()
        .reindex(weekday_order)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    cancel_rate.plot(kind='bar', ax=ax, width=0.8, color=['#99badf', '#29a15c'])
    cancel_rate.plot(marker='o', linewidth=2, ax=ax)
    ax.set_title('Cancellation Rate by Weekday and Hotel')
    ax.set_ylabel('Cancellation Rate')
    ax.set_xlabel('Weekday')
    ax.set_xticklabels(cancel_rate.index)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Hotel')
    plt.tight_layout()
    plt.show()







## ------------------------------------------- Functions for displaying guest level information -------------------------------------------------------- ##







def plot_guest_information_patterns_in_respect_to_cancellation_both_hotels(data):
    # Prep features
    data['total_guests'] = data['adults'] + data['children'] + data['babies']
    data['has_children'] = (data['children'] + data['babies']) > 0
    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cancellation rate vs. adults
    sns.barplot(
        data=data,
        x='adults', y='is_canceled',
        palette='pastel', ax=axes[0, 0],
        ci=95
    )
    axes[0, 0].set_title('Cancellation Rate by Number of Adults')

    # 2. Cancellation rate vs. children
    sns.barplot(
        data=data,
        x='children', y='is_canceled',
        palette='pastel', ax=axes[0, 1],
        ci=95
    )
    axes[0, 1].set_title('Cancellation Rate by Number of Children')

    # 3. Cancellation rate vs. babies
    sns.barplot(
        data=data,
        x='babies', y='is_canceled',
        palette='pastel', ax=axes[1, 0],
        ci=95
    )
    axes[1, 0].set_title('Cancellation Rate by Number of Babies')

    # 4. Cancellation rate vs. total guests
    sns.barplot(
        data=data,
        x='total_guests', y='is_canceled',
        palette='pastel', ax=axes[1, 1],
        ci=95
    )
    axes[1, 1].set_title('Cancellation Rate by Total Guests')

    for ax in axes.flat:
        ax.set_ylabel('Cancellation Rate')
        ax.set_xlabel('')
        ax.set_ylim(0, 1)

    plt.suptitle('Cancellation Rates by Guest Composition (with 95% CI)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # --- With vs. without children ---
    plt.figure(figsize=(6, 4))
    sns.barplot(data=data, x='has_children', y='is_canceled', palette='Set2', ci=95)
    plt.xticks([0, 1], ['No Children', 'With Children'])
    plt.ylabel('Cancellation Rate')
    plt.title('Cancellation Rate: With vs. Without Children (95% CI)')
    plt.tight_layout()
    plt.show()
    # --- Repeated guest vs. new guest ---
    plt.figure(figsize=(6, 4))
    sns.barplot(data=data, x='is_repeated_guest', y='is_canceled', palette='Set3', ci=95)
    plt.xticks([0, 1], ['New Guest', 'Repeated Guest'])
    plt.ylabel('Cancellation Rate')
    plt.title('Cancellation Rate: New vs. Repeated Guest (95% CI)')
    plt.tight_layout()
    plt.show()







def plot_guest_information_patterns_in_respect_to_cancellation_split_by_hotels(data):
    # Prep features
    data['total_guests'] = data['adults'] + data['children'] + data['babies']
    data['has_children'] = (data['children'] + data['babies']) > 0

    hotels = data['hotel'].unique()
    features = ['adults', 'children', 'babies', 'total_guests']
    feature_titles = [
        'Adults', 'Children', 'Babies', 'Total Guests'
    ]
    for hotel in hotels:
        hotel_data = data[data['hotel'] == hotel]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
        for ax, feature, title in zip(axes, features, feature_titles):
            sns.barplot(
                data=hotel_data,
                x=feature,
                y='is_canceled',
                palette='pastel',
                ci=95,
                ax=ax
            )
            ax.set_title(f'{title} (95% CI)')
            ax.set_xlabel('')
            ax.set_ylabel('Cancellation Rate')
            ax.set_ylim(0, 1)
        fig.suptitle(f'{hotel} — Cancellation Rate by Guest Composition', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    # --- With vs. without children + Repeated guest ---
    for hotel in hotels:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        hotel_data = data[data['hotel'] == hotel]
        sns.barplot(data=hotel_data, x='has_children', y='is_canceled', palette='Set2', ci=95, ax=axes[0])
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['No Children', 'With Children'])
        axes[0].set_title('With vs. Without Children')
        axes[0].set_ylabel('Cancellation Rate')
        axes[0].set_ylim(0, 1)
        sns.barplot(data=hotel_data, x='is_repeated_guest', y='is_canceled', palette='Set3', ci=95, ax=axes[1])
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['New Guest', 'Repeated Guest'])
        axes[1].set_title('New vs. Repeated Guest')
        axes[1].set_ylabel('')
        fig.suptitle(f'{hotel} — Cancellation Rate by Customer Type (95% CI)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()






def plot_stays_in_week_nights_of_guest_in_respect_to_cancellation_split_by_hotels(data):
    data = data.copy()
    data['stay_bin'] = pd.cut(
        data['stays_in_week_nights'],
        bins=[0, 2, 4, 6, 10, 30],
        labels=['0–2', '3–4', '5–6', '7–10', '11+']
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=data,
        x='stay_bin',
        y='is_canceled',
        hue='hotel',
        ci=95,
        palette='pastel'
    )
    plt.title("Cancellation Rate by Weeknight Stay Duration")
    plt.ylabel("Cancellation Rate")
    plt.xlabel("Weeknight Stay Duration (Binned)")
    plt.tight_layout()
    plt.show()






def plot_lead_time_and_adr_relationship_with_cancellation_both_hotels(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        data=data,
        x='is_canceled',
        y='lead_time',
        palette=['#99badf', '#29a15c'],
        ax=axes[0]
    )
    axes[0].set_title('Lead Time vs. Cancellation')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Not Canceled', 'Canceled'])
    axes[0].set_ylabel('Lead Time')
    sns.boxplot(
        data=data,
        x='is_canceled',
        y='adr',
        palette=['#99badf', '#29a15c'],
        ax=axes[1]
    )
    axes[1].set_title('Average Daily Rate (ADR) vs. Cancellation')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Not Canceled', 'Canceled'])
    axes[1].set_ylabel('ADR (€)')
    plt.tight_layout()
    plt.show()






def plot_specific_guest_needs_in_respect_to_cancellation_both_hotels(data):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))  # Increased grid for 7 plots

    behavior_vars = [
        ('required_car_parking_spaces', 'Parking Spaces'),
        ('total_of_special_requests', 'Special Requests'),
        ('deposit_type', 'Deposit Type'),
        ('booking_changes', 'Booking Changes'),
        ('previous_cancellations', 'Previous Cancellations'),
        ('previous_bookings_not_canceled', 'Previous Non-Canceled Bookings'),
        ('days_in_waiting_list', 'Days in Waiting List'),
    ]

    for ax, (col, title) in zip(axes.flat, behavior_vars):
        plot_data = data[data[col] <= 4] if data[col].dtype != 'object' else data
        sns.countplot(
            data=plot_data,
            x=col,
            hue='is_canceled',
            palette=['#99badf', '#29a15c'],
            ax=ax
        )
        ax.set_title(f'{title} vs. Cancellation')
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel("Count")
        ax.legend(title='Canceled', labels=['No', 'Yes'])

    # Remove unused subplots if any
    for ax in axes.flat[len(behavior_vars):]:
        ax.remove()

    plt.suptitle('Booking Behavior vs. Cancellation (Both Hotels)', y=1.03, fontsize=16)
    plt.tight_layout()
    plt.show()






def plot_lead_time_and_adr_relationship_with_cancellation_split_by_hotels(data):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")
    hotels = ['Resort Hotel', 'City Hotel']
    palette = ['#99badf', '#29a15c']

    # Plot 1: Lead Time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, hotel in enumerate(hotels):
        subset = data[data['hotel'] == hotel]
        sns.boxplot(
            data=subset,
            x='is_canceled',
            y='lead_time',
            palette=palette,
            ax=axes[i]
        )
        axes[i].set_title(f'{hotel} — Lead Time vs. Cancellation')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Not Canceled', 'Canceled'])
        axes[i].set_ylabel('Lead Time')
    plt.suptitle('Lead Time vs. Cancellation (Hotel Comparison)', y=1.05)
    plt.tight_layout()
    plt.show()

    # Plot 2: ADR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, hotel in enumerate(hotels):
        subset = data[data['hotel'] == hotel]
        sns.boxplot(
            data=subset,
            x='is_canceled',
            y='adr',
            palette=palette,
            ax=axes[i]
        )
        axes[i].set_title(f'{hotel} — ADR vs. Cancellation')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Not Canceled', 'Canceled'])
        axes[i].set_ylabel('ADR (€)')
    plt.suptitle('ADR vs. Cancellation (Hotel Comparison)', y=1.05)
    plt.tight_layout()
    plt.show()







def plot_specific_guest_needs_in_respect_to_cancellation_split_by_hotels(data):

    sns.set(style="whitegrid")
    behavior_vars = [
        ('required_car_parking_spaces', 'Parking Spaces'),
        ('total_of_special_requests', 'Special Requests'),
        ('deposit_type', 'Deposit Type'),
        ('booking_changes', 'Booking Changes'),
        ('previous_cancellations', 'Previous Cancellations'),
        ('previous_bookings_not_canceled', 'Previous Non-Canceled Bookings'),
        ('days_in_waiting_list', 'Days in Waiting List')
    ]

    for col, title in behavior_vars:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, hotel in zip(axes, ['Resort Hotel', 'City Hotel']):
            subset = data[data['hotel'] == hotel]
            filtered = subset[subset[col] <= 4] if subset[col].dtype != 'object' else subset

            sns.countplot(
                data=filtered,
                x=col,
                hue='is_canceled',
                palette=['#99badf', '#29a15c'],
                ax=ax
            )
            ax.set_title(f'{title} — {hotel}')
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.legend(title='Canceled', labels=['No', 'Yes'])

        plt.suptitle(f'{title} vs. Cancellation (Both Hotels)', y=1.03, fontsize=15)
        plt.tight_layout()
        plt.show()















