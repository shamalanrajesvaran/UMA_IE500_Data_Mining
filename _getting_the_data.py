import os
from kaggle.api.kaggle_api_extended import KaggleApi

# OPTIONAL: Automatically move kaggle.json to the right place
kaggle_json_path = 'path_to_your_downloaded/kaggle.json'  # replace with actual path
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
os.system(f'cp {kaggle_json_path} ~/.kaggle/kaggle.json')
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# Initialize API
api = KaggleApi()
api.authenticate()

# Download dataset
api.dataset_download_files('jessemostipak/hotel-booking-demand', path='hotel_booking_data', unzip=True)

print("✅ Dataset downloaded and extracted into 'hotel_booking_data' folder.")