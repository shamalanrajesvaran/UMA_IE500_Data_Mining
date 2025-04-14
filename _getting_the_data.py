import os
from kaggle.api.kaggle_api_extended import KaggleApi

# In case you need it, as never used Kaggle API:
# Automatically move kaggle.json to the right place (you will need to create it on your Kaggle Account)
# kaggle_json_path = 'path_to_your_downloaded/kaggle.json'  # replace with actual path (the downloaded kaggle.json API file)
# os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
# os.system(f'cp {kaggle_json_path} ~/.kaggle/kaggle.json')
# os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

#API
api = KaggleApi()
api.authenticate()

#Download
api.dataset_download_files('jessemostipak/hotel-booking-demand', path='hotel_booking_data', unzip=True)

