import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    def clean_data(self):
        self.df = self.df.drop_duplicates()

        self.df.fillna({
            "Status": "Unknown",
            "delivery_date": "Not Delivered"
        }, inplace=True)

        self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors='coerce')
        self.df["delivery_date"] = pd.to_datetime(self.df["delivery_date"], errors='coerce')

        self.df["Delivery Time"] = (
            self.df["delivery_date"] - self.df["order_date"]
        ).dt.days

        return self.df
        