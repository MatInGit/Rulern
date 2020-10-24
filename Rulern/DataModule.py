import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# this module strealines the preamble needed to read in the data

class DataReader:
    def __init__(self,file,conv = True,date_cols = [],find_diff = True,dorpnan = False,class_col= [],skip_cols=[]):
        self.data = pd.read_csv(file)
        self.data.replace('', "NONE", inplace=True)
        if dorpnan:

            #print(self.data.head())
            self.data.dropna(axis=0,how='any',inplace=True)                     #remove empty rows
        #print(self.data.head())
        self.data_raw = self.data                                               #make a copy of raw data
        self.data.columns = self.data.columns.str.replace(' ', '_')             #replace all spaces
        self.le_dict = {}

        if conv:                                                                #convert if conversion is enabled
            for col in self.data.columns:
                if self.data[col].dtype == 'object' and col not in date_cols and col in class_col and col not in skip_cols:
                    #print(self.data[col])
                    self.le_dict[col] = LabelEncoder()
                    self.data[col] = self.le_dict[col].fit_transform(self.data[col].astype(str))
                if col in date_cols:
                    self.data[col] = pd.to_datetime(self.data[col],infer_datetime_format = True)

        if find_diff == True:
            for col in date_cols:
                #self.data[col+"_delta"][0] = 0
                self.data[col+"_delta"] = self.data[col] - self.data[col].shift(1)
                self.data.at[0,col+"_delta"] = np.timedelta64(0)
                self.data[col+"_delta"] = self.data[col+"_delta"].dt.days.astype(int)
                #self.data[col+"_delta"][0]
        self.data.reset_index(inplace=True,drop=True)

    def desc(self):
        return self.data.describe()

    def column_names(self):
        return self.data.columns
