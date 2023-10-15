import pandas as pd
from glob import glob

# Reading single csv file

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

single_file_gry = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# List all the file in data/raw/metamotion

files = glob("../../data/raw/MetaMotion/*.csv")
print(len(files))

# Extract features from the filename
data_path = "../../data/raw/MetaMotion\\"
f = files[0]

participent = f.split("-")[0].replace(data_path,"")

label = f.split("-")[1]

category = f.split("-")[2].rstrip("123")

df = pd.read_csv(f)

df['participent'] = participent
df['label'] = label
df['category'] = category

# read all files

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participent = f.split("-")[0].replace(data_path,"")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(f)
    
    df['participent'] = participent
    df['label'] = label
    df['category'] = category
    
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
        
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1 
        gyr_df = pd.concat([gyr_df, df])
    
    
    
# Working with Datetime

acc_df.info()

pd.to_datetime(df['epoch (ms)'], unit='ms')
pd.to_datetime(df['time (01:00)'])


acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# create a function 

files = glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_file(files):
    
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    
    acc_set = 1
    gyr_set = 1
    
    for f in files:
        participent = f.split("-")[0].replace(data_path,"")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
        df = pd.read_csv(f)
    
        df['participent'] = participent
        df['label'] = label
        df['category'] = category
    
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1 
            gyr_df = pd.concat([gyr_df, df])
            
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')
    
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_file(files)

# Margin Dataset

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

data_merged.columns = [
    'acc_x',
    'acc_y',
    'acc_z',
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'participent',
    'label',
    'category',
    'set'
]

# resampling data frequency conversion)
# Accelerometer:  12.50 MHz
#Gyrescope: 25.00 MHz

sampling = {
    'acc_x': 'mean',
    'acc_y': 'mean',
    'acc_z': 'mean',
    'gyr_x': 'mean',
    'gyr_y': 'mean',
    'gyr_z': 'mean',
    'participent':'last',
    'label': 'last',
    'category': 'last',
    'set': 'last'
}

#split by day

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resample = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])


data_resample['set'] = data_resample['set'].astype('int')
data_resample.info()


#Export the data set
data_resample.to_pickle("../../data/interim/01_data_processed.pkl")
