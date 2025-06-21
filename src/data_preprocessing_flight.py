import pandas as pd

def preprocess_data(df):
    df = df.dropna()

    df['Journey_Date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['Journey_Day'] = df['Journey_Date'].dt.day
    df['Journey_Month'] = df['Journey_Date'].dt.month
    df = df.drop(['date', 'Journey_Date'], axis=1)

    df['Dep_Hour'] = pd.to_datetime(df['dep_time'], format='%H:%M').dt.hour
    df['Dep_Minute'] = pd.to_datetime(df['dep_time'], format='%H:%M').dt.minute
    df = df.drop('dep_time', axis=1)

    df['Arrival_Hour'] = pd.to_datetime(df['arr_time'], format='%H:%M').dt.hour
    df['Arrival_Minute'] = pd.to_datetime(df['arr_time'], format='%H:%M').dt.minute
    df = df.drop('arr_time', axis=1)

    df['Duration'] = df['time_taken'].apply(convert_duration)
    df = df.dropna(subset=['Duration'])
    df = df.drop('time_taken', axis=1)

    df = pd.get_dummies(df, drop_first=True)

    return df

def convert_duration(duration):
    try:
        h, m = 0, 0
        duration = duration.lower().replace(" ", "")
        if 'h' in duration:
            h_split = duration.split('h')
            h = int(h_split[0])
            duration = h_split[1] if len(h_split) > 1 else ''
        if 'm' in duration:
            m = int(duration.split('m')[0])
        return h * 60 + m
    except:
        return None
