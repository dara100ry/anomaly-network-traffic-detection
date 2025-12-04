# src/load_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler


columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label"
]
useless_features = [    
    "num_outbound_cmds",   # همیشه صفره
    "is_host_login",       # تقریبا همیشه صفره
    "land",                # همیشه صفر یا خیلی نادر
    # "su_attempted",        # خیلی کم اتفاق میفته
    # "num_shells",          # خیلی نادر
    # "num_file_creations",  # خیلی کم استفاده
    # "num_access_files",    # خیلی کم
    # "root_shell",          # خیلی نادر
    # "urgent",              # تقریبا همیشه صفر
    # "wrong_fragment"       # خیلی نادر
    "dst_host_diff_srv_rate",
    "hot",
    "dst_host_srv_count",
    "count"
    ]

def load_nsl_kdd(file_path):

        
    # difficulty
    df = pd.read_csv(file_path, names=columns, sep=',', usecols=range(42))
    df.dropna(inplace=True)


    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')


    encoders={}
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(['label'] + useless_features, axis=1)
    y = df['label']
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, encoders, scaler


def load_nsl_kdd_test(file_path, encoders, scaler):
    # بارگذاری مثل قبل
    df = pd.read_csv(file_path, names=columns, sep=',', usecols=range(42))
    df.dropna(inplace=True)
    # LabelEncoder برای ستون‌های متنی
    # if encoders is None:
    #     encoders = {}
    #     for col in ['protocol_type', 'service', 'flag']:
    #         le = LabelEncoder()
    #         df[col] = le.fit_transform(df[col])
    #         encoders[col] = le
    # else:
    #     for col in ['protocol_type', 'service', 'flag']:
    #         le = encoders[col]
    #         df[col] = le.transform(df[col])

    for col in ['protocol_type', 'service', 'flag']:
        df[col] = encoders[col].transform(df[col])


    df['label'] = df['label'].apply(lambda x: 'attack' if x != 'normal' else 'normal')


    X = df.drop(['label'] + useless_features, axis=1)
    y = df['label']

    X = scaler.transform(X)

    return X, y


# New
def load_nsl_kdd_raw(file_path):
    df = pd.read_csv(file_path, names=columns, sep=',', usecols=range(42))
    df.dropna(inplace=True)

    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    X_df = df.drop(['label'] + useless_features, axis=1)
    y = df['label']
    return X_df, y


def load_nsl_kdd_test_raw(file_path):
    df = pd.read_csv(file_path, names=columns, sep=',', usecols=range(42))
    df.dropna(inplace=True)

    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    X_df = df.drop(['label'] + useless_features, axis=1)
    y = df['label']
    return X_df, y