# src/load_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

def load_nsl_kdd(file_path):

        
    # خواندن فایل CSV که با "," جدا شده + نادیده گرفتن ستون اضافی (difficulty)
    df = pd.read_csv(file_path, names=columns, sep=',', usecols=range(42))

        # حذف ردیف‌های ناقص
    df.dropna(inplace=True)

    
    # برچسب‌ها: normal / attack
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # تبدیل categorical‌ها
    encoders={}
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # ویژگی‌ها و برچسب
    X = df.drop('label', axis=1)
    y = df['label']

    # نرمال‌سازی
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, encoders


def load_nsl_kdd_test(file_path, encoders, scaler=None):
    # بارگذاری مثل قبل
    df = pd.read_csv(file_path, names=columns, sep=',', usecols=range(42))
    df.dropna(inplace=True)
    # LabelEncoder برای ستون‌های متنی
    if encoders is None:
        encoders = {}
        for col in ['protocol_type', 'service', 'flag']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in ['protocol_type', 'service', 'flag']:
            le = encoders[col]
            df[col] = le.transform(df[col])

    # لیبل‌ها
    df['label'] = df['label'].apply(lambda x: 'attack' if x != 'normal' else 'normal')

    X = df.drop(['label'], axis=1)
    y = df['label']

    # نرمال‌سازی با همان اسکیلر آموزش
    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y
