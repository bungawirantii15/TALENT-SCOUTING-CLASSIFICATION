import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_dataset(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Hapus kolom identitas jika ada
    for col in ["NO", "NAMA"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Encoder
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.fit_transform(df[['JENIS KELAMIN (PA/PI)']])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['JENIS KELAMIN (PA/PI)']))

    df = pd.concat([df.drop(columns=['JENIS KELAMIN (PA/PI)']), encoded_df], axis=1)
    
    encoder_label = LabelEncoder()
    df['POTENSI'] = encoder_label.fit_transform(df['POTENSI'])
    df.info()

    # Normalisasi
    categorical_columns = ['POTENSI', 'JENIS KELAMIN (PA/PI)_PI']
    number_columns = [col for col in df.columns if col not in categorical_columns]
    scaler = MinMaxScaler()
    df[number_columns] = scaler.fit_transform(df[number_columns])
    
    #split data
    #pisah feature dan class pada data
    df_x, df_y= df.drop(columns='POTENSI'), df['POTENSI']

    #pisahkan train dan test
    df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=42, stratify=df_y)

    return df_train_x, df_test_x, df_train_y, df_test_y
