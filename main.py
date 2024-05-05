import yfinance as yf
import datetime
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

filename_csv = "data.csv"
def download(ticker_code, ticker_shortname, description):
    # start = datetime.datetime(2008, 1, 1)
    start = datetime.datetime.now() - datetime.timedelta(days=30)
    # Загрузите исторические данные для желаемого тикера
    df = yf.download(ticker_code, start)
    df.drop("Adj Close", axis=1, inplace=True)
    df.drop("Close", axis=1, inplace=True)
    df["Open"]=df["Open"]/df["High"]
    df["Low"]=df["Low"]/df["High"]
    df.columns = [f'{ticker_shortname}_{col}' for col in df.columns]

    # Просмотр данных
    # print(df)
    return df


df2=download('^GSPC', 'SP5oo',"акции 500 крупнейших публичных компаний")

df_s=download('^FCHI', 'CAC40',"фондовым индекс Франции, цены акций 40 крупнейших компаний")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('^DJI', 'Dow',"Dow Jones Industrial Average")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('^RUT', 'Russe',"Russell 2000")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('^IXIC', 'NASDc',"NASDAQ Composite")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('GC=F', 'Gold',"Gold Jun 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('SPPP', 'Platin',"Sprott Physical Platinum & Palladium Tr")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('SI=F', 'Silver',"Silver Jul 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('HG=F', 'Copper ',"Copper  Jul 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('CL=F', 'Oil',"Crude Oil Jun 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('NG=F', 'Gas',"Natural Gas Jun 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('USDJPY=X', 'JPY',"USD/JPY - доллар к йене")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('EURUSD=X', 'EUR',"USD/EUR - доллар к евро")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s=download('GBP=X', 'GBP',"USD/GBP - доллар к фунту")
df2 = df2.merge(df_s, left_index=True, right_index=True)

# for col in df2.columns:
#     if df2[col].dtype == 'float64':
#         df2[col] = df2[col].astype('float32')

if os.path.exists(filename_csv):
    df1 = pd.read_csv(filename_csv, parse_dates=['Date'], index_col='Date')
    # for col in df1.columns:
    #     if df1[col].dtype == 'float64':
    #         df1[col] = df1[col].astype('float32')

    merged_df = pd.concat([df1, df2])
    df2 = merged_df.loc[~merged_df.index.duplicated(keep='first')]
    del merged_df
else:
    print(f"File {filename_csv} not found")

df2.to_csv(filename_csv)


# Сортировка DataFrame по индексу
df2.sort_index(inplace=True)

# Изменение частоты данных на один день
df2 = df2.resample('D').asfreq()
# 0 превратим в пропущенные значения и просто интерполируем
df2.replace(0.0, np.nan, inplace=True)
df2=df2.interpolate()

# Feature engenering

df2['day_of_week'] = df2.index.dayofweek

for col in df2.columns:
    if 'Low' in col:
        df2[col+'_mean14'] = df2[col].rolling(window=14).mean()
    if 'High' in col:
        df2[col+'_lag7'] = df2[col].shift(7)
        df2[col+'_lag28'] = df2[col].shift(28)
        # смотрим насколько большой разброс у параметра High за последние 30 дней
        df2[col+'_rolling_std'] = df2[col].rolling(window=30).std()
        # смотрим насколько большой разброс у параметра High за последние 7 дней
        df2[col+'_rolling_std'] = df2[col].rolling(window=7).std()
        # В этом примере, shift(14) сдвигает данные на 14 дней вперед, что означает, что для каждой даты мы смотрим на данные, начиная с 28 дней назад и заканчивая 14 днями назад. rolling(window=14) создает скользящее окно размером 14 дней, и mean() вычисляет среднее значение в этом окне.
        df2[col+'_rolling_mean'] = df2[col].shift(14).rolling(window=14).mean()
        # делаем усреднение за 7 дней и отнимаем от более старой истории
        df2[col + '_rolling_mean_delta'] = df2[col].rolling(window=7).mean() - df2[col+'_rolling_mean']
    if 'Volume' in col:
        df2[col+'_divided'] = df2[col]/df2[col].shift(28).rolling(window=28).mean()
        # В этом примере, shift(14) сдвигает данные на 14 дней вперед, что означает, что для каждой даты мы смотрим на данные, начиная с 28 дней назад и заканчивая 14 днями назад. rolling(window=14) создает скользящее окно размером 14 дней, и mean() вычисляет среднее значение в этом окне.
        df2[col+'_rolling_mean'] = df2[col].shift(14).rolling(window=14).mean()
        # делаем усреднение за 7 дней и отнимаем от более старой истории
        df2[col + '_rolling_mean_delta'] = df2[col].rolling(window=7).mean() - df2[col+'_rolling_mean']
    df2=df2.copy()

df2=df2.interpolate() # еще раз интерполируем для заполнения
df2 = df2.fillna(df2.mean())

for col in df2.columns:
    if df2[col].isna().sum() > 0:
        df2.drop(col, axis=1, inplace=True)
        # print(f'Колонка {col} имеет NaN, удалили колонку')


numpy_array = df2.values.astype('float32')
scaler = StandardScaler()
numpy_array = scaler.fit_transform(numpy_array)

from sklearn.decomposition import PCA
pca = PCA(n_components=42) # Вы можете изменить количество компонент
transformed_data = pca.fit_transform(numpy_array)

combined_data = np.concatenate((numpy_array, transformed_data), axis=1)

lof = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.01) # Вы можете изменить количество соседей
lof.fit(combined_data[:-40])

anomaly_scores = lof.decision_function(combined_data[-7:])
print('Scores for last 7 days (a high near 0 result means an anomaly):')
print(-anomaly_scores)

anomaly_scores = lof.decision_function(combined_data)

from scipy.signal import medfilt

# Применяем медианную фильтрацию с использованием скользящего окна размером 3
anomaly_scores_filtered = medfilt(anomaly_scores, kernel_size=7)

import plotly.graph_objects as go
# Создаем объект графа
fig = go.Figure()

renge_days=len(df2.index)
renge_days=365

# Добавляем линию на график
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=-anomaly_scores[-renge_days:], mode='lines', line=dict(color='#D3D3D3', dash='dash'),opacity=0.7, name='Raw Anomaly Scores'))
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=-anomaly_scores_filtered[-renge_days:], mode='lines', line=dict(color='blue'),name='Filtered\nAnomaly Scores'))

# Добавляем заголовок графика
fig.update_layout(title_text='Result of stock market anomaly detection (a high result means an anomaly)')

# Сохраняем график в виде HTML-файла
fig.write_html('plot.html')

