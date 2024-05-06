import yfinance as yf
import datetime
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy.signal import medfilt

filename_csv = "data.csv"


def download(ticker_code, ticker_shortname, description):
    start = datetime.datetime.now() - datetime.timedelta(days=30)
    # start = datetime.datetime(2008, 1, 1)
    # Загрузите исторические данные для желаемого тикера
    df = yf.download(ticker_code, start)
    df.drop("Adj Close", axis=1, inplace=True)
    df.drop("Close", axis=1, inplace=True)
    df["Open"] = df["Open"] / df["High"]
    df["Low"] = df["Low"] / df["High"]
    df.columns = [f'{ticker_shortname}_{col}' for col in df.columns]

    # Просмотр данных
    # print(df)
    return df


print('Stage 1: download last data.')

df2 = download('^GSPC', 'SP5oo', "shares of 500 largest public companies")

df_s = download('^FCHI', 'CAC40', "French stock index, share prices of 40 largest companies")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^FTSE', 'FTSE100', "UK stock index, share prices of largest companies")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^GDAXI', 'DaxG', "DAX PERFORMANCE-INDEX, Germany  stock index, share prices of largest companies")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^DJI', 'Dow', "Dow Jones Industrial Average")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^RUT', 'Russe', "Russell 2000")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^IXIC', 'NASDc', "NASDAQ Composite")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('GC=F', 'Gold', "Gold Jun 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('SPPP', 'Platin', "Sprott Physical Platinum & Palladium Tr")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('SI=F', 'Silver', "Silver Jul 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('HG=F', 'Copper', "Copper  Jul 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('CL=F', 'Oil', "Crude Oil Jun 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('NG=F', 'Gas', "Natural Gas Jun 24")
df2 = df2.merge(df_s, left_index=True, right_index=True)
commodity_cols = ['day_of_week', 'Gold', 'Platin', 'Silver', 'Copper', 'Oil', 'Gas']

df_s = download('USDJPY=X', 'JPY', "USD/JPY - доллар к йене")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('EURUSD=X', 'EUR', "USD/EUR - доллар к евро")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('GBP=X', 'GBP', "USD/GBP - доллар к фунту")
df2 = df2.merge(df_s, left_index=True, right_index=True)

# for col in df2.columns:
#     if df2[col].dtype == 'float64':
#         df2[col] = df2[col].astype('float32')

if os.path.exists(filename_csv):
    print('Stage 1.1: downloading previously saved historical data.')
    df1 = pd.read_csv(filename_csv, parse_dates=['Date'], index_col='Date')
    # for col in df1.columns:
    #     if df1[col].dtype == 'float64':
    #         df1[col] = df1[col].astype('float32')

    merged_df = pd.concat([df1, df2])
    df2 = merged_df.loc[~merged_df.index.duplicated(keep='first')]
    del merged_df
else:
    print(f"File {filename_csv} not found")

print('Stage 1.2: save data to csv.')
df2.to_csv(filename_csv)

print('Stage 2: prepare data.')
# Сортировка DataFrame по индексу
df2.sort_index(inplace=True)

# Изменение частоты данных на один день
df2 = df2.resample('D').asfreq()
# 0 превратим в пропущенные значения и просто интерполируем
df2.replace(0.0, np.nan, inplace=True)
df2 = df2.interpolate()

# Feature engenering

df2['day_of_week'] = df2.index.dayofweek

for col in df2.columns:
    if 'Low' in col:
        df2[col + '_mean14'] = df2[col].rolling(window=14).mean()
    if 'High' in col:
        df2[col + '_lag7'] = df2[col].shift(7)
        df2[col + '_lag14'] = df2[col].shift(14)
        df2[col + '_lag28'] = df2[col].shift(28)
        # смотрим насколько большой разброс у параметра High за последние 30 дней
        df2[col + '_rolling_std'] = df2[col].rolling(window=30).std()
        # смотрим насколько большой разброс у параметра High за последние 7 дней
        df2[col + '_rolling_std'] = df2[col].rolling(window=7).std()
        # В этом примере, shift(14) сдвигает данные на 14 дней вперед, что означает, что для каждой даты мы смотрим на данные,
        # начиная с 28 дней назад и заканчивая 14 днями назад. rolling(window=14) создает скользящее окно размером 14 дней, и
        # mean() вычисляет среднее значение в этом окне.
        df2[col + '_rolling1414_mean_delta'] = df2[col].shift(14).rolling(window=14).mean()-df2[col]
        # делаем усреднение за 3 дней и отнимаем от сегодя
        df2[col + '_rolling_mean_delta'] = df2[col].rolling(window=3).mean()-df2[col]
    if 'Volume' in col:
        df2[col + '_divided'] = df2[col] / df2[col].shift(28).rolling(window=28).mean()
        # В этом примере, shift(14) сдвигает данные на 14 дней вперед, что означает, что для каждой даты мы смотрим
        # на данные, начиная с 28 дней назад и заканчивая 14 днями назад. rolling(window=14) создает скользящее окно
        # размером 14 дней, и mean() вычисляет среднее значение в этом окне.
        df2[col + '_rolling_mean'] = df2[col].shift(14).rolling(window=14).mean()
        # делаем усреднение за 7 дней и отнимаем от более старой истории
        df2[col + '_rolling_mean_delta'] = df2[col].rolling(window=7).mean() - df2[col + '_rolling_mean']
    df2 = df2.copy()

df2 = df2.interpolate()  # еще раз интерполируем для заполнения
df2 = df2.fillna(df2.mean())

for col in df2.columns:
    if df2[col].isna().sum() > 0:
        df2.drop(col, axis=1, inplace=True)
        # print(f'Колонка {col} имеет NaN, удалили колонку')

# На данном моменте у нас сформирован dataset

print('Stage 3: process data.')


def get_scores(df2: pd.DataFrame):
    numpy_array = df2.values.astype('float32')
    scaler = StandardScaler()
    numpy_array = scaler.fit_transform(numpy_array)

    num_components = numpy_array.shape[1] // 4  # Вы можете изменить количество компонент
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(numpy_array)

    combined_data = np.concatenate((numpy_array, transformed_data), axis=1)

    lof = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.01)  # Вы можете изменить количество соседей
    lof.fit(combined_data[:-40]) # тренируем не на всей истории, последние 40 дней игнорируем!

    # anomaly_scores7 = - lof.decision_function(combined_data[-7:])

    anomaly_scores = -1 * lof.decision_function(combined_data)

    # Применяем медианную фильтрацию с использованием скользящего окна размером 3
    anomaly_scores_filtered = medfilt(anomaly_scores, kernel_size=3)
    return anomaly_scores, anomaly_scores_filtered


import plotly.graph_objects as go

anomaly_scores, anomaly_scores_filtered = get_scores(df2)
print('\nScores for last 7 days (a high near 0 result means an anomaly):')
print(anomaly_scores[-7:])

# Создаем объект графа
fig = go.Figure()

# renge_days = len(df2.index) # full range
renge_days = 365

# Добавляем линию на график
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=anomaly_scores[-renge_days:], mode='lines',
                         line=dict(color='#D3D3D3', dash='dash'), opacity=0.7, name='Raw Anomaly Scores'))
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='blue'), name='Filtered\nAnomaly Scores'))


def remove_substring_columns(df, substrings: []):
    for substring in substrings:
        df = df[df.columns[~df.columns.str.contains(substring)]]
    return df


def copy_substring_columns(df, substrings: []):
    new_df = pd.DataFrame()
    for substring in substrings:
        matched_columns = df.columns[df.columns.str.contains(substring)]
        new_df = pd.concat([new_df, df[matched_columns]], axis=1)
    return new_df


moneys_cols = ['day_of_week', 'JPY', 'EUR', 'GBP']
df_small = copy_substring_columns(df2, moneys_cols)
money_anomaly_scores, money_anomaly_scores_filtered = get_scores(df_small)
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=money_anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='red'), opacity=0.2, name='Foreign exchange anomalies'))

europe_cols = ['day_of_week', '^FCHI', '^FTSE', '^GDAXI', 'EUR']
df_small = copy_substring_columns(df2, europe_cols)
eur_anomaly_scores, eur_anomaly_scores_filtered = get_scores(df_small)
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=eur_anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='green'), opacity=0.3, name='European zone anomalies'))

df_small = copy_substring_columns(df2, commodity_cols)
commodity_anomaly_scores, commodity_anomaly_scores_filtered = get_scores(df_small)
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=commodity_anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='darkgoldenrod'), opacity=0.5, name='Commodity anomalies'))

# Добавляем заголовок графика
fig.update_layout(
    title_text='Result of stock market anomaly detection (a high result means an anomaly)<BR><sub>Note that these are '
               'not absolute sales or price values, but the degree of abnormality of changes in values and the '
               'abnormality of the whole pattern on a particular day in history.</sub>')

# Сохраняем график в виде HTML-файла
fig.write_html('plot.html')
print('Stage 4: plot.html saved.')
