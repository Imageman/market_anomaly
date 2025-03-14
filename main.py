from loguru import logger
import sys
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
from datetime import date

TRESHOLD_FOR_STD_COLS = 3.0

filename_csv = "data.csv"

DOWNLOAD_full_range=True
fit_days_ignore = -60 # number of last day to ignore in fit

def download(ticker_code, ticker_shortname, description):
    logger.debug(f"Load {ticker_shortname}")
    start = datetime.datetime.now() - datetime.timedelta(days=30)
    if DOWNLOAD_full_range: start = datetime.datetime(2008, 1, 1)
    # Загрузите исторические данные для желаемого тикера
    df = yf.download(ticker_code, start, auto_adjust=False)
    df.drop("Adj Close", axis=1, inplace=True)
    df.drop("Close", axis=1, inplace=True)
    df["Open"] = df["Open"] / df["High"]
    df["Low"] = df["Low"] / df["High"]
    df.columns = [f'{ticker_shortname}_{col}' for col in df.columns]

    if  len(df)<3: 
        logger.warning(f'{ticker_shortname}: row count { len(df)}')
        exit(1)
    if  len(df.columns)<3: logger.warning(f'{ticker_shortname}:col count { len(df.columns)}')


    # Просмотр данных
    # print(df)
    return df


def remove_substring_columns(df, substrings: []):
    for substring in substrings:
        df = df[df.columns[~df.columns.str.contains(substring)]]
    return df

def copy_substring_columns(df, substrings: []):
    new_df = pd.DataFrame()
    for substring in substrings:
        matched_columns = df.columns[df.columns.str.contains(substring)]
        if len(matched_columns) == 0:
            logger.warning(f'copy_substring_columns: {substring} not found')
        new_df = pd.concat([new_df, df[matched_columns]], axis=1)
    return new_df


def process_combined_data(combined_data, threshold):
    try:
        # Ensure combined_data is a 2D numpy array
        if not isinstance(combined_data, np.ndarray) or combined_data.ndim != 2:
            raise ValueError("combined_data must be a 2D NumPy array.")

        # logger.debug(f"Initial combined_data:\n{combined_data}")

        # Take the absolute values
        abs_data = np.abs(combined_data)
        # logger.debug(f"Absolute value of data:\n{abs_data}")

        # Check if there are at least three rows
        if abs_data.shape[0] < 3:
            raise ValueError(
                "combined_data must have at least three rows to compute the median of the last three rows.")

        # Compute the median of the last three rows
        last_three_rows = abs_data[-3:]
        median_last_three_rows = np.median(last_three_rows, axis=0)
        logger.debug(f"Median of the last three rows:\n{median_last_three_rows}")

        # Find indices of columns where median values are greater than the threshold
        column_indices = np.nonzero(median_last_three_rows > threshold)[0]
        logger.info(f"Column indices where median value > threshold ({threshold}): {column_indices}")

        return column_indices

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None


def get_column_names(df, column_indices):
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a Pandas DataFrame.")

        column_names = []
        for i in column_indices:
            if i < len(df.columns):
                column_names.append(df.columns[i])
            else:
                column_names.append(f'PCA_cols{i}')

        logger.info(f"Column names: {column_names}")
        return column_names

    except Exception as e:
        logger.error(f"An error occurred while getting column names: {str(e)}")
        return None

def get_scores(df2: pd.DataFrame):
    numpy_array = df2.values.astype('float32')
    scaler = StandardScaler()
    numpy_array = scaler.fit_transform(numpy_array)

    num_components = numpy_array.shape[1] // 4  # Вы можете изменить количество компонент
    pca = PCA(n_components=num_components, random_state=123)
    transformed_data = pca.fit_transform(numpy_array)
    transformed_data = scaler.fit_transform(transformed_data)

    combined_data = np.concatenate((numpy_array, transformed_data), axis=1)

    std_anomale_cols=process_combined_data(combined_data, TRESHOLD_FOR_STD_COLS)
    std_anomale_cols=get_column_names(df2,std_anomale_cols)
    lof = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.01)  # Вы можете изменить количество соседей

    lof.fit(combined_data[:fit_days_ignore]) # тренируем не на всей истории, последние x дней игнорируем!

    anomaly_scores = -1 * lof.decision_function(combined_data)

    # Применяем медианную фильтрацию с использованием скользящего окна размером 3
    anomaly_scores_filtered = medfilt(anomaly_scores, kernel_size=3)
    return anomaly_scores, anomaly_scores_filtered, std_anomale_cols


logger.remove()
logger.add("finance.log", rotation="21 MB", retention=3, compression="zip", backtrace=True,   diagnose=True)  # Automatically rotate too big file
try:
    logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{message}</level>", level='INFO')
except  Exception as e:
    logger.debug(f'logger.add(sys.stdout) Error: {str(e)}')

logger.info('Start Finance')

print('Stage 1: download last data.')

df2 = download('^GSPC', 'SP5oo', "shares of 500 largest public companies")

df_s = download('^FCHI', 'CAC40', "French stock index, share prices of 40 largest companies")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^FTSE', 'FTSE100', "UK stock index, share prices of largest companies")
df2 = df2.merge(df_s, left_index=True, right_index=True)

df_s = download('^GDAXI', 'DaxG', "DAX PERFORMANCE-INDEX, Germany  stock index, share prices of largest companies")
df2 = df2.merge(df_s, left_index=True, right_index=True)

europe_cols = ['day_of_week', 'CAC40', 'FTSE100', 'DaxG', 'EUR']


df_s = download('^DJI', 'Dow', "Dow Jones Industrial Average")
df2 = df2.merge(df_s, left_index=True, right_index=True)

# отслеживает 2000 малых компаний, которые находятся в индексе Russell 3000. для отслеживания малых компаний в США.
df_s = download('^RUT', 'Russe', "Russell 2000")
df2 = df2.merge(df_s, left_index=True, right_index=True)

# включает почти все акции, которые котируются на фондовой бирже NASDAQ; сильно склоняется к компаниям в секторе информационных технологий
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

def get_merge(data_frame:pd.DataFrame, yachoo_code:str, name:str, description:str):
    df_s = download(yachoo_code, name, description)
    data_frame = data_frame.merge(df_s, left_index=True, right_index=True)
    return data_frame

df2 = get_merge(df2,'^VIX','VIX','индекс CBOE Volatility Index (VIX)')
df2 = get_merge(df2,'^VXN','CVX','индекс CBOE NASDAQ Volatility Index')
df2 = get_merge(df2,'^TNX','TNX','доходность 10-летних казначейских облигаций США')
df2 = get_merge(df2,'^FVX','FVX','кривая доходности казначейских облигаций')
df2 = get_merge(df2,'^IRX','IRX','кривая доходности муниципальных облигаций')
df2 = get_merge(df2,'BTC-USD','Bitcoin','Bitcoin USD')


# for col in df2.columns:
#     if df2[col].dtype == 'float64':
#         df2[col] = df2[col].astype('float32')
# делаем переименовывание колонок в старый формат
df2.columns = df2.columns.str.replace(r"_\('([^']+)',\s*'[^']+'\)", r"_\1", regex=True)


if os.path.exists(filename_csv):
    print('Stage 1.1: downloading previously saved historical data.')
    df1 = pd.read_csv(filename_csv, parse_dates=['Date'], index_col='Date')
    # for col in df1.columns:
    #     if df1[col].dtype == 'float64':
    #         df1[col] = df1[col].astype('float32')

    merged_df = pd.concat([df1, df2])
    df2 = merged_df.loc[~merged_df.index.duplicated(keep='last')]
    del merged_df
else:
    logger.info(f"File {filename_csv} not found")

print('Stage 1.2: save data to csv.')
df2.to_csv(filename_csv)

print('Stage 2: prepare data.')
# Сортировка DataFrame по индексу
df2.sort_index(inplace=True)

# Изменение частоты данных на один день
df2 = df2.resample('D').asfreq()
# 0 превратим в пропущенные значения и просто интерполируем
df2.replace(0.0, np.nan, inplace=True)
# df2 = df2.interpolate()
# Используем метод fillna с forward fill (ffill)
# Это заполнит пропущенные значения последним известным значением
df2 = df2.ffill()

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

# df2 = df2.interpolate()  # еще раз интерполируем для заполнения
# Используем метод fillna с forward fill (ffill)
# Это заполнит пропущенные значения последним известным значением
df2 = df2.ffill()
df2 = df2.fillna(df2.mean())

for col in df2.columns:
    if df2[col].isna().sum() > 0:
        df2.drop(col, axis=1, inplace=True)
        # print(f'Колонка {col} имеет NaN, удалили колонку')

# На данном моменте у нас сформирован dataset

print('Stage 3: process data.')

import plotly.graph_objects as go

anomaly_scores, anomaly_scores_filtered, std_anomale_cols = get_scores(df2)
logger.info('\nScores for last 7 days (a high near 0 result means an anomaly):')
logger.info(anomaly_scores[-7:])

fig = go.Figure()

renge_days = len(df2.index) # full range
# renge_days = 365

# Добавляем линию на график
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=anomaly_scores[-renge_days:], mode='lines',
                         line=dict(color='#D3D3D3', dash='dash'), opacity=0.7, name='Raw Anomaly Scores'))
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='blue'), name='Filtered\nAnomaly Scores'))


moneys_cols = ['day_of_week', 'JPY', 'EUR', 'GBP','Bitcoin']
df_small = copy_substring_columns(df2, moneys_cols)
money_anomaly_scores, money_anomaly_scores_filtered, _ = get_scores(df_small)
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=money_anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='red'), opacity=0.2, name='Foreign exchange anomalies'))

df_small = copy_substring_columns(df2, europe_cols)
eur_anomaly_scores, eur_anomaly_scores_filtered, _ = get_scores(df_small)
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=eur_anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='green'), opacity=0.3, name='European zone anomalies'))

df_small = copy_substring_columns(df2, commodity_cols)
commodity_anomaly_scores, commodity_anomaly_scores_filtered, _ = get_scores(df_small)
fig.add_trace(go.Scatter(x=df2.index[-renge_days:], y=commodity_anomaly_scores_filtered[-renge_days:], mode='lines',
                         line=dict(color='darkgoldenrod'), opacity=0.5, name='Commodity anomalies'))

# Добавляем заголовок графика
fig.update_layout(
    title_text='Result of stock market anomaly detection (a high result means an anomaly)<BR><sub>Note that these are '
               'not absolute sales or price values, but the degree of abnormality of changes in values and the '
               'abnormality of the whole pattern on a particular day in history.</sub>')

if len(std_anomale_cols)>0:
    current_date = date.today()
    date_string = current_date.strftime("%d.%m.%Y")
    annotations=f'{date_string}<BR>Anomalous columns:<BR>{"<BR>".join(std_anomale_cols)}'
    # Определение аннотаций
    annotations = [
        dict(
            xref='paper',
            yref='paper',
            x=1.02,  # немного правее правого края графика
            y=0,  # размещаем аннотацию внизу графика
            xanchor='left',
            yanchor='bottom',
            text=annotations,
            showarrow=False,
            align='left',
            font=dict(size=12)
        )
    ]
    fig.update_layout(annotations=annotations, legend=dict(
        x=1.02,
        y=1,
        traceorder='normal',
        font=dict(
            size=12,
        ),
    ))

# Сохраняем график в виде HTML-файла
fig.write_html('plot.html')
print('Stage 4: plot.html saved.')

final_score = (np.max(anomaly_scores[-7:]) + np.max(anomaly_scores[-3:]) + anomaly_scores[-1]) / 3

# Открываем файл (или создаем, если его нет) в режиме добавления
with open('res.csv', 'a') as f:
    # Добавляем новую строку
    f.write(f'{datetime.datetime.now().date().strftime("%Y-%m-%d")}, {str(final_score)}, {anomaly_scores[-1]}, {commodity_anomaly_scores[-1]}\n')

# Проверяем, существует ли файл
if os.path.exists('result.txt'):
    # Удаляем файл
    os.remove('result.txt')


if datetime.datetime.now().date().day==5:
    # Находим индекс максимального элемента
    index_max = np.argmax(anomaly_scores[-30:])

    # Получаем значение в колонке 'Date' (это у нас индекс!)
    date_value = df2.index[len(df2) - 30 + index_max]
    msg=f'In the last 30 days, the maximum anomaly  {np.max(anomaly_scores[-30:]):.3} was {date_value.strftime("%Y-%m-%d")}'
    logger.info(f'Тестовое сообщение: {msg}')
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(msg)

if final_score>0.6 or anomaly_scores[-1]>0.7:
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(f'{datetime.datetime.now().date().strftime("%Y-%m-%d")}, {str(final_score)}, {anomaly_scores[-1]}')

logger.info('Fin.\n')
