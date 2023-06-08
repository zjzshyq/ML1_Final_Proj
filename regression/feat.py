from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd
import numpy as np


def na_col_new(df, col):
    na_mask = df[col].isna()
    df[col + '_NA'] = np.where(na_mask, 1, 0)
    return df


def int_standard(df, col):
    if col == 'prenatal_care_month':
        fen_mu = 9
    elif col == 'number_prenatal_visits' \
            or col == 'cigarettes_before_pregnancy':
        fen_mu = 40
    elif col == 'father_age':
        fen_mu = 80
    else:
        fen_mu = 100
    df[col] = df[col].astype(float)/fen_mu
    df[col] = df[col].clip(upper=1)
    return df


def float_standard(df, col_lst):
    scaler = RobustScaler()
    df[col_lst] = scaler.fit_transform(df[col_lst])
    return df


def float_process(df: pd.DataFrame, col_lst: list):
    for col in col_lst:
        df[col].replace(0, np.nan, inplace=True)
        df = na_col_new(df, col)
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

        sd_value = df[col].std()
        threshold = 3
        df.loc[np.abs(df[col] - mean_value) > threshold * sd_value, col] = mean_value

        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        # if col == 'mother_delivery_weight':
        #     df[col] = df[col].astype(float) / 400
        # else:
        #     df[col] = df[col].astype(float) / 100
        # df[col] = df[col].clip(upper=1)
    return df


def int_process(df: pd.DataFrame, col_lst: list):
    for col in col_lst:
        # if col == 'prenatal_care_month':
        #     df[col].replace(99, np.nan, inplace=True)
        df = na_col_new(df, col)

        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

        sd_value = df[col].std()
        threshold = 3
        df.loc[np.abs(df[col] - mean_value) > threshold * sd_value, col] = mean_value
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]].astype(int))
        # df = int_standard(df, col)
    return df


def float_process2(df: pd.DataFrame, col_lst: list):
    for col in col_lst:
        df = na_col_new(df, col)
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

        sd_value = df[col].std()
        threshold = 3
        df.loc[np.abs(df[col] - mean_value) > threshold * sd_value, col] = mean_value
        scaler = RobustScaler()
        df[col] = scaler.fit_transform(df[[col]])
    return df


def int_process2(df: pd.DataFrame, col_lst: list):
    for col in col_lst:
        med_value = df[col].median()
        df[col].fillna(med_value, inplace=True)
        sd_value = df[col].std()
        threshold = 3
        df.loc[np.abs(df[col] - med_value) > threshold * sd_value, col] = med_value
        scaler = RobustScaler()
        df[col] = scaler.fit_transform(df[[col]])
    return df


def square_process(df: pd.DataFrame, col_lst: list):
    for col in col_lst:
        df[col+'_square'] = df[col].apply(lambda x: x**2)
    return df


def discrete_process(df, col_lst):
    for col in col_lst:
        uniques = df[col].unique()
        if len(uniques) > 2:
            if col == 'mother_marital_status':
                df[col] = df[col].fillna('unknown')
                df[col] = df[col].replace({1.0: 'couple', 2.0: 'single'})
            df = pd.get_dummies(df, columns=[col], prefix=col)
        else:
            if set(uniques) == {'F', 'M'}:
                df[col] = df[col].apply(lambda x: 1 if x == 'M' else 0)
    return df


def check_target(df, col):
    min_weight = 400
    max_weight = 5500
    count_min = df[df[col] < min_weight].shape[0]
    count_max = df[df[col] > max_weight].shape[0]

    ratio_min = count_min / df.shape[0]
    ratio_max = count_max / df.shape[0]

    print("样本数量小于下界的数量:", count_min)
    print("样本数量大于上界的数量:", count_max)
    print("样本数量小于下界的比例:", ratio_min)
    print("样本数量大于上界的比例:", ratio_max)

    df.loc[df[col] < min_weight, col] = min_weight
    df.loc[df[col] > max_weight, col] = max_weight
    return df


def outliers4target(df, col):
    # df = check_target(df, col)
    mean_value = df[col].mean()
    sd_value = df[col].std()
    threshold = 4.5
    # df.loc[np.abs(df[col] - mean_value) > threshold * sd_value, col] = mean_value
    original_count = len(df)
    df = df.drop(df[np.abs(df[col] - mean_value) > threshold * sd_value].index)
    dropped_count = original_count - len(df)
    print(f"Dropped {dropped_count} samples")
    return df


def mk_feat_pure(df):
    float_lst = ['mother_body_mass_index', 'mother_delivery_weight',
                 'mother_height', 'mother_weight_gain']
    int_lst = ['father_age', 'cigarettes_before_pregnancy',
               'number_prenatal_visits', ]
    discrete_lst = ['mother_race', 'father_education', 'prenatal_care_month',
                    'previous_cesarean', 'mother_marital_status',
                    'newborn_gender']

    df = float_process(df, float_lst)
    df = int_process(df, int_lst)
    df = discrete_process(df, discrete_lst)

    significant_feat = ['mother_delivery_weight', 'mother_height', 'mother_weight_gain', 'number_prenatal_visits']
    df = square_process(df, significant_feat)

    return df
