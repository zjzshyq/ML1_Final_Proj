from feat import mk_feat_pure
import pandas as pd
import lightgbm as lgb
import re
import time

print('Loading train data...')
train_df = pd.read_csv('../../data/newborn_train.csv', sep=',')\
    .rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

print('Loading test data...')
test_df = pd.read_csv('../../data/newborn_test.csv', sep=',')\
    .rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

print('Feature engineering...')
train = mk_feat_pure(train_df)
test_X = mk_feat_pure(test_df)
train_y = train['newborn_weight']
train_X = train.drop('newborn_weight', axis=1)

print('Training...')
start_time = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mape',
    'max_depth': 8,
    'num_leaves': 30,
    'learning_rate': 0.12,
    'n_estimators': 130,
    'reg_alpha': 0.2,
    'reg_lambda': 0.5,
    'subsample': 0.98,
    'colsample_bytree': 0.97
}

model = lgb.LGBMRegressor(**params)
model.fit(train_X, train_y)
print('Train cost:', time.time()-start_time)

print('Predicting...')
y_pred = model.predict(test_X)
test_y = pd.DataFrame({'pred_weight': y_pred})
test_y.to_csv('newborn_pred.csv', index=False)
print('Output successfully.')
