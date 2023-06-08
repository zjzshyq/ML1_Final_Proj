from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

ready_dir = 'newborn_test_y.csv'
pred_y_dir = 'newborn_pred.csv'
real_y = np.array(pd.read_csv(ready_dir)).flatten()
pred_y = np.array(pd.read_csv(pred_y_dir)).flatten()

mape = mean_absolute_percentage_error(real_y, pred_y)*100
print('mape:%.6f%%' % mape)
