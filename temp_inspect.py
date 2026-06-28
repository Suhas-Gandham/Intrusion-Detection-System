import os
import pandas as pd
os.chdir(r'c:/Users/Harshitha/Downloads/sem 5/innopro')
for fn in ['UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv']:
    try:
        df = pd.read_csv(fn, nrows=1)
        print(fn)
        print(df.columns.tolist())
    except Exception as e:
        print('ERROR', fn, e)
