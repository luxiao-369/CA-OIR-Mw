import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_OIR.csv')
X= df[['R1','R2','R3','Metal_NPA','d_cp-M','d_B-M','E_els','E_rep','LUMO','1-octene','Al/M','T']]
Y = df[['OIR']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True,random_state=42)

train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)

train_data_activity = 'train_data_OIR.csv'
test_data_activity = 'test_data_OIR.csv'

train_df.to_csv(train_data_activity, index=False, encoding='utf-8')
test_df.to_csv(test_data_activity, index=False, encoding='utf-8')