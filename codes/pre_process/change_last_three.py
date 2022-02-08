# before testing typeB cross validation's prediction, use this program to change the last three columns of the discarded dataset.
import pandas as pd
def change_last_three(file):
    df=pd.read_csv('{}'.format(file))
    print(df.columns)
    df.rename(columns={'256':'gene'},inplace=True)
    df.rename(columns={'257':'label'},inplace=True)
    df.rename(columns={'258':'mirna'},inplace=True)
    df.to_csv('{}'.format(file),index=None)
    print(df)

for i in range(3,5):
 change_last_three('r2rhn-sglstmwhole_discard_{}.csv'.format(i))
