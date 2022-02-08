import pandas as pd
#
lo=0
hi=20
for i in range(lo,hi):
  df1=pd.read_csv('SRG_vote_result_matrix_{}.csv'.format(i))
  if i==0:
      temp=df1
      i+=1
  else:
      temp=pd.concat([temp,df1],axis=0)
      print(temp.shape)
temp.to_csv('SRG_vote_result_matrix.csv'.format(lo,hi),index=None)
