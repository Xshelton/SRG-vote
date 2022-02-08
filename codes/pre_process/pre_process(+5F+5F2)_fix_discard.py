
#mode1:
#input the embedding file name of mirna/gene
#input the positive samples and negative samples dataset（the whole dataset)
#output the embedding file for the training/validating(one file)
#mode2（five-fold cross validation):output the embedding file for the training80%/validating10%/testing%(two files* 5fold for each epoch)
#mode3（five-fold cross validation-typeB):output the embedding file for the discard, training/validating, testing file( 1 discard+(2 files)*5 for each epoch)
#parameters seeds for np, mirna_file,gene_file,dataset_name
import pandas as pd
from numpy.random import seed
from sklearn.model_selection import train_test_split
def embedding2dataset(mirnafile,genefile,randomseed,dataset,outputname):
           seed(randomseed)
           test_mode='On'
           df_gene=pd.read_csv(genefile)
           label_gene=df_gene['label']
           df_gene=df_gene.drop(['label'],axis=1)
           df_mirna=pd.read_csv(mirnafile)
           label_mirna=df_mirna['label']
           df_mirna=df_mirna.drop(['label'],axis=1)
           print('successfully load embedding data file for mirna/gene')
           label_gene_list=label_gene.values
           label_gene_list=label_gene_list.tolist()
           label_mirna_list=label_mirna.values
           label_mirna_list=label_mirna_list.tolist()
           key=1
           list_label=[]
           if key==1:
             df_dataset=pd.read_csv(dataset)
             try:
              D_mirna=df_dataset['mirna']
              D_mirna=D_mirna.values
              D_mirna=D_mirna.tolist()
              D_gene=df_dataset['gene']
              D_gene=D_gene.values
              D_gene=D_gene.tolist()
              D_label=df_dataset['label']
             except:
              D_mirna=df_dataset['0_mirna']
              D_mirna=D_mirna.values
              D_mirna=D_mirna.tolist()
              D_gene=df_dataset['1_gene']
              D_gene=D_gene.values
              D_gene=D_gene.tolist()
              D_label=df_dataset['label']   
             miss=0
             print('successfully load dataset for mirna/gene/label')
           else:
             print('dataset load error:make sure the dataset with /mirna/gene as header')
           print('embedding dataset generation begin...')
           for i in range(0,len(df_dataset)):
               
               try:
                   mirnaname=D_mirna[i]
                   genename=D_gene[i]
                   mirna_index=label_mirna_list.index(mirnaname)
                   gene_index=label_gene_list.index(genename)
                   if D_label[i]>0:
                      list_label.append({'mirna':mirnaname,'gene':genename,'label':1})
                   else:
                      list_label.append({'mirna':mirnaname,'gene':genename,'label':0})
                   feature1=df_mirna[mirna_index:mirna_index+1]
                   feature1=feature1.reset_index(drop=True)
                   feature2=df_gene[gene_index:gene_index+1]
                   feature2=feature2.reset_index(drop=True)
                   if i==0:
                       temp0=pd.concat([feature1,feature2],axis=1,join='outer')
                   else:
                       temp1=pd.concat([feature1,feature2],axis=1,join='outer')
                       temp0=pd.concat([temp0,temp1],axis=0)
                   if i%500==0 and i!=0:
                    if len(temp0)!=None:
                      print('total',len(temp0),'/',len(D_mirna),'miss number',miss)
               except:
                  miss+=1
           fea_label=pd.DataFrame(list_label)
           Y=fea_label
           temp0=temp0.reset_index(drop=True)
           X=temp0
           temp0=pd.concat([temp0,fea_label],axis=1)#
           temp0.to_csv(outputname,index=None)
           if test_mode== 'On':
            FIVE=pd.concat([X,Y],axis=1)
            FIVE.to_csv('{}_FIVE.csv'.format(outputname[0:-4]),index=None)
            X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.1, random_state=0)#5
            Train_set=pd.concat([X_train,y_train],axis=1)
            Train_set.to_csv(outputname,index=None)
            Test_set=pd.concat([X_test,y_test],axis=1)
            Test_set.to_csv('{}_test.csv'.format(outputname[0:-4]),index=None)
def five_fold_crossvalidation(outputname,seq,ex):#feed X into theses return five 
    #filename='hsa+neg'
    df=pd.read_csv('{}_FIVE.csv'.format(outputname[0:-4]))
    columns=df.columns
    label=df['label']
    df=df.sample(frac=1).reset_index(drop=True)#reindex df
    X=df.values
    Y=label.values
    def details(Y):
        #print('length of Y',len(Y))
        count0=0
        count1=0
        MM=Y.reset_index(drop=True)
        MM=MM.values
        MM=MM.tolist()
        for i in range(0,len(MM)):
              #print(type(MM[i]),MM[i])
              if MM[i][0]==0:
                  #print(i,count0)
                  count0=count0+1
              if MM[i][0]==1:
                  #print(i,count0)
                  count1=count1+1
        return count0,count1
   # kf = 
    print('read_end')
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    KF=StratifiedKFold(n_splits=5, random_state=seq, shuffle=True)
    #KF=KFold(n_splits=5)  #establish KFOLD
    count=0
    for train_index,test_index in KF.split(X,Y):
      print("TRAIN:",train_index,"TEST:",test_index)
      
      X_train,X_test=X[train_index],X[test_index]
      Y_train,Y_test=Y[train_index],Y[test_index]
      X_train=pd.DataFrame(X_train)
      X_test=pd.DataFrame(X_test)
      Y_train=pd.DataFrame(Y_train)
      Y_test=pd.DataFrame(Y_test)
      #print('first_row_of_Y-test',Y_test[0])
      c1,c2=details(Y_train)
      c3,c4=details(Y_test)
      print('Y_train_details','0:',c1,'1:',c2,',Y_test_details','0:',c3,'1:',c4)#show details about the dataset
      #print(Y_test)
      count+=1
      X_train.columns=columns
      X_test.columns=columns
      if ex==1:
        X_train.to_csv('{}_epoch_{}_{}.csv'.format(outputname[0:-4],seq,count),index=None)
        X_test.to_csv('{}_epoch_{}_{}_test.csv'.format(outputname[0:-4],seq,count),index=None)

        
def five_fold_crossvalidation_type2(outputname,seed1,ex):#feed X into theses return five file
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    from sklearn import model_selection
    #filename='hsa+neg'
    seed(seed1)
    partion=0.1#discard data
    df=pd.read_csv('{}_FIVE.csv'.format(outputname[0:-4]))#read the original file
    columns=df.columns
    label=df['label']
    df=df.sample(frac=1).reset_index(drop=True)#shuffle the df
    X=df.values
    Y=label.values
    def details(Y):#this function is to see the details of the label Y
        #print('length of Y',len(Y))
        count0=0
        count1=0
        MM=Y.reset_index(drop=True)
        MM=MM.values
        MM=MM.tolist()
        for i in range(0,len(MM)):
              #print(type(MM[i]),MM[i])
              if MM[i][0]==0:
                  #print(i,count0)
                  count0=count0+1
              if MM[i][0]==1:
                  #print(i,count0)
                  count1=count1+1
        return count0,count1
    print('read_end')
    #we discard 
    X_remain,X_discard,Y_remain,Y_discard=\
          model_selection.train_test_split(X, Y, 
          train_size=1-partion, test_size=0.1, random_state=seed1,stratify = Y)
    print(X_discard.shape)
    print(Y_discard.shape)
    print(type(X_discard))
    X_discard_pd=pd.DataFrame(X_discard)
    #X_discard_pd['label']=Y_discard
    # Before the K fold , 
    KF=StratifiedKFold(n_splits=5, random_state=seed1, shuffle=True)
    #KF=KFold(n_splits=5)  # establish KFOLD
    count=0
    for train_index,test_index in KF.split(X_remain,Y_remain):
      print("TRAIN:",train_index,"TEST:",test_index)
      
      X_train,X_test=X[train_index],X[test_index]
      Y_train,Y_test=Y[train_index],Y[test_index]
      X_train=pd.DataFrame(X_train)
      X_test=pd.DataFrame(X_test)
      Y_train=pd.DataFrame(Y_train)
      Y_test=pd.DataFrame(Y_test)
      #print('first_row_of_Y-test',Y_test[0])
      c1,c2=details(Y_train)
      c3,c4=details(Y_test)
      print('Y_train_details','0:',c1,'1:',c2,',Y_test_details','0:',c3,'1:',c4)#show details about the dataset
      #print(Y_test)
      count+=1
      X_train.columns=columns
      X_test.columns=columns
      #X_discard_pd.rename(columns={'256':'mirna'},inplace=True)
      #X_discard_pd.rename(columns={'257':'gene'},inplace=True)
      #X_discard_pd.rename(columns={'258':'label'},inplace=True)
      X_discard_pd.to_csv('{}_discard_{}.csv'.format(outputname[0:-4],seed1),index=None)
      if ex==1:
        X_train.to_csv('{}_epoch_{}_{}.csv'.format(outputname[0:-4],seed1,count),index=None)
        X_test.to_csv('{}_epoch_{}_{}_test.csv'.format(outputname[0:-4],seed1,count),index=None)



#this will generate the training dataset_from WHOLE-dataset,given mirna/gene embedding.(the DATASET has to be generated before)
#embedding2dataset('gcn_mirna_plusnetwork_degree-C1-L2bs.csv','GCN-PPI-4.910-features-role2vec-local-cluster5-bs.csv',0,'SGLSTM-whole-DATASET（P+N).csv','GCNGCN-SGlstmwhole.csv')
#the parameters are:
#embedding-mirna
#embdding-gene
#randomseed
#file
#outputname
for i in range(0,5):
 #for type A, it will not discard first, for type B, it will discard 10% of the dataset first.then do the FIVE fold in the rest of the data.
 #five_fold_crossvalidation(outputname,seq,ex)#seed of divi/ ex:export to csv（1 True,0 false) GCNGCN-SGlstmwhole_FIVE
 #for type B, it will discard some data first, have to use the change_last_three.py to adjust the dicarded label before testing
 five_fold_crossvalidation_type2('r2rhn-sglstmwhole.csv',seed1=i,ex=1)#given a file name 'r2rhn-sglstmwhole_FIVE.csv', the input name should be "r2rhn-sglstmwhole.csv" here.
