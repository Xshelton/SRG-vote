# SRG-vote
## welcome to the webpage of SRG-vote, 
![SRGvote6 edited framework](https://user-images.githubusercontent.com/33061177/153282930-295cdfd2-03b5-401c-883e-ac72a3f6fa25.png)

## citation
Please cite our paper if your find the features/code/results useful.
Thanks in advance.
Xie W, Zheng Z, Zhang W, et al. SRG-vote: Predicting miRNA-gene relationships via embedding and LSTM ensemble[J]. IEEE Journal of Biomedical and Health Informatics, 2022.
##

## Features
We provide 6 files, including:  
1.miRNA doc2vec features 128  
2.miRNA role2vec features 128  
3.miRNA GCN features 128  
4.Gene doc2vec features 128(derived from 3'UTR sequence)  
5.Gene role2vec features 128  
6.gene GCN features 128


## Datasets
By using the Features and Datasets, we can get the dataset for the five-fold cross validation.(TypeA,B)  

## Code
Fisrt use the preprocess to generate 10 files(CV typeA) or 11 files(CV typeB, with one more discarded file)  
CV typeB could be used to generate scores for test files the discarded files, as well as novel pairs( as long as those pairs are with the same format as training set)  
TF1.13 version has been tested in the env of Tensorflow 1.13  
TF2.3 version has been tested in the env of Tensorflow 2.3 (with the latest nvidia driver).  
## Models
The predict models from files(miRNA embedding files, and gene embedding files,generate automatically and produce the scores, only tf1.13 version for now)
![image](https://user-images.githubusercontent.com/33061177/153279013-f56ce762-665b-4114-9c96-dbc1d916519a.png)
How to use the Model.
First prepare the csvfile of gene and miRNA.
Secondly, unzip the models.
If you want to retrain the whole model:  
the train_pre function should be like this:  
def train_pre(file,i,MMode):  
   tf.reset_default_graph()   
   train_it(file,0,0,i,'train_all_dataset for the predict',MMode)
   tf.reset_default_graph()   
   auc=train_it(file,0,0,i,'predict',MMode)  
   tf.reset_default_graph()   
   return auc    
   
 If you want to regenerate the score:  
 #delete or denote the first 3 lines.Then run the program.  
 def train_pre(file,i,MMode):  
   auc=train_it(file,0,0,i,'predict',MMode)  
   tf.reset_default_graph()   
   return auc  
 
## Results
due to the limitation of 25MB of file for github.  
the file is divided into small files.  
use the combine.py to concate those files into one file.  
