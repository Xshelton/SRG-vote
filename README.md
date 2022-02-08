# SRG-vote
## welcome to the webpage of SRG-vote, 
![SRGvote5 edited framework](https://user-images.githubusercontent.com/33061177/152988170-7378c75a-b7db-412d-8f32-416e65f17740.png)
## citation
-------------
---------------------
----------------------------------
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
## Results
due to the limitation of 25MB of file for github.  
the file is divided into small files.  
use the combine.py to concate those files into one file.  
