import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from numpy.random import seed
import pandas as pd
from tensorflow.random import set_seed
#from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
global count_i
count_i=0
global count_i2
count_i2=0

def train_it(filename,seed1,seed2,iteration,mode,model,list_opt,predict_mode):
    import tensorflow as tf
    #setting 
    seed(seed1)
    set_seed(seed2)
   # training_iters = iteration
    epo=iteration
    #default setting
    lr = 0.001                #learning rate，用于梯度下降
    batch_size = 128
    n_inputs = 16
    n_steps = 16
    n_hidden_units = 128
    n_classes = 2
    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
    tppath="./models/models_predict{}_{}/train_all_for_predict.ckpt".format(filename[0:-4],iteration)
    #predict_save_file='./outcome/models_{}_{}/{}_AUC{}.csv'.format(filename[0:20],iteration,iteration,aucvalue)
    mode_save_file="./models/models_predict{}_{}".format(filename[0:-4],iteration)
   ## predict_file='
    if model=='lstm':
     weights = {
    # (28, 128)
    'in': tf.Variable(tf.random.normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random.normal([n_hidden_units, n_classes]))
      }
     biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
     }
    if model=='bilstm':
        weights = {'out': tf.Variable(tf.random.normal([2*n_hidden_units, n_classes]))}
        biases = {'out': tf.Variable(tf.random.normal([n_classes]))}
 
    def RNN(X, weights, biases):
     X = tf.reshape(X, [-1, n_inputs])
     X_in = tf.matmul(X, weights['in']) + biases['in']
     X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
     cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden_units)  #num_units
     init_state = cell.zero_state(batch_size, dtype=tf.float32)
     outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
     outputs = tf.unstack(tf.transpose(a=outputs, perm=[1,0,2]))
     results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
     return results
    def BiRNN(x, weights, biases):
      x = tf.unstack(x, n_steps, 1)
      lstm_fw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    # Backward direction cell
      lstm_bw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0)

    # Get lstm cell output
      try:
        outputs, _, _ = tf.compat.v1.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
      except Exception: # Old TensorFlow version only returns outputs not states
        outputs = tf.compat.v1.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
      return tf.matmul(outputs[-1], weights['out']) + biases['out']
    if model=='lstm':
     pred = RNN(x, weights, biases)
     cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(y)))
     train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)
     correct_pred = tf.equal(tf.argmax(input=pred, axis=1), tf.argmax(input=y, axis=1))
     accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))
     init = tf.compat.v1.global_variables_initializer()
    if model=='bilstm':
        pred = BiRNN(x, weights, biases)
        prediction = tf.nn.softmax(pred)
        cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(y)))
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(cost)
        correct_pred = tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(input=y, axis=1))
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))
        init = tf.compat.v1.global_variables_initializer()
        print('using bi-lstm model')
    def batches(batch_size, features, labels):
               global list_bx
               global list_by
               assert len(features) == len(labels)
               output_batches = []
               list_bx=[]
               list_by=[]
               sample_size = len(features)
               for start_i in range(0, sample_size, batch_size):
                   end_i = start_i + batch_size
                   batch_x = features[start_i:end_i]
                   list_bx.append(batch_x)
                   #print(type(batch_x))
                   batch_y = labels[start_i:end_i]
                   #output_batches.append(batch)
                   list_by.append(batch_y)
    def next_batches(i):
            global list_bx
            global list_by
            global count_i
            if count_i<len(list_bx)-2:
                batch_x=list_bx[count_i]
                batch_y=list_by[count_i]
                count_i+=1
            else:
                batch_x=list_bx[count_i]
                batch_y=list_by[count_i]
                count_i=0
            batch_x=batch_x.values
            batch_y=batch_y.values
            return batch_x,batch_y
    def next_test_batches(i):
            global test_bx
            global test_by
            global count_i2
            if count_i2<len(test_bx)-2:
                batch_x=test_bx[count_i2]
                batch_y=test_by[count_i2]
                count_i2+=1
            else:
                batch_x=test_bx[count_i2]
                batch_y=test_by[count_i2]
                count_i2=0
            batch_x=batch_x.values
            batch_y=batch_y.values
            return batch_x,batch_y
    def test_batches(batch_size, features, labels):
            global test_bx
            global test_by
            assert len(features) == len(labels)
            output_batches = []
            test_bx=[]
            test_by=[]
            sample_size = len(features)
            for start_i in range(0, sample_size, batch_size):
                end_i = start_i + batch_size
                batch_x = features[start_i:end_i]
                test_bx.append(batch_x)
                #print(type(batch_x))
                batch_y = labels[start_i:end_i]
                #output_batches.append(batch)
                test_by.append(batch_y) 
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    df=pd.read_csv(filename)
    print('before shuffle',len(df))
    df=df.sample(frac=1) #shuffle the dataset first 
    print('after shuffle',len(df))
    df=df.reset_index(drop=True)
    print(df)
    label=df['label']
    try:
     df=df.drop(['0_mirna'],axis=1)
     df=df.drop(['1_gene'],axis=1)
     df=df.drop(['label'],axis=1)
    except:
       df=df.drop(['mirna'],axis=1)
       df=df.drop(['gene'],axis=1)
       df=df.drop(['label'],axis=1) 
    list_label2=[]
    #change the label into [1,0]
    for i in range(0,len(label)):
       if label[i]==1:
            list_label2.append(0)
       else:
            list_label2.append(1)
    list_2_pd=pd.DataFrame(list_label2)
    label=pd.concat([label,list_2_pd],axis=1)
    #build dataset
    if mode=='test_bi':
        X=df
        Y=label
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=0)
        test_batches(batch_size,X_test,y_test)
        batches(batch_size, X_train, y_train)
        with tf.compat.v1.Session() as sess:
             sess.run(init)
             for step in range(0, epo+1):
                     batch_x, batch_y = next_batches(step)
                     batch_x = batch_x.reshape((batch_size, n_steps, n_inputs))
                     #print(batch_x.shape)
                     # Run optimization op (backprop)
                     sess.run([train_op], feed_dict={x: batch_x, y: batch_y})
                     if step % 10 == 0 or step == 1:
                         # Calculate batch loss and accuracy
                         loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
                         print('训练集结果')
                         print("Step " + str(step) + ", Minibatch Loss= " + \
                               "{:.4f}".format(loss) + ", Training Accuracy= " + \
                               "{:.3f}".format(acc))
                         print('测试集结果',step)
                         x_test_batch,y_test_batch=next_test_batches(step)
                         print(step,"这里的i的结果是")
                         x_test_batch = x_test_batch.reshape([batch_size, n_steps, n_inputs])
                         print(sess.run(accuracy, feed_dict={
                         x: x_test_batch,
                         y: y_test_batch,
            
                         }))
                     #print("Optimization Finished!")
               #  saver.save(sess=sess, save_path="./models/my_data.ckpt", global_step=step+1)
    if mode=='train_all_dataset for the predict':
        X_train=df
        y_train=label
        batches(batch_size, X_train, y_train)
        save_point=len(df)
        print(save_point)
        with tf.compat.v1.Session() as sess:
             sess.run(init)
             step = 0
             while step * batch_size < epo*save_point:
                batch_xs, batch_ys = next_batches(step)
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                sess.run([train_op], feed_dict={x: batch_xs,y: batch_ys,})
                if step % 400 == 0:
                    #peak
                   print('training the dataset',step)
                   loss,acc=sess.run([cost,accuracy], feed_dict={x: batch_xs,y: batch_ys,})
                   print(loss,acc)
                   #save per200 step
                   #model save for the test dataset@path 
                   saver.save(sess=sess, save_path="./models/models_predict{}_{}_{}/train_all_for_predict.ckpt".format(filename[0:-4],epo,model), global_step=step+1)
                   print('model begin to save,epoch equals to',step*128//save_point)
                      
                step += 1
    if mode=='train_some_dataset_for validation':
              X=df
              Y=label
              X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=0)#5 fold cross validation
              print(len(X_test))#一开始有6388个测试集
              test_batches(batch_size,X_test,y_test)#test set should also be considered
              batches(batch_size, X_train, y_train)
              save_point=len(df)
              print(save_point)
              with tf.compat.v1.Session() as sess:
                 sess.run(init)
                 step = 0
                 while step * batch_size < epo*save_point:
                     batch_xs, batch_ys = next_batches(step)
                     batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                     sess.run([train_op], feed_dict={x: batch_xs,y: batch_ys,})
                     if step % 200 == 0:
                         print('results in training set',step)
                         print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys,}))
                         print('results in test set',step)
                         x_test_batch,y_test_batch=next_test_batches(step)
                         x_test_batch = x_test_batch.reshape([batch_size, n_steps, n_inputs])
                         loss,acc=sess.run([cost,accuracy], feed_dict={x: x_test_batch,y: y_test_batch,})
                         print(loss,acc)
                         saver.save(sess=sess, save_path="./models_{}_{}_{}/train_for_validate.ckpt".format(filename[0:-4],iteration,model), global_step=step+1)
                     if step % 500 == 0:
                          print('-------------monitor-----------------------------------line-------------------')
                     step += 1
    if mode=='validation':
     X=df
     Y=label
     X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=0)#5折交叉验证
     from sklearn.metrics import roc_curve, auc,accuracy_score,classification_report
     import matplotlib.pyplot as plt
     print(len(X_test))#一开始有6388个测试集 
   
    
     test_batches(batch_size,X_test,y_test)
     batches(batch_size, X_train, y_train)
     def roc_pic(y_test,y_score):
      fpr,tpr,threshold = roc_curve(y_test, y_score)
      roc_auc = auc(fpr,tpr)
      print('开始载入图片')
      plt.figure()
      lw = 2
      plt.figure(figsize=(10,10))
      plt.plot(fpr, tpr, color='darkorange'
          ,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
      plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic example')
      plt.legend(loc="lower right")
      plt.show()
     def roc_value(y_test,y_score):
         fpr,tpr,threshold = roc_curve(y_test, y_score)
         roc_auc = auc(fpr,tpr)
         print('ROC_values',roc_auc)
         return roc_auc
     with tf.compat.v1.Session() as sess:
        test_label=[]
        y_score=[]
        y_pred_label=[]
        yreal=[]
        try:
          module_file = tf.train.latest_checkpoint('./models_{}_{}_{}'.format(filename[0:-4],iteration,model))
          saver.restore(sess,module_file)
          print('model load successfully')
        except:
          print('model load fail,please check out the path')
        itera=int(len(X_test)/128)
        for i in range(0,itera):#128一个轮回
    #
         x_test_batch,y_test_batch=next_test_batches(i)
         x_test_batch=x_test_batch.reshape([batch_size, n_steps, n_inputs])
    #print(type(x_test_batch))
    #print(x_test_batch.shape)
    #print(y_test_batch.shape)
    #print(y_test_batch)
         print(sess.run(accuracy, feed_dict={
            x: x_test_batch,
            y: y_test_batch,
           
            }))  
         y2 = sess.run(correct_pred,feed_dict={x:x_test_batch,y: y_test_batch})
         y3 = sess.run(pred,feed_dict={x:x_test_batch,y: y_test_batch})
         #print(len(y2),len(y3))
         for j in range(0,len(y3)):
            test_label.append(y_test_batch[j][0])
            y_score.append(  y3[j][0])
            #print(type(y2[j]))
            if y3[j][0]>=0:
                #yreal.append(y_test[j][0])
                
                y_pred_label.append(0)
            else:
                #print(y_test[j][0])
                #yreal.append(y_test[j][0])
                y_pred_label.append(1)
         #print(y2)
        print(y_test[0])
        print(y_test[0].values)
        
        print(len(test_label),'/',len(X_test))#25125个测试样本
        aucvalue=roc_value(test_label,y_score)
        #print(len(y_test[0]),y_test[0])
        #print(y_pred_label)
        #print(y_pred_label)
        le=len(y_pred_label)
        acc=accuracy_score(y_test[0].values[0:le],y_pred_label)
        print('acc',acc)  
        print(classification_report(y_test[0].values[0:le],y_pred_label))
        list_result=[]
        list_result.append(aucvalue)
        lsr=pd.DataFrame(list_result)
        lsr.to_csv('Train_vali_results_{}_{}_{}.csv'.format(filename[0:-4],iteration,model))
        return aucvalue,acc
        #X_test.to_csv('LSTM_x_test.csv')
        ##np.save('LSTM_y_score',y_score)
        ##np.save('LSTM_y_label',test_label)
        ##print('We produced the y_score and y_label for your test set, named LSTM_y_score and LSTM_y_label')        
    if mode=='predict':   
       global df_gene
       global label_gene
       global df_mirna
       global label_mirna
       global count
       predict_mode=predict_mode#change from file to all 
       def generate_all_samples(i,j):#bestow 2 number to indicate mirna/gene
          global df_gene
          global label_gene
          global df_mirna
          global label_mirna
          #list_label=[]
          mirna_i=df_mirna[i:i+1]
          gene_j=df_gene[j:j+1]
          mirna_i=mirna_i.reset_index(drop=True)
          gene_j=gene_j.reset_index(drop=True)
          temp=pd.concat([mirna_i,gene_j],axis=1)
          key=({'mirna':label_mirna[i],'gene':label_gene[j]})
          return temp,key#return the features&& name of the gene/miRNA
       def concat2afile(mirna_low,mirna_high,gene_low,gene_high):#想办法先出一个128的就好了)
          global count
          count=0
    #print('count',count)
    #list_pairs=[]
          for i in range(mirna_low,mirna_high):
              if gene_high>len(df_gene)-1:#such
                  #print('gene_high>df(gene) last epoch of this mirna',gene_high,len(df_gene))
                  for j in range(gene_low,len(df_gene)): #7552-7534 等于18 384(3*128)-317
                    if count==0:
                      temp,key=generate_all_samples(i,j)
                      feature=temp
                      count+=1
                    else:    
                      temp,key=generate_all_samples(i,j)
                      feature=pd.concat([feature,temp],axis=0)
                      count+=1
                  last_one=gene_high-int(len(df_gene))
            #print(feature)
                  for j in range(0,last_one):
            #      print('three')
                        temp,key=generate_all_samples(0,0)
                        feature=pd.concat([feature,temp],axis=0)
                        count+=1
                        #if j== last_one-1:
                        #    print('for mirna',i,'gene geneated (miRNA0,gene0)score for ',j,'times,in totoal got pairs',(count))
              else:#gene high less than gene
               for j in range(gene_low,gene_high):
                if count==0:
                 temp,key=generate_all_samples(i,j)
                 feature=temp
                 count+=1
                else:
                 temp,key=generate_all_samples(i,j)
                 feature=pd.concat([feature,temp],axis=0)
                 count+=1
          #print(feature)
          feature=feature.reset_index(drop=True)
          feature1=feature.values
          #print('Total length of the feature',len(feature1),'length of one row',len(feature1[0]))
          return feature1#,list_pairs
       #this----------
       print('predict model start','mode selection','{}'.format(predict_mode))
       
       import math
       with tf.compat.v1.Session() as sess:
         if predict_mode=='all':
           global df_gene
           global label_gene
           global df_gene
           global label_mirna
           global count
           df_gene=pd.read_csv('core_297geneC1-L2bs.csv')
           label_gene=df_gene['label']
           df_gene=df_gene.drop(['label'],axis=1)
           df_mirna=pd.read_csv('core_197mirnaC1-L2bs.csv')
           label_mirna=df_mirna['label']
           df_mirna=df_mirna.drop(['label'],axis=1)
           print('successfully load data for mirna/gene')
           test_label=[]
           y_score=[]
           module_file = tf.train.latest_checkpoint()#
           #initial_load()
           saver.restore(sess,module_file)
           print('succssfully load the prediction model')
           list_score=[]
         #  initial_load()
           end=len(df_mirna)#length of mirna file
           #end=10#for test
           for i in range(0,end):
              if i%5 ==0 and i!=0:
                  print('predict mirna',i,',',len(Finalframe.T),'out of',end)
              #print(i)
              y_score=[]
              end2=math.ceil(len(df_gene)/128)
              for j in range(0,end2):#128 a epoch#In the last epoch we will reapeat the mirna 0,gene 0 in the end for 18 times 3个就够了
                 x_test_batch=concat2afile(i,i+1,128*j,128*(j+1))
                 #print(j)
                 x_test_batch=x_test_batch.reshape([batch_size, n_steps, n_inputs])
                 y3 = sess.run(pred,feed_dict={x:x_test_batch})#
                 for m in range(0,len(y3)):
                     y_score.append(y3[m][0])#第一行是标签， 第二行是反标签，所以第一行越高 越说明是正的 
              list_score.append(y_score)
              if i==0:
                  Finalframe=pd.DataFrame(y_score)
              else:
                  
                  Tempframe=pd.DataFrame(y_score)
                  Finalframe=pd.concat([Finalframe,Tempframe],axis=1)
           TF=Finalframe[0:len(df_gene)]
           TF.columns=label_mirna[0:end].values.tolist()
           
           TF=TF.T
           TF.columns=label_gene.values.tolist()
           TF.to_csv('./models_predict{}/sgLSTM_result.csv'.format(filename[0:-4]))
           print(TF)
           print('The CSV format file has been generated')
           np.save('./models_predict{}/sgLSTM_result'.format(filename[0:-4]),list_score)
           print('The np format file has been generated generated')

         #another type of prediction,by using files.
         if predict_mode=='file':
             from sklearn.metrics import roc_curve, auc,accuracy_score,classification_report
             import matplotlib.pyplot as plt
             test_file=list_opt[0]
             model_dir=list_opt[1]
             dfp=pd.read_csv('{}'.format(test_file,index=None))#this should be the discard one'
             
             print("./models/{}".format(model_dir))
             module_file = tf.train.latest_checkpoint("./models/{}".format(model_dir))#
             saver.restore(sess,module_file)
             print('succssfully load the prediction model')
             
             ori=dfp
             df_mirna=dfp['mirna']
             df_gene=dfp['gene']
             df_label=dfp['label']
             print('size of predict dataset',len(dfp))
             dfp=dfp.drop(['mirna'],axis=1)
             dfp=dfp.drop(['gene'],axis=1)
             dfp=dfp.drop(['label'],axis=1)
             print('size of one row',len(dfp[0:1]))
             y_score=[]
             end2=int((len(df_gene)/128))
             for j in range(0,end2):#128 a epoch#In the last epoch we will reapeat the mirna 0,gene 0 in the end for 18 times 3个就够了
                 x_test_batch=dfp[128*j:128*(j+1)].values
                 x_test_batch=x_test_batch.reshape([batch_size, n_steps, n_inputs])
                 y3 = sess.run(pred,feed_dict={x:x_test_batch})#
                 for m in range(0,len(y3)):
                      y_score.append(y3[m][0])
             for i in range(0,128-(len(df_gene)-128*end2)):
                        if i==0:
                             temp=dfp[0:1]
                        else:
                             tep=dfp[0:1]
                             temp=pd.concat([temp,tep],axis=0)
             print(temp,len(df_gene)-128*end2)
             x_test_last=dfp[128*end2:len(df_gene)]
             x_test_last=pd.concat([x_test_last,temp],axis=0)
             x_test_last=x_test_last.values
             print(len(x_test_last))
             x_test_last_batch=x_test_last.reshape([batch_size, n_steps, n_inputs])
             y3 = sess.run(pred,feed_dict={x:x_test_last_batch})#
             for m in range(0,len(y3)):
                 y_score.append(y3[m][0])
             #print(len(df_gene),len(y_score))
             y_score=y_score[0:len(df_gene)]
             def roc_pic(y_test,y_score):
                   fpr,tpr,threshold = roc_curve(y_test, y_score)
                   roc_auc = auc(fpr,tpr)
                   print('AUC值',roc_auc)
                   print('开始载入图片')
                   plt.figure()
                   lw = 1
                   plt.figure(figsize=(10,10))
                   plt.plot(fpr, tpr, color='darkorange'
                       ,
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
                   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                   plt.xlim([0.0, 1.0])
                   plt.ylim([0.0, 1.05])
                   plt.xlabel('False Positive Rate')
                   plt.ylabel('True Positive Rate')
                   plt.title('Receiver operating characteristic example')
                   plt.legend(loc="lower right")
                   plt.show()
             #roc_pic(df_label,y_score)
             fpr,tpr,threshold = roc_curve(df_label, y_score)
             roc_auc = auc(fpr,tpr)
             df_score=pd.DataFrame(y_score)
             ori=pd.concat([ori,df_score],axis=1)
             print(ori)
             ori.to_csv('./Dresults/{}_test_results_{}_{}_auc{}.csv'.format(filename[0:-4],model,iteration,roc_auc),index=None)
             return roc_auc
         if predict_mode=='five':
             from sklearn.metrics import roc_curve, auc,accuracy_score,classification_report
             import matplotlib.pyplot as plt
             dfp=pd.read_csv('{}_test.csv'.format(filename[0:-4],index=None))
             ori=dfp
             df_mirna=dfp['mirna']
             df_gene=dfp['gene']
             df_label=dfp['label']
             print('size of predict dataset',len(dfp))
             dfp=dfp.drop(['mirna'],axis=1)
             dfp=dfp.drop(['gene'],axis=1)
             dfp=dfp.drop(['label'],axis=1)
             print('size of one row',len(dfp[0:1]))
            
             print("./models/models_predict{}_{}".format(filename[0:-4],epo))
             module_file = tf.train.latest_checkpoint("./models/models_predict{}_{}_{}".format(filename[0:-4],epo,model))#
             #epo+=50
             saver.restore(sess,module_file)
             print('succssfully load the prediction model')
             y_score=[]
             end2=int((len(df_gene)/128))
             for j in range(0,end2):#128 a epoch#In the last epoch we will reapeat the mirna 0,gene 0 in the end for 18 times 3个就够了
                 x_test_batch=dfp[128*j:128*(j+1)].values
                 x_test_batch=x_test_batch.reshape([batch_size, n_steps, n_inputs])
                 y3 = sess.run(pred,feed_dict={x:x_test_batch})#
                 for m in range(0,len(y3)):
                      y_score.append(y3[m][0])
             for i in range(0,128-(len(df_gene)-128*end2)):
                        if i==0:
                             temp=dfp[0:1]
                        else:
                             tep=dfp[0:1]
                             temp=pd.concat([temp,tep],axis=0)
             print(temp,len(df_gene)-128*end2)
             x_test_last=dfp[128*end2:len(df_gene)]
             x_test_last=pd.concat([x_test_last,temp],axis=0)
             x_test_last=x_test_last.values
             print(len(x_test_last))
             x_test_last_batch=x_test_last.reshape([batch_size, n_steps, n_inputs])
             y3 = sess.run(pred,feed_dict={x:x_test_last_batch})#
             for m in range(0,len(y3)):
                 y_score.append(y3[m][0])
             #print(len(df_gene),len(y_score))
             y_score=y_score[0:len(df_gene)]
             def roc_pic(y_test,y_score):
                   fpr,tpr,threshold = roc_curve(y_test, y_score)
                   roc_auc = auc(fpr,tpr)
                   print('AUC值',roc_auc)
                   print('开始载入图片')
                   plt.figure()
                   lw = 1
                   plt.figure(figsize=(10,10))
                   plt.plot(fpr, tpr, color='darkorange'
                       ,
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
                   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                   plt.xlim([0.0, 1.0])
                   plt.ylim([0.0, 1.05])
                   plt.xlabel('False Positive Rate')
                   plt.ylabel('True Positive Rate')
                   plt.title('Receiver operating characteristic example')
                   plt.legend(loc="lower right")
                   plt.show()
             #roc_pic(df_label,y_score)
             fpr,tpr,threshold = roc_curve(df_label, y_score)
             roc_auc = auc(fpr,tpr)
             df_score=pd.DataFrame(y_score)
             ori=pd.concat([ori,df_score],axis=1)
             print(ori)
             ori.to_csv('./outcome/{}_test_results_{}_{}_auc{}.csv'.format(filename[0:-4],model,iteration,roc_auc),index=None)
             return roc_auc
   #np.save('LSTM_y_label',test_label)


def check_path():
    import os
#print(type(os.listdir(filePath)))
#print(os.listdir(filePath))
    print('label',os.path.exists('.\models'))
    if os.path.exists('.\models')== False:
      print('model folder not exist,create one')
      os.makedirs('.\models')
    if os.path.exists('.\outcome')==False:
      print('folder not exist, create one')
      os.makedirs('.\outcome')
def train_vali(i):
 tf.compat.v1.reset_default_graph()   
 train_it(filename,0,0,i,'train_some_dataset_for validation','lstm')#train_some_dataset_for validation
 tf.compat.v1.reset_default_graph()   
 train_it(filename,0,0,i,'validation','lstm')#validation       
 tf.compat.v1.reset_default_graph()
def train_pre(file,i,MMode):
 list_opt=[]#for five fold cross, this is empty
 tf.compat.v1.reset_default_graph() 
 train_it(file,0,0,i,'train_all_dataset for the predict',MMode,list_opt,'five')#train_some_dataset_for validation
 tf.compat.v1.reset_default_graph() 
 auc=train_it(file,0,0,i,'predict',MMode,list_opt,'five')
 tf.compat.v1.reset_default_graph()
def train_pre2(file,i,MMode,list_opt):
 #tf.compat.v1.reset_default_graph() 
 #train_it(file,0,0,i,'train_all_dataset for the predict',MMode)#train_some_dataset_for validation
 tf.compat.v1.reset_default_graph() 
 auc=train_it(file,0,0,i,'predict',MMode,list_opt,'file')
 tf.compat.v1.reset_default_graph() 
 return auc
def train_pre_fivefold(filename,mode):
   check_path()
   MMode=mode#for different model using lstm or bilstm
   for j in range(0,1):#control file 
    list_result=[]
    for i in range(1,6):#control epoch
      five_name='{}_epoch_{}_{}.csv'.format(filename[0:-4],j,i) 
      print(five_name)
      epoch=500#defalt is 500, here set to 5 for quick test
      auc=train_pre(five_name,epoch,MMode)
      list_result.append({'iter':j,'epoch':epoch,'auc':auc})
      lis=pd.DataFrame(list_result)
      lis.to_csv('summary{}_{}_{}_{}.csv'.format(MMode,filename[0:-4],j,epoch),index=None)

def trian_pre_discard(filename,mode):
    MMode=mode#  'lstm' or 'bilstm'
    list_result=[]
    epoch=500#defalt is 500, here set to 5 for quick test
    print('start discard test')
    for j in range(0,1):#control file 
        test_file='{}_discard_{}.csv'.format(filename[0:-4],j)
        
        for i in range(1,6):##control epoch, decide which model
           list_opt=[]
           five_name='{}_epoch_{}_{}.csv'.format(filename[0:-4],j,i) 
           model_path='models_predict{}_epoch_{}_{}_{}_{}'.format(filename[0:-4],j,i,epoch,mode)
           list_opt.append(test_file)
           list_opt.append(model_path)
           print('begin to train')
           auc=train_pre2(five_name,i,MMode,list_opt)
           list_result.append({'trial:j':j,'fivefold_seq:i':i,'auc':auc})
    lrp=pd.DataFrame(list_result)
    lrp.to_csv('summary_discard_{}_{}_{}_{}.csv'.format(MMode,filename[0:-4],j,epoch),index=None)


list_file=['GCNGCN_lstmcore.csv']

#for the five-fold cross-validation（TypeA,B all suitable)
#----------------------------------------------#
#for kk in range(0,1):
#  filename=list_file[kk]
 # train_pre_fivefold(filename,'bilstm')
#  train_pre_fivefold(filename,'lstm')
#----------------------------------------------#

#for new cross-validation's prediction:
#（TypeB Only:because only type B have discarded dataset) 
import os
if os.path.exists('.\Dresults')==False:
      print('discard results folder not exist, create one')
      os.makedirs('.\Dresults')
for kk in range(0,1):
    filename=list_file[kk]
    trian_pre_discard(filename,'lstm')
#    trian_pre_discard(filename,'bilstm')
