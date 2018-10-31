import tensorflow as tf
import cv2
import numpy as np
import glob
import os
import time

def weight_variable(shape):
    initial=tf.random_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    # value=[0.1 for i in range(shape)]
    # initial=tf.constant_initializer(value)
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

def read_img(path):
    # TODO NEEDS UPDATE
    imgpath=path+'Frame/'
    gtpath=path+'Ground/'

    cateimg=[imgpath+x for x in os.listdir(imgpath) if os.path.isdir(imgpath+x)]
    categt=[gtpath+x for x in os.listdir(gtpath) if os.path.isdir(gtpath+x)]

    file1 = []
    file2 = []

    print('Start read the image ...')

    for folder in enumerate(cateimg):
        for im in glob.glob(folder[1] + '/*.png'):
            file1.append(im)
    
    for folder in enumerate(categt):
        for im in glob.glob(folder[1]+'/*.png'):
            file2.append(im)

    print('read OK')

    print('Finished ...')
    return file1,file2

class salNet(object):
    def __init__(self,sess,M=3,input_size=320,batch_size=64):

        self.sess=sess  # session
        self.batch_size=batch_size  
        self.input_size=input_size

        self.weights=dict()
        self.weights['w1']=weight_variable([7,7,3,96])
        self.weights['w2']=weight_variable([5,5,96,256])
        self.weights['w3']=weight_variable([3,3,256,512])
        self.weights['w4']=weight_variable([5,5,512,512])
        self.weights['w5']=weight_variable([5,5,512,512])
        self.weights['w6']=weight_variable([7,7,512,256])
        self.weights['w7']=weight_variable([11,11,256,128])
        self.weights['w8']=weight_variable([11,11,128,32])
        self.weights['w9']=weight_variable([13,13,32,1])


        self.weights['de']=weight_variable([8,8,1,1])

        self.bias=dict()
        self.bias['b1']=bias_variable([96])
        self.bias['b2']=bias_variable([256])
        self.bias['b3']=bias_variable([512])
        self.bias['b4']=bias_variable([512])
        self.bias['b5']=bias_variable([512])
        self.bias['b6']=bias_variable([256])
        self.bias['b7']=bias_variable([128])
        self.bias['b8']=bias_variable([32])
        self.bias['b9']=bias_variable([1])
        

    def net(self):
        
        # input data

        data1=tf.placeholder(tf.float32,shape=[None,self.input_size,self.input_size,3],name='x')
        gt=tf.placeholder(tf.float32,shape=[None,self.input_size,self.input_size,1],name='gt')

        conv1=tf.nn.relu(conv(data1,self.weights['w1'])+self.bias['b1'])
        norm1=tf.nn.lrn(conv1,5,1e-4,0.0001,0.75)
        pool1=maxpool(norm1)
        conv2=tf.nn.relu(conv(pool1,self.weights['w2'])+self.bias['b2'])
        pool2=maxpool(conv2)
        conv3=tf.nn.relu(conv(pool2,self.weights['w3'])+self.bias['b3'])
        conv4=tf.nn.relu(conv(conv3,self.weights['w4'])+self.bias['b4'])
        conv5=tf.nn.relu(conv(conv4,self.weights['w5'])+self.bias['b5'])
        conv6=tf.nn.relu(conv(conv5,self.weights['w6'])+self.bias['b6'])
        conv7=tf.nn.relu(conv(conv6,self.weights['w7'])+self.bias['b7'])
        conv8=tf.nn.relu(conv(conv7,self.weights['w8'])+self.bias['b8'])
        conv9=conv(conv8,self.weights['w9'])+self.bias['b9']

        deconv1=tf.nn.conv2d_transpose(conv9,filter=self.weights['de'],output_shape=[self.batch_size,320,320,1],strides=[1,4,4,1],padding='SAME')
        prob=tf.nn.sigmoid(deconv1)
        
        print('conv9: '+conv9.shape.__str__())

        return prob,data1,gt # return GT to calc loss_fn
    
    def loss_fn(self,prob,gt):
        loss=tf.sqrt(tf.reduce_sum(tf.square(prob-gt)))
        return loss

    def minibatches(self,imageinputs=None,targets=None,batch_size=None,shuffle=False):
        if shuffle:
            indices=np.arange(len(imageinputs))
            np.random.shuffle(indices)

        for start_idx in range(0,len(imageinputs)-batch_size+1,batch_size):
            if shuffle:
                excerpt=indices[start_idx:start_idx+batch_size]
            else:
                excerpt=slice(start_idx,start_idx+batch_size)
            yield np.array(imageinputs)[excerpt],np.array(targets)[excerpt]

    def getimg(self,imageinputs,targets):
        imgs = []
        gts = []
        for item in imageinputs:
            im=cv2.imread(item)
            im=cv2.resize(im,(320,320))
            imgs.append(im)
        for item in targets:
            im=cv2.imread(item,cv2.IMREAD_GRAYSCALE)
            im=cv2.resize(im,(320,320))
            im=np.expand_dims(im,axis=2)    # 0-255
            gts.append(im)
        return np.asarray(imgs, np.float32), np.asarray(gts, np.float32)

    def runable(self,x_train,y_train,x_val,y_val,train_op,loss):
        num_epochs=1
        self.sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=1)

        for epoch in range(num_epochs):
            train_loss,n_batch=0,0
            for x_train_a,y_train_a in self.minibatches(x_train,y_train,self.batch_size,shuffle=True):
                
                start=time.time()
                x_train_img,y_train_img=self.getimg(x_train_a,y_train_a)
                _,err=self.sess.run([train_op,loss],feed_dict={data1:x_train_img, gt:y_train_img})
                duration=time.time()-start
                train_loss+=err
                n_batch+=1
                if n_batch%100==0:
                    print("Epoch %d, batch: %d, train loss: %f, duration: %.5f" % (epoch, n_batch, (train_loss/n_batch), duration))
            print("Epoch %d train loss: %f" % (epoch, (train_loss/n_batch)))
            if epoch%1==0:
                saver.save(sess,'salNetModel/my-model',global_step=epoch+1)
                print('-----Save model-----')

        self.sess.close()

    def inference(self,data1):

        conv1=tf.nn.relu(conv(data1,self.weights['w1'])+self.bias['b1'])
        norm1=tf.nn.lrn(conv1,5,1e-4,0.0001,0.75)
        pool1=maxpool(norm1)
        conv2=tf.nn.relu(conv(pool1,self.weights['w2'])+self.bias['b2'])
        pool2=maxpool(conv2)
        conv3=tf.nn.relu(conv(pool2,self.weights['w3'])+self.bias['b3'])
        conv4=tf.nn.relu(conv(conv3,self.weights['w4'])+self.bias['b4'])
        conv5=tf.nn.relu(conv(conv4,self.weights['w5'])+self.bias['b5'])
        conv6=tf.nn.relu(conv(conv5,self.weights['w6'])+self.bias['b6'])
        conv7=tf.nn.relu(conv(conv6,self.weights['w7'])+self.bias['b7'])
        conv8=tf.nn.relu(conv(conv7,self.weights['w8'])+self.bias['b8'])
        conv9=conv(conv8,self.weights['w9'])+self.bias['b9']

        deconv1=tf.nn.conv2d_transpose(conv9,filter=self.weights['de'],output_shape=[self.batch_size,320,320,1],strides=[1,4,4,1],padding='SAME')
        prob=tf.nn.sigmoid(deconv1)

        return prob

if __name__=='__main__':
    # process img
    # trainpath='/media/cvmedia/Data/fukui/DIS/AVS1K/trainSet/'
    # valpath='/media/cvmedia/Data/fukui/DIS/AVS1K/validSet/'
    trainpath='data/trainSet/'
    valpath='data/valSet/'
    x_train,y_train=read_img(path=trainpath)
    x_val,y_val=read_img(path=valpath)

    # define net
    sess=tf.InteractiveSession()
    dva=salNet(sess,batch_size=4)
    prob, data1, gt=dva.net()

    # lr decay
    current_epoch=tf.Variable(0)
    learning_rate=tf.train.exponential_decay(5*1e-4,current_epoch,decay_steps=10000,decay_rate=0.1)
    optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9) 

    # optim
    loss=dva.loss_fn(prob,gt)
    train_op=optimizer.minimize(loss,global_step=current_epoch)
    
    dva.runable(x_train,y_train,x_val,y_val,train_op,loss)

    print('END')
    

# forward
# gt=tf.random_normal([3,320,320,1])
# x=tf.random_normal([3,320,320,1])

