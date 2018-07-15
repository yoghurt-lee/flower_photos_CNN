import tensorflow as tf
import os
import glob
import cv2
import numpy as np
path = "./flower_photos/"
model_path = "./model/flower/flower1"

# resize 所有图片为 100*100*3 的 ( RGB 三颜色通道 )
h = 100
w = 100
d = 3

def read_img(path):
    cate = [os.path.join(path,_) for _ in os.listdir(path) if os.path.isdir(os.path.join(path,_))]
    print(cate)
    images = []
    labels = []
    for idx,folder in enumerate(cate):
        for im in glob.glob(os.path.join(folder,"*.jpg")): #利用glob 获取某个文件夹下所有图片
            img = cv2.imread(im)
            img = cv2.resize(img,(w,h))
            images.append(img)
            labels.append(idx)

    print("read over!")
    return np.asarray(images,np.float32), np.asarray(labels,np.int32)

def divide_train_test(data,labels,ratio=0.8): #将所有数据划分为测试集与验证集两部分
    idx = np.int(data.shape[0]*ratio)
    x_train = data[:idx]
    y_train = labels[:idx]
    x_test = data[idx:]
    y_test = labels[idx:]
    return x_train,y_train,x_test,y_test

def CNN(input_tensor,regularizer,keep_prob): # 构建神经网络
    with tf.variable_scope('layer1_conv1'): #第一层 ,卷积层 此层输入 100x100x3 输出为 100x100x32
        W_conv1 = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,W_conv1,strides=[1,1,1,1],padding="SAME") #第一层卷积结果
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,b_conv1)) # relu函数

    with tf.name_scope('layer2_pool1'): #第二层 池化层 此层输入为 100x100x32 输出为 50x50x32
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID") #使用 2x2 的核进行池化

    with tf.variable_scope('layer3_conv2'): #第三层 卷积层 此层输入 50x50x32 输出为 50x50x64
        W_conv2 = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable("bias",[64],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,W_conv2,strides=[1,1,1,1],padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,b_conv2))

    with tf.name_scope('layer4_pool2'):#第四层 池化层 此层输入为 50x50x32 输出为 25x25x64
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('layer5_conv3'):# 第五层 卷积层 此层输入为 25x25x64 输出为 25x25x128
        W_conv3 = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable("bias",[128],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2,W_conv3,strides=[1,1,1,1],padding="SAME")
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,b_conv3))

    with tf.name_scope('layer6_pool3'): #第六层 池化层 此层输入为 25x25x128 输出为 12x12x128
        pool3 = tf.nn.max_pool(relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope('layer7_conv4'): #第七层 卷积层 此层输入为 12x12x128 输出为12x12x128
        W_conv4 = tf.get_variable("weight", [3, 3, 128, 128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3,W_conv4,strides=[1,1,1,1],padding="SAME")
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,b_conv4))

    with tf.name_scope('layer8_pool4'): #第八层 池化层 此层输入为 12x12x128 输出为6x6x128
        pool4 = tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
        nodes = 6*6*128 # 一张图像的池化层最终大小
        reshaped = tf.reshape(pool4,[-1,nodes])

    with tf.variable_scope('layer9_fc1'): #第九层 全连接层(输入->隐藏层1) 此层输入为 6*6*128 隐藏层有 1024个结点
        W_fc1 = tf.get_variable("weights",[nodes,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection("losses",regularizer(W_fc1))
        b_fc1 = tf.get_variable("bias",[1024],initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped,W_fc1)+b_fc1)
        fc1 = tf.nn.dropout(fc1,keep_prob=keep_prob)  # 如果是在训练时 使用dropout操作防止过拟合
        with tf.variable_scope('layer10_fc2'): # 第十层 全连接(隐藏层1->隐藏层2) 此层输入为 1024 输出为 512
            W_fc2 = tf.get_variable("weights", [1024, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None: tf.add_to_collection("losses", regularizer(W_fc2))
            b_fc2 = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

            fc2 = tf.nn.relu(tf.matmul(fc1,W_fc2)+b_fc2)
            fc2 = tf.nn.dropout(fc2,keep_prob=keep_prob)


    with tf.variable_scope('layer11_fc3'): #第11层 全连接(隐藏层2->输出层) 此层输入为 512 输出为 5 分别代表五种花的类别
        W_fc3 = tf.get_variable("weights", [512, 5], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection("losses", regularizer(W_fc3))
        b_fc3 = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))

        output = tf.matmul(fc2, W_fc3) + b_fc3

    return output

def minibatch(inputs,labels,batch_size,shuffle=False): #每次在训练集取出一批数据,最后一个参数代表打乱(训练集需要打乱)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], labels[excerpt]

def main():
    data,labels =read_img(path)

    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]

    x_train, y_train, x_test, y_test = divide_train_test(data,labels)

    x = tf.placeholder(tf.float32, shape=[None, w, h, d], name="x") #占位符
    y_ = tf.placeholder(tf.int32, shape=[None, ], name="y_")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    regularizer = tf.contrib.layers.l1_regularizer(0.001) #正则化项 (L1范式)
    logits = CNN(x,regularizer,keep_prob)

    # (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
    b = tf.constant(value=1, dtype=tf.float32)
    logits_eval = tf.multiply(logits, b, name='logits_eval')

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    n_epoch = 10
    batch_size = 64
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):


        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatch(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a,keep_prob:1.0})
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        print("   train loss: %f" % (np.sum(train_loss) / n_batch))
        print("   train acc: %f" % (np.sum(train_acc) / n_batch))

        # validation
        test_loss, test_acc, n_batch = 0, 0, 0
        for x_test_a, y_test_a in minibatch(x_test, y_test, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_test_a, y_: y_test_a,keep_prob:1.0})
            test_loss += err;
            test_acc += ac;
            n_batch += 1
        print("   test loss: %f" % (np.sum(test_loss) / n_batch))
        print("   test acc: %f" % (np.sum(test_acc) / n_batch))
    saver.save(sess, model_path)
    sess.close()

if __name__ == "__main__":
    main()