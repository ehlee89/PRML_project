import os
import time

import numpy as np
import tensorflow as tf

from datetime import datetime
import matplotlib.pyplot as plt

def find_str(list_str):
    if "AAD" in list_str:
        return 0
    elif "PNC" in list_str:
        return 2
    elif "ANC" in list_str:
        return 1
    elif "PPD" in list_str:
        return 3

def init_weight_bias(name, shape, filtercnt, trainable):

    if name[0] == 'c':
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1,
                                                                dtype = tf.float32, name = name + "w",), trainable=trainable)
    else:
        weights = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32, trainable=trainable)
    biases = tf.Variable(initial_value=tf.constant(0, shape=[filtercnt], dtype=tf.float32), name=name + "b",
                         trainable=trainable)
    return weights, biases

def conv3d_layer(data, weight, bias, padding, is_inception):
    conv = tf.nn.conv3d(input=data, filter=weight, strides=[1, 1, 1, 1, 1], padding=padding)

    if is_inception:
        return tf.nn.bias_add(conv, bias)
    return tf.nn.bias_add(conv, bias)

def conv_layer(data, weight, bias, padding, is_inception):
    conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
    if is_inception:
        return tf.nn.bias_add(conv, bias)
    return tf.nn.bias_add(conv, bias)

def relu_layer(conv):
    return tf.nn.relu(conv)

def pool3d_layer(data):
    return tf.nn.max_pool3d(input=data, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="VALID")

def pool_layer(data):
    return tf.nn.max_pool(value=data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

def fc_layer(data, weight, bias, dropout, batch_norm):
    shape = data.get_shape().as_list()
    shape = [shape[0], np.prod(shape[1:])]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
    if batch_norm:
        hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = tf.nn.relu(hidden)
    if dropout < 1.:
        hidden = tf.nn.dropout(hidden, dropout)
    return hidden

def output_layer(data, weight, bias, label):
    shape = data.get_shape().as_list()
    shape = [shape[0], np.prod(shape[1:])]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
    if label is None:
        return None, tf.nn.softmax(hidden, dim=-1)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=hidden)), \
           tf.nn.softmax(hidden, dim=-1)

class model_def:
    def __init__(self):
        self.patch_size = 33
        self.lbl_cnt = 2
        self.filters = 64

        self.batch_size = 100
        self.do_rate = 0.5

    def AP_CNN(self, train=True):
        """
          layer: [3,3,3@64] [3,3,3@64] MP [3,3,3,3@128] [3,3,3@128] MP [3,3,3@128] fc
        """
        if train:
            do_rate = self.do_rate
            train_data_node = tf.placeholder(tf.float32, shape=(
                self.batch_size, self.patch_size, self.patch_size, self.patch_size, 1))
            train_labels_node = tf.placeholder(tf.int64, shape=self.batch_size)

        else:
            batch_size = 1
            do_rate = 1.
            train_data_node = tf.placeholder(tf.float32, shape=(
                batch_size, self.patch_size, self.patch_size, self.patch_size, 1))
            train_labels_node = None

        layers = [train_data_node]

        cross_entropy, softmax = None, None

        w1, b1 = init_weight_bias(name="c%d" % (0), shape=[3, 3, 3, 1, self.filters], filtercnt=self.filters, trainable=train)
        conv3d_1 = conv3d_layer(data=layers[-1], weight=w1, bias=b1, padding="SAME", is_inception=False)
        layers.append(conv3d_1)
        relu3d_1 = relu_layer(conv3d_1)
        layers.append(relu3d_1)

        w2, b2 = init_weight_bias(name="c%d" % (1), shape=[3, 3, 3, self.filters, self.filters], filtercnt=self.filters, trainable=train)
        conv3d_2 = conv3d_layer(data=layers[-1], weight=w2, bias=b2, padding="SAME", is_inception=False)
        layers.append(conv3d_2)
        relu3d_2 = relu_layer(conv3d_2)
        layers.append(relu3d_2)

        pool3d_1 = pool3d_layer(data=layers[-1])
        layers.append(pool3d_1)

        w3, b3 = init_weight_bias(name="c%d" % (2), shape=[3, 3, 3, self.filters, self.filters*2], filtercnt=self.filters*2, trainable=train)
        conv3d_3 = conv3d_layer(data=layers[-1], weight=w3, bias=b3, padding="SAME", is_inception=False)
        layers.append(conv3d_3)
        relu3d_3 = relu_layer(conv3d_3)
        layers.append(relu3d_3)

        w4, b4 = init_weight_bias(name="c%d" % (2), shape=[3, 3, 3, self.filters*2, self.filters*2], filtercnt=self.filters*2, trainable=train)
        conv3d_4 = conv3d_layer(data=layers[-1], weight=w4, bias=b4, padding="SAME", is_inception=False)
        layers.append(conv3d_4)
        relu3d_4 = relu_layer(conv3d_4)
        layers.append(relu3d_4)

        pool3d_2 = pool3d_layer(data=layers[-1])
        layers.append(pool3d_2)

        w5, b5 = init_weight_bias(name="c%d" % (2), shape=[3, 3, 3, self.filters * 2, self.filters * 2],
                                  filtercnt=self.filters * 2, trainable=train)
        conv3d_5 = conv3d_layer(data=layers[-1], weight=w5, bias=b5, padding="SAME", is_inception=False)
        layers.append(conv3d_5)
        relu3d_5 = relu_layer(conv3d_5)
        layers.append(relu3d_5)

        w6, b6 = init_weight_bias(name="c%d" % (2), shape=[3, 3, 3, self.filters * 2, self.filters * 2],
                                  filtercnt=self.filters * 2, trainable=train)
        conv3d_6 = conv3d_layer(data=layers[-1], weight=w6, bias=b6, padding="SAME", is_inception=False)
        layers.append(conv3d_6)
        relu3d_6 = relu_layer(conv3d_6)
        layers.append(relu3d_6)

        fw1, fb1 = init_weight_bias(name="f%d" % (0), shape=[7 * 7 * 7 * 128, self.filters*4],
                                    filtercnt=self.filters*4, trainable=train)
        fc_1 = fc_layer(data=layers[-1], weight=fw1, bias=fb1, dropout=do_rate, batch_norm=False)
        layers.append(fc_1)
        fw2, fb2 = init_weight_bias(name="f%d" % (1), shape=[self.filters*4, self.lbl_cnt],
                                    filtercnt=self.lbl_cnt, trainable=train)
        cross_entropy, softmax = output_layer(data=layers[-1], weight=fw2, bias=fb2, label=train_labels_node)

        return cross_entropy, softmax, layers, train_data_node, train_labels_node

class model_execute:
    def __init__(self, data_path, patch_path, label_path, exp_path, output_path, patch_name, label_name, data_shape):
        self.epochs = 100
        self.eval_freq = 100
        self.init_lr = 0.0001
        self.final_lr = 0.000001

        self.patch_size = [33]
        self.lbl_cnt = 2
        self.filters = [64]
        self.batch_size = 100

        self.data_name = patch_name
        self.label_name = label_name
        self.data_shape = data_shape
        self.exp_path = exp_path
        self.output_path = output_path
        self.data_path = data_path
        self.label_path = label_path
        self.patch_path = patch_path

    def mk_3D_voxel_data(self):
        """
            AD: 0 / NC: 1 / PNC: 2 / PPD: 3
        """
        logthis("Make training data started!")

        ADFlag = True
        NCFlag = True
        PNCFlag = True
        PPDFlag = True

        test_dSize, train_dSize = count_dShape(self.data_name,self.data_shape)

        trainData = np.memmap(filename=self.exp_path + "/trainData.dat", dtype=np.float32,
                              mode='w+', shape=(train_dSize, 33, 33, 33))
        testData = np.memmap(filename=self.exp_path + "/testData.dat", dtype=np.float32,
                             mode='w+', shape=(test_dSize, 33, 33, 33))

        trainlbl = np.memmap(filename=self.exp_path + "/trainLabel.lbl", dtype=np.uint8, mode='w+',
                             shape=(1, 1, train_dSize))
        testlbl = np.memmap(filename=self.exp_path + "/testLabel.lbl", dtype=np.uint8,
                            mode='w+', shape=(1, 1, test_dSize))
        trStart = 0
        teStart = 0
        for i, dName in enumerate(self.data_name):
            print('%d  ' % i)

            diseaseNum = find_str(dName)
            dShape = (self.data_shape[i][0],  self.data_shape[i][1], self.data_shape[i][2],  self.data_shape[i][3])
            temp = np.memmap(filename=dName, dtype=np.float32, mode='r', shape=dShape)
            lbl = np.memmap(filename=self.label_name[i], dtype=np.uint8, mode='r')
            # print(dName, self.label_name[i])
            if diseaseNum == 0:
                if ADFlag:
                    if teStart == 0:
                        teEnd = self.data_shape[i][0]
                    else:
                        teStart = teEnd
                        teEnd += self.data_shape[i][0]
                    # print(tStart, tEnd, test_dSize)
                    testData[teStart:teEnd, :, :, :] = temp.copy()
                    testlbl[:, :, teStart:teEnd] = lbl.copy()
                    print('Make Test AD %d'% testlbl[0, 0, teEnd])
                    ADFlag = False
                else:
                    if trStart == 0:
                        trEnd = self.data_shape[i][0]
                    else:
                        trStart = trEnd
                        trEnd += self.data_shape[i][0]
                    # print((tStart - test_dSize), (tEnd - test_dSize), train_dSize)
                    trainData[(trStart - test_dSize):(trEnd - test_dSize), :, :, :] = temp.copy()
                    trainlbl[:, :, (trStart - test_dSize):(trEnd - test_dSize)] = lbl.copy()

            elif diseaseNum == 1:
                if NCFlag:
                    if teStart == 0:
                        teEnd = self.data_shape[i][0]
                    else:
                        teStart = teEnd
                        teEnd += self.data_shape[i][0]
                    # print(tStart, tEnd, test_dSize)
                    testData[teStart:teEnd, :, :, :] = temp.copy()
                    testlbl[:, :, teStart:teEnd] = lbl.copy()
                    print('Make Test NC %d'% testlbl[0,0,teEnd])
                    NCFlag = False
                else:
                    if trStart == 0:
                        trEnd = self.data_shape[i][0]
                    else:
                        trStart = trEnd
                        trEnd += self.data_shape[i][0]
                    # print((tStart - test_dSize), (tEnd - test_dSize), train_dSize)
                    trainData[(trStart - test_dSize):(trEnd - test_dSize), :, :, :] = temp.copy()
                    trainlbl[:, :, (trStart - test_dSize):(trEnd - test_dSize)] = lbl.copy()
            elif diseaseNum == 2:
                if PNCFlag:
                    if teStart == 0:
                         teEnd = self.data_shape[i][0]
                    else:
                        teStart = teEnd
                        teEnd += self.data_shape[i][0]
                    # print(tStart, tEnd, test_dSize)
                    testData[teStart:teEnd, :, :, :] = temp.copy()
                    testlbl[:, :, teStart:teEnd] = lbl.copy()
                    print('Make Test PNC %d' % testlbl[0, 0, teEnd])
                    PNCFlag = False
                else:
                    if trStart == 0:
                        trEnd = self.data_shape[i][0]
                    else:
                        trStart = trEnd
                        trEnd += self.data_shape[i][0]
                    # print((tStart - test_dSize), (tEnd - test_dSize), train_dSize)
                    trainData[(trStart - test_dSize):(trEnd - test_dSize), :, :, :] = temp.copy()
                    trainlbl[:, :, (trStart - test_dSize):(trEnd - test_dSize)] = lbl.copy()

            elif diseaseNum == 3:
                if PPDFlag:
                    if trStart == 0:
                        teEnd = self.data_shape[i][0]
                    else:
                        trStart = teEnd
                        teEnd += self.data_shape[i][0]
                    # print(tStart, tEnd, test_dSize)
                    testData[teStart:teEnd, :, :, :] = temp.copy()
                    testlbl[:, :, teStart:teEnd] = lbl.copy()
                    print('Make Test PPD %d' % testlbl[0, 0, teEnd])
                    PPDFlag = False
                else:
                    if trStart == 0:
                        trEnd = self.data_shape[i][0]
                    else:
                        trStart = trEnd
                        trEnd += self.data_shape[i][0]
                    # print((tStart - test_dSize), (tEnd - test_dSize), train_dSize)
                    trainData[(trStart - test_dSize):(trEnd - test_dSize), :, :, :] = temp.copy()
                    trainlbl[:, :, (trStart - test_dSize):(trEnd - test_dSize)] = lbl.copy()
        print('Make Test train')

    def train_AP_CNN(self, cross_entropy, softmax, data_node, label_node):
        logthis("Original CNN training started!")

        lbl = np.memmap(filename=self.exp_path + "/trainLabel.lbl", dtype=np.uint8, mode="r")
        train_size = lbl.shape[0]
        dataShape = (train_size, 33, 33, 33, 1)
        data = np.memmap(filename=self.exp_path + "/trainData.dat", dtype=np.float32,
                              mode="r", shape=dataShape)

        rand_idx = np.random.permutation(train_size)

        batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
        learning_rate = tf.train.exponential_decay(learning_rate=self.init_lr, global_step=batch * self.batch_size,
                                                   decay_steps=train_size, staircase=True,
                                                   decay_rate=np.power(self.final_lr / self.init_lr,
                                                                       np.float(1) / self.epochs))

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=batch)
        predict = tf.to_double(100) * (
            tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(softmax, label_node, 1))))

        with tf.Session() as sess:
            summary_path = self.output_path + "/summary_AP_CNN/%d" % (int(time.time()))
            model_path = self.output_path + "/model_AP_CNN"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            tf.global_variables_initializer().run()
            print("Variable Initialized")
            tf.summary.scalar("error", predict)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=10, max_to_keep=200)
            start_time = time.time()

            # batch size
            cur_epoch = 0

            for step in range(int(self.epochs * train_size) // self.batch_size):
                offset = (step * self.batch_size) % (train_size - self.batch_size)
                batch_data = data[rand_idx[offset:offset + self.batch_size]]
                batch_labels = lbl[rand_idx[offset:offset + self.batch_size]]
                feed_dict = {data_node: batch_data, label_node: batch_labels}

                _, l, lr, predictions, summary_out = sess.run(
                    [optimizer, cross_entropy, learning_rate, predict, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_out, global_step=step * self.batch_size)
                if step % self.eval_freq == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print(
                        'Step %d (epoch %.2f), %d s' % (step, float(step) * self.batch_size / train_size, elapsed_time))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % predictions)

                if np.floor(cur_epoch) != np.floor((step * self.batch_size) / train_size):
                    print(cur_epoch)
                    print((step * self.batch_size) / train_size)
                    # print(cur_epoch==(step * self.batch_size) / train_size)
                    print("Saved in path", saver.save(sess, model_path + "/%d.ckpt" % (cur_epoch)))
                    rand_idx = np.random.permutation(train_size)

                cur_epoch = (step * self.batch_size) / train_size

            print("Saved in path", saver.save(sess, model_path + "/savedmodel_final.ckpt"))
        tf.reset_default_graph()

    def test_AP_CNN(self, softmax, data_node):
        """
            AP_CNN_1 : 3D CNN for PRML
        """
        logthis("Original CNN testing started!")

        data_path = self.exp_path + "/test%s"
        model_path = self.output_path + "/model_AP_CNN/%d.ckpt" % 10
        result_path = self.output_path + "/AP_cnn_1.dat"
        lbl = np.memmap(filename=data_path % "Label.lbl", dtype=np.uint8, mode="r")
        data = np.memmap(filename=data_path % "Data.dat", dtype=np.float32, mode="r", shape=(lbl.shape[0], 33, 33, 33, 1))

        all_result = np.memmap(filename=result_path, dtype=np.float32, mode="w+", shape=(data.shape[0], 2))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            for i, dat in enumerate(data):

                all_result[i] = sess.run(softmax, feed_dict={data_node: dat.reshape([1,33,33,33,1])})
                if i % 1000 == 0:
                    print(i, data.shape[0])

    def code_test(self):

        pm_path = self.output_path + "/AP_cnn_1.dat"
        data_path = self.exp_path + "/testLabel.%s"
        lbl = np.memmap(filename=data_path % "lbl", dtype=np.uint8, mode="r")
        all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="r", shape=(lbl.shape[0], 2))

        all = np.argmax(all_result, axis=-1).astype(np.uint8)
        print(np.bincount(lbl))
        from sklearn.metrics import confusion_matrix
        aa = confusion_matrix(lbl, all)
        TP = aa[1, 1]
        TN =  aa[0, 0]

        FP =  aa[0, 1]
        FN =  aa[1, 0]
        ACC = ((TP + TN) / (TP+FP+TN+FN)) * 100

        """ 

           Thus in binary classification, the count of true negatives is
        :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
        :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
        """

        print("TP ", TP)
        print("TN ", TN)
        print("FP ", FP)
        print("FN ", FN)
        print('ACC ', ACC)
        print("Sen ", TP / (TP + FN) * 100)
        print("Spe ", TN / (FP + TN) * 100)
        print("PPV ", TP / (TP + FP) * 100)
        print("NPV ", TN / (FN + TN) * 100)

        print("\n\nTP", aa[1, 1], aa[1, 1] * 100. / np.sum(aa))
        print("TN", aa[1, 0], aa[1, 0] * 100. / np.sum(aa))
        print("FP", aa[0, 1], aa[0, 1] * 100. / np.sum(aa))
        print("FN", aa[0, 0], aa[0, 0] * 100. / np.sum(aa))
        # print("ACCURACY", aa[1,1],aa[1,1] / )

def logthis(a):
    print("\n" + str(datetime.now()) + ": " + str(a))

def one_hot(lists):
    mk_lbl = np.zeros(shape=(lists.shape[0], 2),dtype=int)

    for l, list in enumerate(lists):
        if l == 0:
            mk_lbl[l, :] = [0, 1]
        else:
            mk_lbl[l, :] = [1, 0]

    return mk_lbl

def count_dShape(data_name,lists):

    test_dSize = 0
    train_dSize = 0
    t_num = 0
    ADFlag = True
    NCFlag = True
    PNCFlag = True
    PPDFlag = True

    for dname, list in zip(data_name,lists):
        diseaseNum = find_str(dname)

        if diseaseNum == 0:
            if ADFlag:
                test_dSize += list[0]
                ADFlag = False
            else:
                train_dSize += list[0]

        elif diseaseNum == 1:
            if NCFlag:
                test_dSize += list[0]
                NCFlag = False
            else:
                train_dSize += list[0]
        elif diseaseNum == 2:
            if PNCFlag:
                test_dSize += list[0]
                PNCFlag = False
            else:
                train_dSize += list[0]

        elif diseaseNum == 3:
            if PPDFlag:
                test_dSize += list[0]
                PPDFlag = False
            else:
                train_dSize += list[0]

    return test_dSize, train_dSize