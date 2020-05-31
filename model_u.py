import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import  metrics

plt.rc('font',family='Times New Roman')
epoch = 1000
batch_size = 128
initial_learning_rate = 0.0001
decay_steps = 1000
decay_rate = 0.98
def TemporalBlock(input, n_outputs, kernel_size, stride, dilation, dropout=0.2,name='net'):
    conv1 = tf.layers.conv1d(input, filters=n_outputs, kernel_size=kernel_size, strides=stride, padding='SAME', dilation_rate=dilation,
                             activation=tf.nn.relu, name=name+'_conv1')
    drop1 = tf.layers.dropout(conv1, rate=dropout, name=name+'_drop1')

    conv2 = tf.layers.conv1d(drop1, filters=n_outputs, kernel_size=kernel_size, strides=stride, padding='SAME', dilation_rate=dilation,
                             activation=tf.nn.relu, name=name+'_conv2')
    drop2 = tf.layers.dropout(conv2, rate=dropout, name=name+'_drop2')
    print("this is input shape", input.shape[-1])
    if (input.shape[-1] == n_outputs):
        return tf.nn.relu(input + drop2)
    else:
        res = tf.layers.conv1d(input, filters=n_outputs, kernel_size=3, strides=1, padding='SAME', dilation_rate=1,
                               activation=tf.nn.relu, name=name+'_res')
        return tf.nn.relu(drop2 + res)
def deal_data(data, label, lenght = 30):

    r_data = []
    r_label = []
    data = np.array(data)
    data_shape = data.shape
    for i in range(data_shape[0]-lenght+1):
        r_data.append(data[i:i+lenght,:])
        r_label.append(label[i:i+lenght])

    r_data = np.array(r_data)
    r_label = np.array(r_label)
    r_label = r_label.squeeze()

    return r_data, r_label




def shuffer(data, label):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)

    return data, label


def read_data(data_path, label_path):
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    data = loadmat(data_path)
    data = pd.DataFrame(data["XTrain"])

    label = loadmat(label_path)
    label = pd.DataFrame(label["YTrain"])

    data_shape = data.shape
    data_len = []
    for i in range(data_shape[0]):
        if i == 0:
            data_len.append((data[0][i].shape)[1])
            real_data = np.array(data[0][i])
            real_label = np.array(label[0][i])
        else:
            data_len.append((data[0][i].shape)[1])
            real_data = np.concatenate([real_data, data[0][i]], axis=1)
            real_label = np.concatenate([real_label, label[0][i]], axis=1)
    print(data_len)
    real_data = np.transpose(real_data, (1, 0))
    real_data = x_scaler.fit_transform(real_data)
    real_data = np.array(real_data)


    real_label = np.transpose(real_label, (1, 0))
    real_label = y_scaler.fit_transform(real_label)
    real_label = np.array(real_label)

    g_real_data = []
    g_real_label = []
    for l in range(len(data_len)):
        if l == 0:
            g_real_data.append(real_data[0:data_len[l],:])
            g_real_label.append(real_label[0:data_len[l],:])
            sc = data_len[l]
        else:
            g_real_data.append(real_data[sc:sc+data_len[l],:])
            g_real_label.append(real_label[sc:sc+data_len[l],:])
            sc = sc + data_len[l]


    train_data = g_real_data[0:80]
    train_label = g_real_label[0:80]
    test_data = g_real_data[80:]
    test_label = g_real_label[80:]

    return train_data, train_label, test_data, test_label, x_scaler, y_scaler,


def read_test_data(data_path, label_path, num):

    data = loadmat(data_path)
    data = pd.DataFrame(data["XTrain"])

    final_data = np.transpose(data[0][num], (1,0))


    label = loadmat(label_path)
    label = pd.DataFrame(label["YTrain"])
    final_label = np.transpose(label[0][num], (1, 0))



    return final_data, final_label

def report_accuracy(logist, label):
    logist = np.reshape(logist, [-1])
    label = np.reshape(label, [-1])
    average = np.sum(label) / len(label)
    ssr = 0
    sst = 0
    for i in range(len(logist)):
        ssr = ssr + pow((label[i] - logist[i]), 2)
        sst = sst + pow((label[i] - average), 2)

    y_mse = mean_squared_error(label, logist)
    y_rmse = y_mse ** 0.5
    R2 = metrics.r2_score(label, logist)
    mae = metrics.mean_absolute_error(label, logist)
    #print("模型泛用性为：", 1 - (ssr / sst))
    print("模型泛用性y_rmse:", y_rmse)
    print("模型泛用性R2：", R2)
    #print("模型泛用性mae：", mae)
    return y_mse, y_rmse,R2, mae





def encoded(X):
    with tf.name_scope("encoded"):
        conv1 = tf.layers.conv1d(inputs=X, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu,
                                 name='encoder_conv1')
        print("this is conv1 shape", conv1.get_shape().as_list())
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same',
                                        name='encoder_pool1')
        print("this is pool1 shape", pool1.get_shape().as_list())
        conv2 = tf.layers.conv1d(inputs=pool1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu,
                                 name='encoder_conv2')
        print("this is conv2 shape", conv2.get_shape().as_list())
        pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same',
                                        name='encoder_pool2')
      #  print("this is pool2 shape", pool2.get_shape().as_list())
        conv3 = tf.layers.conv1d(inputs=pool2, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu,
                                 name='encoder_conv3')
       # print("this is conv3 shape", conv3.get_shape().as_list())
       # pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same',
                #                        name='encoder_pool3')
       # print("this is pool3 shape", pool3.get_shape().as_list())
       # conv4 = tf.layers.conv1d(inputs=pool3, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu,
             #                    name='encoder_conv4')
        #print("this is conv3 shape", conv3.get_shape().as_list())
        encoded = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same', name='encoder')
        print("this is encoded shape", encoded.get_shape().as_list())



    return encoded








X = tf.placeholder(dtype=tf.float32, shape=[None, 30, 17])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 30])

encoded_data = encoded(X)

#tcn_out = TemporalConvNet(encoded_data,num_chanels
with tf.name_scope("TCN_net"):

    dilation_size = 1
    # in_channels = num_inputs if i == 0 else num_channels[i - 1]
    net = TemporalBlock(encoded_data, 32, 3, stride=1, dilation=dilation_size,
                        dropout=0.2,name='net')
    print("this is net shape", net.get_shape().as_list())
    dilation_size = 2
    # in_channels = num_inputs if i == 0 else num_channels[i - 1]

    net1 = TemporalBlock(net, 64, 3, stride=1, dilation=dilation_size,
                        dropout=0.2, name='net1')
    print("this is net1 shape", net1.get_shape().as_list())
    dilation_size = 4
    # in_channels = num_inputs if i == 0 else num_channels[i - 1]

    net2 = TemporalBlock(net1, 128, 3, stride=1, dilation=dilation_size,
                        dropout=0.2, name='net2')
    print("this is net2 shape", net2.get_shape().as_list())
    #dilation_size = 4
    # in_channels = num_inputs if i == 0 else num_channels[i - 1]

   # tcn_out = TemporalBlock(net2, 128, 3, stride=1, dilation=dilation_size,
       #                 dropout=0.2, name='net_out')
   # print("this is tcn_out shape", tcn_out.get_shape().as_list())

#shape = tf.shape(tcn_out)
with tf.name_scope("FC_net"):
    fc_input = tf.layers.flatten(net2)
    fc1 = tf.layers.dense(fc_input,units=64, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),name='fc1')
    fc2 = tf.layers.dense(fc1,units=32, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='fc2')

    logits = tf.layers.dense(fc2,units=30, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='logits')


global_step = tf.Variable(0, trainable=False)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(logits, [-1,30]) - tf.reshape(Y, [-1,30])),axis=1))

tf.summary.scalar("loss", loss)




lrn_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step,
                                           decay_steps,
                                           decay_rate,
                                           staircase=True)

tf.summary.scalar("learn_rate", lrn_rate)

optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate).minimize(loss,global_step=global_step)

merged_summary_op = tf.summary.merge_all()
with tf.Session() as session:


    train_data_path = './data/train.mat'
    train_label_path = './data/label.mat'
    logs_path = './log/'
    session.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)


    t_data, t_label,v_data, v_label, t_x_scaler, t_y_scaler, = read_data(train_data_path, train_label_path)


    mode = 'test'
    if(mode=='train'):

        f_y_mse = []
        f_y_rmse = []
        f_R2 = []
        f_mae = []
        for i in range(epoch):
            for n in range(80):
                b_t_data = t_data[n]
                b_t_label = t_label[n]
                b_t_data, b_t_label = deal_data(b_t_data, b_t_label)
                num = (b_t_data.shape)[0]
                for batch in range(num // batch_size):
                    losss,summary, step, _ = session.run([loss,merged_summary_op,global_step, optimizer], feed_dict={X: b_t_data[batch*batch_size:(batch +1)*batch_size, :, :],
                                                                Y: b_t_label[batch*batch_size:(batch +1)*batch_size,:]})

                print("当前轮数： ",i, "当前数据：",n, "this is loss",losss)



            all_rmse = 0
            for t_n in range(20):
                b_v_data = v_data[t_n]
                b_v_label = v_label[t_n]
                b_v_data, b_v_label = deal_data(b_v_data, b_v_label)
                num = (b_v_data.shape)[0]
                _, _, pres = session.run([loss, merged_summary_op, logits], feed_dict={X: b_v_data, Y: b_v_label})

                r_pres = []
                for t_i in range(num):
                    if t_i == 0:
                        for t_j in range(len(pres[t_i])):
                            r_pres.append(pres[t_i][t_j])
                    else:
                        r_pres.append(pres[t_i][-1])
                r_pres = np.reshape(np.array(r_pres),[-1,1])
                r_pres = t_y_scaler.inverse_transform(r_pres)
                r_label = t_y_scaler.inverse_transform(v_label[t_n])


                _,mse,accc,_ = report_accuracy(r_pres, r_label)
                all_rmse+=mse

            mean_rmse = all_rmse/20
            if(mean_rmse<=21):
                saver.save(session, './checkpoint/tcn_model', global_step=step)


    if(mode=='test'):

        ckpt = tf.train.latest_checkpoint('./checkpoint/')
        saver.restore(session, ckpt)
        all_rmse = 0
        #words = [4,5,6,7]
        for t_n in [4]:#4，6，8，18
            b_v_data = v_data[t_n]
            b_v_label = v_label[t_n]
            b_v_data, b_v_label = deal_data(b_v_data, b_v_label)
            num = (b_v_data.shape)[0]
            _, _, pres = session.run([loss, merged_summary_op, logits], feed_dict={X: b_v_data, Y: b_v_label})

            r_pres = []
            for t_i in range(num):
                if t_i == 0:
                    for t_j in range(len(pres[t_i])):
                        r_pres.append(pres[t_i][t_j])
                else:
                    r_pres.append(pres[t_i][-1])
            r_pres = np.reshape(np.array(r_pres), [-1, 1])
            r_pres  = t_y_scaler.inverse_transform(r_pres)
            r_label = t_y_scaler.inverse_transform(v_label[t_n])

            plt.rcParams['figure.dpi'] = 300
            plt.plot(r_label, label='Estimated RUL',linewidth=2)
            plt.plot(r_pres, label='Actual RUL',linewidth=2)
            font_format1 = {'family': 'Times New Roman','size': 18}
            font_format = {'family': 'Times New Roman', 'size': 18}
            plt.legend(['Actual RUL', 'Estimated RUL'], prop=font_format1)
            plt.xlabel('Cycle number', font_format)
            plt.ylabel('RUL', font_format)
            plt.title('Test Engine Unit #85', fontsize=20)
            plt.yticks(fontproperties='Times New Roman', size=18)
            plt.xticks(fontproperties='Times New Roman', size=18)
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)



            _, mse, accc, _ = report_accuracy(r_pres, r_label)
            print("this is mse", mse)
            print("this is acc", accc)
            all_rmse += mse


        mean_rmse = all_rmse / 20
        print("所有图的平均rmse为：",mean_rmse)
        #plt.legend()
        plt.show()









