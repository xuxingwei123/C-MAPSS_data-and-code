import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import  metrics
epoch = 1000
batch_size = 16
#可变学习率的初始学习率
initial_learning_rate = 0.001
#可变学习率的损失步长
decay_steps = 100
#可变学习率的衰减率
decay_rate = 0.98


#特征提取函数，就是一些东西的卷积

def encoded(X):
    with tf.name_scope("encoded"):
        input_x, input_y, input_z = tf.split(X, [3, 3, 1], axis=2)
        #第一个并行卷积
        convx = tf.layers.conv1d(inputs=input_x, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu,
                                  name='encoder_convx')
        poolx = tf.layers.max_pooling1d(inputs=convx, pool_size=2, strides=2, padding='same',
                                         name='encoder_poolx')
        convy = tf.layers.conv1d(inputs=input_y, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu,
                                 name='encoder_convy')
        pooly = tf.layers.max_pooling1d(inputs=convy, pool_size=2, strides=2, padding='same',
                                        name='encoder_pooly')
        convz = tf.layers.conv1d(inputs=input_z, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu,
                                 name='encoder_convz')
        poolz = tf.layers.max_pooling1d(inputs=convz, pool_size=2, strides=2, padding='same',
                                        name='encoder_poolz')

        cnn_out = tf.concat((poolx, pooly, poolz), axis=2)

        def conv_op(input,f, name):
            # W = tf.get_variable("%s_W" % name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
            # b = tf.get_variable("%s_b" % name, shape[-1], tf.float32, tf.constant_initializer(1.0))
            # return tf.add(tf.layers.conv1d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
            return tf.layers.conv1d(inputs=input, filters=f, kernel_size=3,strides =2,bias_initializer=tf.zeros_initializer(), padding='same')
        #Gated CNN
        filters = [48, 64, 128, 128]  # 卷积核尺寸 64个，128*128
        strides = [2, 2]
        res_input =cnn_out
        h = cnn_out

        for i in range(2):

                with tf.variable_scope("layer_%d" % i):
                    # 计算两个卷积w，v
                    conv_w = conv_op(h,filters[i], "linear")
                    conv_v = conv_op(h,filters[i], "gated")
                    # 计算门限输出h
                    h = conv_w * tf.sigmoid(conv_v)
                                        # 将每5层Gated CNN组合成一个block。
                    # if i % 4 == 0:
                    #      h += res_input
                    #      res_input = h
        encoded = h

        print("this is encoded shape", encoded.get_shape().as_list())



        # conv1 = tf.layers.conv1d(inputs=X, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu,
        #                          name='encoder_conv1')
        # print("this is conv1 shape", conv1.get_shape().as_list())
        # pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same',
        #                                 name='encoder_pool1')
        # print("this is pool1 shape", pool1.get_shape().as_list())
        # conv2 = tf.layers.conv1d(inputs=pool1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu,
        #                          name='encoder_conv2')
        # print("this is conv2 shape", conv2.get_shape().as_list())
        # pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same',
        #                                 name='encoder_pool2')
        # print("this is pool2 shape", pool2.get_shape().as_list())
        # conv3 = tf.layers.conv1d(inputs=pool2, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu,
        #                          name='encoder_conv3')
        # print("this is conv3 shape", conv3.get_shape().as_list())
        # pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same',
        #                                 name='encoder_pool3')
        # print("this is pool3 shape", pool3.get_shape().as_list())
        # conv4 = tf.layers.conv1d(inputs=pool3, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu,
        #                          name='encoder_conv4')
        # print("this is conv3 shape", conv3.get_shape().as_list())
        # encoded = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same', name='encoder')
        # print("this is encoded shape", encoded.get_shape().as_list())



    return encoded




#TCN残差块
def TemporalBlock(input, n_outputs, kernel_size, stride, dilation, dropout=0.2,name='net'):
    #空洞卷积
    conv1 = tf.layers.conv1d(input, filters=n_outputs, kernel_size=kernel_size, strides=stride, padding='SAME', dilation_rate=dilation,
                             activation=tf.nn.relu, name=name+'_conv1')
    drop1 = tf.layers.dropout(conv1, rate=dropout, name=name+'_drop1')

    conv2 = tf.layers.conv1d(drop1, filters=n_outputs, kernel_size=kernel_size, strides=stride, padding='SAME', dilation_rate=dilation,
                             activation=tf.nn.relu, name=name+'_conv2')
    drop2 = tf.layers.dropout(conv2, rate=dropout, name=name+'_drop2')
    print("this is input shape", input.shape[-1])
    # 残差部分
    if (input.shape[-1] == n_outputs):
        return tf.nn.relu(input + drop2)
    else:
        res = tf.layers.conv1d(input, filters=n_outputs, kernel_size=3, strides=1, padding='SAME', dilation_rate=1,
                               activation=tf.nn.relu, name=name+'_res')
        return tf.nn.relu(drop2 + res)

#处理数据函数
def deal_data(data, label):
    # 对数据进行打乱
    # 题的，但是要注意的是，测试的过成不要打乱

    data_shape = data.shape

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    label = scaler.fit_transform(label)

    new_data = []
    new_label = []
    # 计算有多少条数据
    num = data_shape[0] // 20000
    # 将数数据拆分成(num,20000,7)的形式
    for i in range(num):
        new_data.append(data[i * 20000:(i + 1) * 20000, :])
        new_label.append(label[i, :])
    new_data = np.array(new_data)
    new_label = np.array(new_label)




    return new_data, new_label, num, scaler


# 打乱数据
def shuffer(data, label):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)

    return data, label


# 读取数据，返回的是文件里的所有的数据之和
def read_data(path_data):
    file_name = os.listdir(path_data)
    data_file_name = []
    label_file_name = []
    for i in file_name:
        if (i[-8:-4] == 'data'):
            data_file_name.append(os.path.join(path_data, i))
        elif (i[-8:-4] == 'wear'):
            label_file_name.append(os.path.join(path_data, i))
        else:
            print("请修改文件的名字以符合要求")
    final_data = []
    final_label = []
    for j in data_file_name:
        data = pd.read_csv(j, header=None).values
        label = pd.read_csv(j[:-8] + 'wear.csv', header=None).values
        final_data.append(data)
        final_label.append(label)
    final_data = np.array(final_data)
    final_label = np.array(final_label)
    print(final_data.shape)
    print(final_label.shape)
    final_data = np.reshape(final_data, [-1, 7])
    final_label = np.reshape(final_label, [-1, 1])

    return final_data, final_label

#计算准确率
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
    #print("模型泛用性y_rmse:", y_rmse)
    print("模型泛用性R2：", R2)
    print("模型泛用性mae：", mae)
    return y_mse, y_rmse,R2, mae






#X,Y 两个占位符
X = tf.placeholder(dtype=tf.float32, shape=[None, 20000, 7])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

encoded_data = encoded(X)
#输出的shape是【？，1250，128}
#tcn_out = TemporalConvNet(encoded_data,num_chanels
with tf.name_scope("TCN_net"):
    #用了4个tcn残差块，注意每层的空洞卷积率都不一样
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

    net2 = TemporalBlock(net1, 64, 3, stride=1, dilation=dilation_size,
                        dropout=0.2, name='net2')
    print("this is net2 shape", net2.get_shape().as_list())
    dilation_size = 8
    # in_channels = num_inputs if i == 0 else num_channels[i - 1]

    tcn_out = TemporalBlock(net2, 128, 3, stride=1, dilation=dilation_size,
                        dropout=0.2, name='net_out')
    print("this is tcn_out shape", tcn_out.get_shape().as_list())

shape = tf.shape(tcn_out)
#全连接部分
with tf.name_scope("FC_net"):
    fc_input = tf.reshape(tcn_out, shape=[-1, 128*2500])
    fc1 = tf.layers.dense(fc_input,units=16, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),name='fc1')
    fc2 = tf.layers.dense(fc1,units=4, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='fc2')

    logits = tf.layers.dense(fc_input,units=1, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), name='logits')

#全局步长
global_step = tf.Variable(0, trainable=False)

#损失值
loss = tf.reduce_mean(tf.square(tf.reshape(logits, [-1]) - tf.reshape(Y, [-1])))
#将损失写入tensorboard中
tf.summary.scalar("loss", loss)



# 可变学习率
lrn_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step,
                                           decay_steps,
                                           decay_rate,
                                           staircase=True)
#将学习率写入tensorboard中
tf.summary.scalar("learn_rate", lrn_rate)
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate).minimize(loss,global_step=global_step)
#理解为tensorboard的初始化
merged_summary_op = tf.summary.merge_all()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    path_data = './data/'
    logs_path = './log/'
    test_data = './test_data/'
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)



    mode = 'train'   #这个控制模型处于哪个状态
    if(mode=='train'):
        #读取数据，path_data是文件夹的路径
        data, label = read_data(path_data)
        #对读入进来的数据进行一个整合处理
        t_data, t_label, num, scaler = deal_data(data, label)
        t_data = np.reshape(t_data, [-1, 20000, 7])
        t_label = np.reshape(t_label, [-1])

        # ckpt = tf.train.latest_checkpoint('./checkpoint/')
        # saver.restore(session, ckpt)


        #下面这些注释是在训练过程测试用的
        datas, labels = read_data(test_data)
        # 对读入进来的数据进行一个整合处理
        v_data, v_label, nums, scalers = deal_data(datas, labels)
        v_data = np.reshape(v_data, [-1, 20000, 7])
        v_label = np.reshape(v_label, [-1])


        f_y_mse = []
        f_y_rmse = []
        f_R2 = []
        f_mae = []
        t_MAE = []
        t_RMSE = []
        t_R2 = []
        t_Loss = []
        batch_loss = []
        for i in range(epoch):
            print("this is epoch", i)
            t_data, t_label = shuffer(t_data, t_label)
            for batch in range(num // batch_size):
                losss,summary, step,pre, _ = session.run([loss,merged_summary_op,global_step,logits, optimizer], feed_dict={X: t_data[batch*batch_size:(batch +1)*batch_size, :, :],
                                                            Y: t_label[batch*batch_size:(batch +1)*batch_size]})

                pre = scaler.inverse_transform(pre)
                t_l = np.reshape(t_label[batch*batch_size:(batch +1)*batch_size],[-1,1])
                l = scaler.inverse_transform(t_l)

                summary_writer.add_summary(summary, step)
                y_mse, y_rmse, R2, mae = report_accuracy(pre, l)
                f_y_mse.append(y_mse)
                f_y_rmse.append(y_rmse)
                f_R2.append(R2)
                f_mae.append(mae)
            print(losss)
            t_loss, _, pres = session.run([loss, merged_summary_op, logits], feed_dict={X: t_data, Y: t_label})
            pres = scaler.inverse_transform(pres)
            tt_label = np.reshape(t_label, [-1, 1])
            tt_label = scaler.inverse_transform(tt_label)
            # R2是acc
            _, RMSE, acc, MAE = report_accuracy(pres, tt_label)
            t_MAE.append(MAE)
            t_R2.append(acc)
            t_RMSE.append(RMSE)
            t_Loss.append(t_loss)
            # mae
            var2 = pd.DataFrame(t_MAE)
            var3 = pd.DataFrame(t_R2)
            var4 = pd.DataFrame(t_RMSE)
            var5 = pd.DataFrame(t_Loss)
            # var1 = pd.DataFrame(batch_acc)
            path_data = './data/'
            var2.to_csv(path_data + '/MAE.csv', index=False, header=False)
            var3.to_csv(path_data + '/R2.csv', index=False, header=False)
            var4.to_csv(path_data + '/RMSE.csv', index=False, header=False)
            var5.to_csv(path_data + '/loss.csv', index=False, header=False)


            #下面这这些代码是训练过程中测试用的
            _, _, pres = session.run([loss, merged_summary_op, logits], feed_dict={X: v_data, Y: v_label})
            pres = scalers.inverse_transform(pres)
            vv_label = np.reshape(v_label, [-1, 1])
            vv_label = scalers.inverse_transform(vv_label)
            _,_,acc,_ = report_accuracy(pres, vv_label)
            print("测试集的",acc)
            if(acc >= 0.97):  #当准确率大于0.98的时候，保存下来的模型
                saver.save(session, './checkpoint/tcn_model', global_step=step)
                break
            saver.save(session, './checkpoint/tcn_model',global_step=step)
        #画图的东西
        font_format = {'family': 'Times New Roman', 'size': 18}
        font_format1 = {'family': 'Times New Roman', 'size': 16}
        plt.plot(f_y_mse, label='f_y_mse')
        plt.legend()
        plt.show()
        plt.plot(t_RMSE, linewidth=2.5)
        plt.xlabel('Epoch', font_format)
        plt.ylabel('RMSE', font_format)
        plt.legend(['RMSE'], prop=font_format1)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # # # # 调整上下左右四个边框的线宽为2
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.show()
        plt.plot(t_R2, linewidth=2.5)
        plt.xlabel('Epoch', font_format)
        plt.ylabel('R2', font_format)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # # # # 调整上下左右四个边框的线宽为2
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.legend(['R2'], prop=font_format1)
        plt.show()
        plt.plot(t_Loss, linewidth=2.5)
        plt.xlabel('Epoch', font_format)
        plt.ylabel('Loss', font_format)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # # # # 调整上下左右四个边框的线宽为2
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.legend(['Loss'], prop=font_format1)
        plt.show()
        plt.plot(t_MAE, linewidth=2.5)
        plt.xlabel('Epoch', font_format)
        plt.ylabel('MAE', font_format)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # # # # 调整上下左右四个边框的线宽为2
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        # 显示图例：
        plt.legend(['MAE'], prop=font_format1)
        plt.show()

    if(mode=='test'):
        test_data = './test_data/'
        # 读取数据，path_data是文件夹的路径
        data, label = read_data(test_data)
        # 对读入进来的数据进行一个整合处理
        t_data, t_label, num, scaler = deal_data(data, label)
        t_data = np.reshape(t_data, [-1, 20000, 7])
        t_label = np.reshape(t_label, [-1])
        ckpt = tf.train.latest_checkpoint('./checkpoint/')
        saver.restore(session, ckpt)
        losss, summary, pre = session.run([loss, merged_summary_op, logits],feed_dict={X: t_data,Y: t_label})
        # 这部分代码有问题
        pre = scaler.inverse_transform(pre)
        t_label = np.reshape(t_label, [-1,1])
        t_label = scaler.inverse_transform(t_label)
        acc = report_accuracy(pre, t_label)
        print("this is acc", acc)
        print("liss", losss)
        plt.plot(t_label, label='label')
        plt.plot(pre, label='pre')
        # 显示图例：
        plt.legend()
        plt.show()
        print(losss)








