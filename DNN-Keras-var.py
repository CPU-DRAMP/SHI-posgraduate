import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

np.random.seed(16)  # 定个随机种子
tf.random.set_seed(16)


# 创建Tensorboard log日志及其内文件的函数
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") + parameter
    run_id = parameter
    return os.path.join(root_logdir, run_id)


# 每条训练样本由n+2列构成，其中第一列[0]是DBAASP的ID，最后一列[-1]是二分类结果，[1:-1]是分子描述符

train_file = "C:/Users/Shi_Crazy/Desktop/Ensemble/varthreshold_defesin_staple_descriptors.csv"


# pred_file="../stapled_pep_test.csv"


def read_file(filename):  # 读取训练资料（即所有包含二分类标签的数据）
    df = pd.read_csv(filename)
    cols = df.columns  # 返回所有列名
    #     id_cols = list(cols[0])  #返回列名id
    features_cols = list(cols[1:240])  # 返回特征列名
    labels_cols = list(cols[240:])  # 返回结果列名
    train_df = df[features_cols + labels_cols]  # 返回训练数据(特征加结果)
    return train_df


train_df = read_file(train_file)


# 特征归一化：将非独热码特征缩放至0~1
# shuffle：打乱样本排序 数据拆分：将数据集拆分为训练集和测试集，测试集占比20%


def scale_fea(scaler, samples_train_x, samples_test_x):  # sc是scaler（即StandardScaler()）
    features_train_x_std = samples_train_x.values[:, :-1]  # 分子描述符特征值
    features_test_x_std = samples_test_x.values[:, :-1]
    scaler.fit(features_train_x_std)  # fit&transform
    features_train_x_std = scaler.transform(features_train_x_std)
    features_test_x_std = scaler.transform(features_test_x_std)
    return features_train_x_std, features_test_x_std


def gen_feature(train_df):
    train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle+sequencing
    l_col = train_df.columns[-1]
    labels = train_df[l_col]
    print(labels)
    features = train_df.drop(columns=[l_col])
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=labels)
    scaler = StandardScaler()
    train_x, test_x = scale_fea(scaler, train_x, test_x)
    return train_x, train_y, test_x, test_y, scaler


train_x, train_y, test_x, test_y, scaler = gen_feature(train_df)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
# 求class_weight并转换成字典
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_y),
                                                 train_y)
class_weight = dict(enumerate(class_weight))
print(class_weight)


def build_model(featureLen, optimizer, units, learning_rate):  # featureLen代表特征数目
    # 调参部分，几层神经网络，每层几个神经元，激活函数用什么
    # binary_crossentropy的loss对应的最后一层激活函数最好为sigmoid，而categorical_crossentropy的loss对应的最后一层激活函数为softmax
    # 你所使用的损失函数(loss)需要和评估指标(metrics)相对应
    global network
    if optimizer == 'SGD':
        network = Sequential()
        network.add(Dense(units=units, activation='relu', input_shape=(featureLen,)))
        network.add(Dense(units=units, activation='relu'))
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        network.compile(loss='binary_crossentropy', optimizer=opt, metrics=[BinaryAccuracy()])
    elif optimizer == "NAG":
        network = Sequential()
        network.add(Dense(units=units, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          kernel_initializer="he_normal", input_shape=(featureLen,)))
        network.add(BatchNormalization())
        network.add(Dense(units=units, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          kernel_initializer="he_normal"))
        network.add(BatchNormalization())
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        network.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=[BinaryAccuracy(),
                                                                                           Precision(), Recall(),
                                                                                           AUC()])
    elif optimizer == "AdaGrad":
        network = Sequential()
        network.add(Dense(units=units, activation='relu', input_shape=(featureLen,)))
        network.add(Dense(units=units, activation='relu'))
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        network.compile(loss='binary_crossentropy', optimizer=opt, metrics=[BinaryAccuracy()])
    elif optimizer == "RMSprop":
        network = Sequential()
        network.add(Dense(units=units, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          kernel_initializer="he_normal", input_shape=(featureLen,)))
        network.add(BatchNormalization())
        network.add(Dense(units=units, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          kernel_initializer="he_normal"))
        network.add(BatchNormalization())
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        network.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=[BinaryAccuracy(),
                                                                                           Precision(), Recall(),
                                                                                           AUC()])
    elif optimizer == "Adam":
        network = Sequential()
        network.add(Dense(units=units, activation='elu', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(featureLen,)))
        network.add(BatchNormalization())
        network.add(Dense(units=units, activation='elu', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        network.add(BatchNormalization())
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        network.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=[BinaryAccuracy(),
                                                                                           Precision(), Recall(),
                                                                                           AUC()])
    elif optimizer == 'Adamax':
        network = Sequential()
        network.add(Dense(units=units, activation='relu', input_shape=(featureLen,)))
        network.add(Dense(units=units, activation='relu'))
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
        network.compile(loss='binary_crossentropy', optimizer=opt, metrics=[BinaryAccuracy()])
    elif optimizer == 'Nadam':
        network = Sequential()
        network.add(Dense(units=units, activation='relu', input_shape=(featureLen,)))
        network.add(Dense(units=units, activation='relu'))
        network.add(Dense(units=1, activation='sigmoid'))
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        network.compile(loss='binary_crossentropy', optimizer=opt, metrics=[BinaryAccuracy()])
    return network


def train_model(train_x, train_y, valid_size, featureLen, optimizer, model_path, epoch, batch, patience, units,
                learning_rate, train_type="train"):
    global history
    network = build_model(featureLen, optimizer, units, learning_rate)
    valid_x = train_x[:valid_size]  # 验证集特征
    partial_train_x = train_x[valid_size:]  # 除验证集外的训练集特征
    valid_y = train_y[:valid_size]  # 验证集结果
    partial_train_y = train_y[valid_size:]  # 除验证集外的训练集结果
    es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=patience)
    ckpt = ModelCheckpoint(model_path, monitor='val_auc', save_best_only=True, verbose=1, mode='max')
    if 'train' == train_type:  # 训练数据不包括验证集
        # history = network.fit(partial_train_x, partial_train_y, epochs=epoch, batch_size=batch,
        #                       validation_data=(valid_x, valid_y), callbacks=[es, ckpt], verbose=1)
        history = network.fit(partial_train_x, partial_train_y, epochs=epoch, batch_size=batch,
                              validation_data=(valid_x, valid_y), callbacks=[es, ckpt, tensorboard_callback], verbose=1,
                              class_weight=class_weight)
    elif 'total' == train_type:  # 训练数据包括验证集
        # history = network.fit(train_x, train_y, epochs=epoch, batch_size=batch, validation_data=(valid_x, valid_y),
        #                       callbacks=[es, ckpt])  # ,callbacks=[es]
        history = network.fit(train_x, train_y, epochs=epoch, batch_size=batch, validation_data=(valid_x, valid_y),
                              callbacks=[es, ckpt, tensorboard_callback], verbose=1,
                              class_weight=class_weight)  # ,callbacks=[es]
    return network, history


def train_day_model(train_x, train_y, optimizer, model_path, epoch, batch, patience, units, learning_rate,
                    train_type='train'):
    valid_size = int(train_x.shape[0] / 4)
    featureLen = train_x.shape[1]
    network, history = train_model(train_x, train_y, valid_size, featureLen, optimizer, model_path, epoch, batch,
                                   patience, units, learning_rate, train_type)
    #     network.save("model."+str(optimizer)+".h5")
    return network, history


def metrics_evaluation(test_x, test_y, network):
    pred_y = network(test_x)  # 这两步的到的结果和下面那一步是一样的
    pred_y = pred_y.numpy()
    #     pred_y = network.predict(test_x)
    #     print(pred_y)
    pred_y_quant = network.predict_on_batch(test_x)
    #     print(pred_y_quant)
    class_names = ["NonAntiGram-", "AntiGram-"]
    for i in range(len(pred_y)):
        if pred_y[i] < 0.5:
            pred_y[i] = 0
        elif pred_y[i] > 0.5:
            pred_y[i] = 1
        else:
            pred_y[i] = random.choice([0, 1])

    test_loss, test_acc, test_precision, test_recall, test_auc = network.evaluate(test_x, test_y)
    print('test_loss:%f,test_acc:%f' % (test_loss, test_acc))
    print("accuracy_score:", accuracy_score(test_y, pred_y))
    print("precision_score:", precision_score(test_y, pred_y))
    print("recall_score:", recall_score(test_y, pred_y))
    print("F1_score:", f1_score(test_y, pred_y))
    print("MCC:", matthews_corrcoef(test_y, pred_y))

    #     print(test_y,pred_y)
    print(classification_report(test_y, pred_y, target_names=class_names, digits=3))
    matrix = confusion_matrix(test_y, pred_y)
    print(matrix)

    fpr, tpr, thresholds = roc_curve(test_y, pred_y_quant)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='0.3')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 14
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate(1 - Specificity)')
    plt.ylabel('True Positive Rate (Sentivity)')
    plt.grid(True)
    print('ROC_AUC:', auc(fpr, tpr))


def plot_history(history):
    # 获取训练集和测试集的损失历史数值
    history_dict = history.history
    training_loss = history_dict['loss']
    validation_loss = history_dict['val_loss']
    training_acc = history_dict['binary_accuracy']
    validation_acc = history_dict['val_binary_accuracy']

    # 为每个epoch创建编号
    epoch_count_1 = range(1, len(training_loss) + 1)
    epoch_count_2 = range(1, len(training_acc) + 1)

    # 画出损失的历史数值
    plt.subplot(1, 2, 1)
    plt.plot(epoch_count_1, training_loss, 'r--')
    plt.plot(epoch_count_1, validation_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])

    plt.subplot(1, 2, 2)
    plt.plot(epoch_count_2, training_acc, 'r--')
    plt.plot(epoch_count_2, validation_acc, 'b-')
    plt.legend(['Training Acc', 'Validation Acc'])


root_logdir = os.path.join(os.curdir, "log")
parameter = "Adam_elu_he_normal_l20.0001_batchsize16_units120_lr0.0005"
run_logdir = get_run_logdir()
print(run_logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(run_logdir)
model_path_1 = "./model.Adam_elu_he_normal_l20.0001_batchsize16_units120_lr0.0005.h5"
epoch = 300
batch = 16
patience = 15
units = 120
learning_rate = 0.0005
network, history = train_day_model(train_x, train_y, "Adam", model_path_1, epoch, batch, patience, units, learning_rate)
print(network.summary())
# plt.figure(figsize=(15, 5))
# plot_history(history)  分开绘图

# 绘图（loss，accuracy，roc_auc曲线）
pd.DataFrame(history.history).plot(figsize=(15, 10))
plt.grid(True)
plt.gca().set_ylim(-0.05, 1.05)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
metrics_evaluation(test_x, test_y, network)
plt.show()

# # 加载神经网络
# from keras.models import load_model
# network = load_model('model.h5')
