# Import relevant packages and fix the seeds for repeatability
import os
import pickle
import random
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
import tensorflow.compat.v1 as tf
#import tensorflow as tf
from keras import backend as K

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session_conf = InteractiveSession(config=config)

# Basic setting
warnings.filterwarnings('ignore')
np.random.seed(int(time.time()))
random.seed(1)
tf.set_random_seed(1)

# USE GPU
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# os.environ['PYTHONHASHSEED'] = '0'
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# session_conf = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
# print(tf.test.is_gpu_available())

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9


# ------------------------------ #
#       Generate dataset         #
# ------------------------------ #


def GenerateDatasetAsPNG(path_x, path_y):
    f = open(path_y, 'r')  # open txt file (only read)
    contents = f.readlines()
    f.close()
    x, y_ = [], []
    idpng = 1
    for content in contents:
        value = content.split()  
        img_path = path_x + str(idpng) + '.png'
        img = Image.open(img_path)
        img = img.resize((60, 60), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L')) 
        img_arr = img_arr / 255.
        x.append(img_arr)
        y_.append(value[0])    
        idpng += 1
        if idpng == 1001:
            break

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.float32)
    return x, y_


print('-' * 10 + 'Generate Dataset' + '-' * 10)
# train_x_path = './X_train.npy/'
train_x_path = 'train_x_path'
train_y_path = 'train_y_path'
X_train, Y_train = GenerateDatasetAsPNG(train_x_path, train_y_path)  # read training set (x, y)
X_train = np.expand_dims(X_train, axis=3)  # change the shape of x to [batch height width channel]
Y_train = np.reshape(Y_train, (len(Y_train), -1))  # change the shape of y to [[y0],[y1]..]

# Filter out outliers using inter-quartile range, delete the data out of boundary
# IQR = - np.quantile(Y_train, 0.25) + np.quantile(Y_train, 0.75)
# lower_bound, upper_bound = np.quantile(Y_train, 0.25) - 1.5 * IQR, np.quantile(Y_train, 0.75) + 1.5 * IQR
# idx, val = np.where((Y_train >= lower_bound) & (Y_train <= upper_bound))
# Y_train = Y_train[idx]
# X_train = X_train[idx]

# Sanity check on the data shape and range of bandgaps
X_train = X_train[0:1000, :, :, :]
Y_train = Y_train[0:1000, :]
print('-' * 10 + 'Check Dataset' + '-' * 10)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('Y_train[0]:', Y_train[0])
print('Y_train_max:', np.max(Y_train))
print('Y_train_min:', np.min(Y_train))

plt.imshow(X_train[1])
plt.savefig('example1.jpg')

# Create placeholders
tf.disable_eager_execution()
y_size = int(X_train.shape[1])
real_data = tf.placeholder(tf.float32, shape=[None, y_size, y_size, 1])
z = tf.placeholder(tf.float32, shape=[None, 128])
y = tf.placeholder(tf.float32, [None, 1])


# ------------------------------ #
#     Construct GCGAN Model      #
# ------------------------------ #

# Leaky Relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


# Structure of the generator



def generator(z, y_, reuse=None):
    with tf.variable_scope('G', reuse=reuse):
        y_ = tf.concat([z, y_], axis=1)  # Connect z and y
        hidden2 = tf.layers.dense(y_, units=10 * 10 * 512)
        R1 = tf.reshape(hidden2, shape=[-1, 10, 10, 512])

        for i in range(5):
            # for i in range(3):
            C1 = tf.layers.conv2d_transpose(R1, filters=512, kernel_size=3, strides=(1, 1), padding='same')
            B1 = tf.layers.batch_normalization(C1)
            R1 = lrelu(B1)

        C1 = tf.layers.conv2d_transpose(R1, filters=512, kernel_size=3, strides=(2, 2), padding='same')
        B1 = tf.layers.batch_normalization(C1)
        R1 = lrelu(B1)

        C2 = tf.layers.conv2d_transpose(R1, filters=256, kernel_size=3, strides=(3, 3), padding='same')
        B2 = tf.layers.batch_normalization(C2)
        R2 = lrelu(B2)

        C3 = tf.layers.conv2d_transpose(R2, filters=1, kernel_size=3, strides=(1, 1), padding='same')
        O = tf.nn.tanh(C3)
        return O


# Structure of the discriminator
def discriminator(X, y_, reuse=None):
    with tf.variable_scope('D', reuse=reuse):
        y1 = tf.concat([X, y_], axis=1)

        y2 = tf.layers.dense(y1, units=256)
        y2 = lrelu(y2)
        y2 = tf.layers.dropout(y2, 0.5)

        yout = tf.layers.dense(y2, units=1)
        O = tf.nn.sigmoid(yout)

        return O


# Structure of regressor
def regressor(X, reuse=None):
    with tf.variable_scope('R', reuse=reuse):
        tower0 = tf.layers.conv2d(X, filters=32, kernel_size=(1, 1), padding='same')

        tower1 = tf.layers.conv2d(X, filters=64, kernel_size=(1, 1), padding='same')
        tower1 = tf.layers.conv2d(tower1, filters=64, kernel_size=(3, 3), padding='same')

        tower2 = tf.layers.conv2d(X, filters=32, kernel_size=(1, 1), padding='same')
        tower2 = tf.layers.conv2d(tower2, filters=32, kernel_size=(5, 5), padding='same')

        tower3 = tf.layers.max_pooling2d(X, pool_size=(3, 3), strides=(1, 1), padding='same')
        tower3 = tf.layers.conv2d(tower3, filters=32, kernel_size=(1, 1), padding='same')

        h = tf.concat([tower0, tower1, tower2, tower3], axis=-1)
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(1, 1), padding='same')

        for i in range(6):
            # for i in range(3):
            tower0 = tf.layers.conv2d(h, filters=32, kernel_size=(1, 1), padding='same')
            tower1 = tf.layers.conv2d(h, filters=64, kernel_size=(1, 1), padding='same')
            tower1 = tf.layers.conv2d(tower1, filters=64, kernel_size=(3, 3), padding='same')

            tower2 = tf.layers.conv2d(h, filters=32, kernel_size=(1, 1), padding='same')
            tower2 = tf.layers.conv2d(tower2, filters=32, kernel_size=(5, 5), padding='same')

            # tower3 = tf.layers.max_pooling2d(h, pool_size=(3, 3), strides=(1, 1), padding='same')
            # tower3 = tf.layers.conv2d(tower3, filters=32, kernel_size=(1, 1), padding='same')
            #
            # h = tf.concat([tower0, tower1, tower2, tower3], axis=-1)
            h = tf.concat([tower0, tower1, tower2], axis=-1)
            h = tf.nn.relu(h)
            if i % 2 == 0 and i != 0:
                h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(1, 1), padding='same')

        Z0 = tf.layers.flatten(h)
        Z0 = tf.layers.dense(Z0, units=512, activation='relu')
        Z0 = tf.layers.dropout(Z0, 0.2)

        Z0 = tf.layers.dense(Z0, units=32)
        Z_out = Z0

        O = tf.layers.dense(Z0, units=1)

        return O, Z_out



# Helper function of L2 Loss
def L2_Loss(y, y_hat):
    return tf.reduce_mean((y - y_hat) ** 2)


# Generate data
fake_data = generator(z, y)

# Feed both real and fake data into the regressor for latent features
true_pred, true_logit = regressor(real_data)
fake_pred, fake_logit = regressor(fake_data, reuse=True)

# Use latent feature, bandgap combination to perform authentication
true_label_pred = discriminator(true_logit, y)
fake_label_pred = discriminator(fake_logit, y, reuse=True)

gen_sample = generator(z, y, reuse=True)
gen_pred, gen_logit = regressor(gen_sample, reuse=True)

# Regressor helper loss:
R_loss = L2_Loss(true_pred, y)

# Discriminator losses:
D_real_loss = L2_Loss(true_label_pred, 0.9 * tf.ones_like(true_label_pred))
D_fake_loss = L2_Loss(fake_label_pred, tf.zeros_like(fake_label_pred))
D_loss = (D_real_loss + D_fake_loss) / 2

# Generator loss:
G_loss = L2_Loss(fake_label_pred, 0.9 * tf.ones_like(fake_label_pred)) + 25 * L2_Loss(fake_pred, y)

# Getting variables
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'D' in var.name]
g_vars = [var for var in tvars if 'G' in var.name]
r_vars = [var for var in tvars if 'R' in var.name]

# Opimizer
lr = 
opt_d = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=d_vars)
opt_g = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=g_vars)
opt_r = tf.train.AdamOptimizer(learning_rate=lr).minimize(R_loss, var_list=r_vars)

# ------------------------------ #
#       Training process         #
# ------------------------------ #

epochs_gan = 
batch_size = 
batches = X_train.shape[0] // batch_size
saver = tf.train.Saver()
tol_time = 0

# Gather the losses for plot
D_Losses = []
G_Losses = []
R_Losses = []

print('-' * 10 + 'Training Process' + '-' * 10)

with tf.Session(config=session_conf) as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs_gan + 1):
        a = time.time()
        D_Losses_ = []
        G_Losses_ = []
        R_Losses_ = []

        for b in range(batches):
            idx = np.arange(b * batch_size, (b + 1) * batch_size)
            train = X_train[idx, :, :, :]
            batch_y = Y_train[idx, :]
            batch_z = np.random.normal(0, 1, size=(batch_size, 128))

            # Each round, G, R and D will be trained using a batch of data
            G_loss_, _ = sess.run([G_loss, opt_g], feed_dict={y: batch_y, z: batch_z})
            R_loss_, _ = sess.run([R_loss, opt_r], feed_dict={real_data: train, y: batch_y})
            D_loss_, _ = sess.run([D_loss, opt_d], feed_dict={real_data: train, y: batch_y, z: batch_z})
            D_Losses_.append(D_loss_)
            G_Losses_.append(G_loss_)
            R_Losses_.append(R_loss_)

        # For timing purposes
        b = time.time()
        delta_t = b - a
        tol_time += delta_t

        D_Losses.append(np.mean(D_Losses_))
        G_Losses.append(np.mean(G_Losses_))
        R_Losses.append(np.mean(R_Losses_))

        print('-' * 20)
        print('Currently on epoch {} of {}'.format(e, epochs_gan))
        print('G Loss: {}, D Loss: {}, R Loss: {}'.format(np.mean(G_Losses_), np.mean(D_Losses_), np.mean(R_Losses_)))

        # Generate data and plot loss functions for sanity check every 10 epochs
        if e % 10 == 0:
            samples = []
            scores = []
            sample_y = np.random.uniform(0.49, 2.15, size=(1, 1))
            sample_y = np.round(sample_y, 5)
            sample_y_ = sample_y * np.ones([100, 1])
            sample_z = np.random.normal(0, 1, size=(100, 128))
            gen_sample_, pred_score = sess.run([gen_sample, gen_pred], feed_dict={y: sample_y_, z: sample_z})
            samples = np.array(gen_sample_)
            samples = samples.reshape([100, y_size, y_size, 1])

            # samples = np.where(samples > 0.75, 1, samples)
            # samples = np.where((samples > 0.25) & (samples < 0.75), 0.5, samples)
            # samples = np.where(samples < 0.25, 0, samples)

            # Gray scale normalization value: 0.5 - solid  0 - hole
            samples = np.where(samples > 0.5, 0.5, samples)
            samples = np.where((samples > 0) & (samples < 0.25), 0.25, samples)
            samples = np.where(samples < 0, 0, samples)

            S_ = samples.tolist()
            not_in = 0
            for s_ in S_:
                if s_ not in X_train.tolist():
                    not_in += 1

            print('Current number of unique samples: {}'.format(np.shape(np.unique(samples[:, :, :, 0], axis=0))[0]))
            print('------------')
            print('Currently Generating {}'.format(sample_y))
            print('Predicted bandgap: {}'.format(np.mean(pred_score)))
            print('Samples not in database :{}'.format(not_in))
            print('------------')

            if e != 0:
                plt.figure()
                plt.plot(G_Losses)
                plt.plot(D_Losses)
                plt.plot(R_Losses)
                plt.legend(['G Loss', 'D Loss', 'R Loss'])
                # plt.show()
                plt.savefig('LOSS_' + str(e) + '.jpg')

        # Save the model every 10 epochs
        if e % 10 == 0:
            print('-----------')
            print('Saving current model')
            saver.save(sess, "cGAN_model.ckpt")
            with open('cGAN_Losses.pickle', 'wb') as f:
                pickle.dump((G_Losses, D_Losses, R_Losses), f)

# Average time spent on training
print('average running time: {}'.format(tol_time / epochs_gan))

# Check on the accuracy of the regressor
print('-' * 10 + 'Check on the accuracy of the regressor' + '-' * 10)
saver = tf.train.Saver()
batch_size = 2

batches = X_train.shape[0] // batch_size
# batches = 100
preds = []

with tf.Session(config=session_conf) as sess:
    saver.restore(sess, 'cGAN_model.ckpt')
    for b in range(batches):
        if b % 2 == 0:
            print('Currently: {}/{}'.format(b, batches))
        idx = np.arange(b * batch_size, (b + 1) * batch_size)
        data = X_train[idx, :, :, :]
        predicted, _ = sess.run(regressor(real_data, reuse=True), feed_dict={real_data: data})
        preds.extend(predicted.reshape([-1]))

# Fractional MAE for regressor
preds = np.asarray(preds)
print(preds.shape)
print(Y_train.shape)
print('Fractional MAE: {}'.format(np.mean(np.abs((preds - Y_train.reshape([-1])) / Y_train.reshape([-1])))))

# Generate new data for testing purposes
print('-' * 10 + 'Generate new data for testing purposes' + '-' * 10)
OUT = pd.DataFrame()
saver = tf.train.Saver()
for K in range(5 + 1):
    print('Progress: {}/{}'.format(K, 5))
    with tf.Session(config=session_conf) as sess:
        saver.restore(sess, 'cGAN_model.ckpt')
        print('------------')
        Samples = []
        labels = []
        Inception_scores = []
        gen_preds = []

        for i in range(2):
            labels_pred = []
            sample_y = np.random.uniform(1.0, 2.0, size=(1, 1))  # required value 1.0~2.0
            sample_y = np.round(sample_y, 4)
            sample_y_ = sample_y * np.ones([100, 1])

            sample_z = np.random.normal(0, 1, size=(100, 128)) # noise
            gen_sample = sess.run(generator(z, y, reuse=True), feed_dict={z: sample_z, y: sample_y_})

            # gen_sample = np.where(gen_sample < 0, -1, 1) # original paper setting: hole=-1 solid=1
            gen_sample = np.where(gen_sample < 0.125, 0, 0.25)
            gen_score = sess.run(R_loss, feed_dict={real_data: gen_sample, y: sample_y_})

            Inception_scores.extend(list(np.unique(gen_score)))

            gen_pred_, _ = sess.run(regressor(real_data, reuse=True),
                                    feed_dict={real_data: np.unique(gen_sample, axis=0)})
            gen_preds.append(gen_pred_)

            # print('------------')
            gen_sample = np.array(gen_sample)
            gen_sample = gen_sample.reshape([100, y_size, y_size, 1])

            # print('gen_sample')
            # print(gen_sample.shape)
            # print(gen_sample)
            # print('gen_pred')
            # print(gen_pred.shape)
            # print(gen_pred_)
            # print('sample_y')
            # print(sample_y)

            # plt.imshow(gen_sample[0])
            # plt.savefig('gen_sample' + str(i) + '.jpg')

            Samples.extend(list(np.unique(gen_sample, axis=0)))  # delete repeat samples
            labels.extend(list(sample_y * np.ones(shape=(1, np.shape(np.unique(gen_sample[:, :, :, 0], axis=0))[0]))))

    labels_ = []
    for l in labels:
        for j in range(len(l)):
            labels_.append(l[j])

    Gen_preds_ = []
    for l in gen_preds:
        for j in range(len(l)):
            Gen_preds_.append(l[j])

    Samples_ = np.where(np.array(Samples) < 0.125, 0, 0.25)
    Samples_ = Samples

    summary = {'X': [], 'require_label': [], 'predicted_label': []}

    for s, l, gen_p in zip(Samples_, labels_, Gen_preds_):
        # summary['X'].append(int('0b' + ''.join(list(s.reshape([y_size ** 2, ]).astype('str').tolist())), 2))
        # sample_out = list(s.reshape([y_size ** 2, ]).astype('int'))
        sample_out = list(s.reshape([y_size ** 2, ]).astype('float'))

        summary['X'].append(sample_out)
        summary['require_label'].append(l)
        summary['predicted_label'].append(gen_p[0])

        # print('sample_out')
        # print(sample_out)
        # print('require_label')
        # print(l)
        # print('predicted_label')
        # print(gen_p[0])

    # print(summary['X'])
    # print('require_label')
    # print(summary['require_label'])
    # print('predicted_label')
    # print(summary['predicted_label'])
    out = pd.DataFrame(summary)
    out = out.sort_values('X').reset_index().drop('index', axis=1)
    OUT = pd.concat([OUT, out], axis=0, ignore_index=True)

    with open('summary.pickle', 'wb') as f:
        pickle.dump(OUT, f)

# Fractional MAE for generation
r = OUT.require_label.values
p = OUT.predicted_label.values

print('Fractional MAE: {}'.format(np.mean(np.abs((p - r) / r))))
OUT.to_csv('summary.csv', index=False)
