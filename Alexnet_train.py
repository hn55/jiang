# encoding:utf-8
import tensorflow as tf
from gen_data_batch import get_next_batch, load_images_path, read_images, onehot
from Alexnet_model import Model
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'# 只显示 warning 和 E

##load imgs
train_path = r'./UCM/train'
train_data = load_images_path(train_path)
test_path = r'./UCM/test'
test_data = load_images_path(test_path)
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="inputs")
y_ = tf.placeholder(tf.float32, shape=[None, 21], name="labels")
out_dim = int(y_.get_shape()[1])
is_training = tf.placeholder(tf.bool, name="is_training")
# dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")
lr_steps = tf.Variable(0, trainable=False)

epoches = 30
learning_lr = 1e-4
train_batch_size = 64
test_batch_size = 64
train_iter_nums = len(train_data[0]) // train_batch_size
test_iter_nums = len(test_data[0]) // test_batch_size
decay_epoches = 20 * train_iter_nums
train_layers = ['fc6', 'fc7', 'fc8']
pretrained_weights_path = r'./bvlc_alexnet.npy'
model = Model(x, out_dim, is_training, False)  # 使用预训练模型

# vanilla single-task loss
# loss
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=model.y))
    # tf.add_to_collection("weights_l2_losses", cross_entropy)
    # loss_merge = tf.add_n(tf.get_collection("weights_l2_losses"),name='total_loss')
    # train_loss_summary = tf.summary.scalar('train_loss', cross_entropy)

##lr
with tf.name_scope('lr'):
    learning_rate = tf.train.exponential_decay(learning_lr, global_step=lr_steps, decay_steps=decay_epoches,
                                               decay_rate=0.9, staircase=True)  # lr decay=1

# optimizer
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads_vars = optimizer.compute_gradients(cross_entropy)
        train_op = optimizer.apply_gradients(grads_vars, global_step=lr_steps)

    # List of trainable variables of the layers we want to train
    # opt_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# performance metrics
# accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(model.y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # train_acc_summary = tf.summary.scalar('train_acc', accuracy)

# only save trainable and bn variables
# var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list = model.params+ bn_moving_vars
init = tf.global_variables_initializer()

##set GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
with tf.Session(config=config) as sess:
    sess.run(init)
    sess.run(tf.variables_initializer([lr_steps]))

    # load pretrain weights
    model.load_with_skip(sess, pretrained_weights_path, train_layers)

    ##start training
    for epoch in range(1, epoches + 1):
        print("Epoch:", epoch)
        acc_sum, loss_sum = 0, 0
        for iter in range(train_iter_nums):
            batch_x, batch_y = get_next_batch(train_data[0], train_data[1], train_batch_size, out_dim, data_aug=True)
            _, lr_steps_v, lr, acc, loss = sess.run([train_op, lr_steps, learning_rate, accuracy, cross_entropy],
                                                    feed_dict={x: batch_x, y_: batch_y, is_training: True})
            acc_sum += acc
            loss_sum += loss
        print("Epoch%d:初始模型训练精度：" % (epoch), str(acc_sum / train_iter_nums) + "   loss:" + str(
            loss_sum / train_iter_nums) + "   learning_rate:" + str(lr))

    # test
    test_accuracy_sum = 0
    for t in range(test_iter_nums):
        test_batch_x, test_batch_y = test_data[0][t * test_batch_size:(t + 1) * test_batch_size], \
                                     test_data[1][t * test_batch_size:(t + 1) * test_batch_size]
        test_batch_x = read_images(test_batch_x)
        test_batch_y = onehot(test_batch_y, out_dim)
        test_acc = sess.run(accuracy, feed_dict={x: test_batch_x, y_: test_batch_y, is_training: False})
        test_accuracy_sum += test_acc
    test_mean_acc = test_accuracy_sum / test_iter_nums
    print("original model #test acc：", test_mean_acc)
    saver.save(sess, "./Alexnet-model-save/my-model", global_step=1)