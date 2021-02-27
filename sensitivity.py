##对模型的每一层做sensitivity分析，确定该层合理的剪枝比例

#encoding:utf-8
import tensorflow as tf
import numpy as np
from copy import deepcopy
from Alexnet_model import Model
from gen_data_batch import get_next_batch,get_omega_batch,load_images_path,read_images,onehot
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'# 只显示 warning 和 E

##修剪单层
##对每一层单独进行剪枝，每次只剪一层，剪完再训练回复精度，之后再开始重新评估下一层灵敏度，对下一层进行剪枝
##修剪单层
def get_single_layer_mask(Omega_v,percent,layer,params):
    mask = deepcopy(Omega_v)  ##掩膜矩阵
    for dd in range(len(Omega_v)):
        mask[dd][mask[dd]>=0]=1 ##重要性矩阵中可能有0值，没有剪到下一层的时候，即使下一层omega为0，也不应把该参数剪掉
    if layer>=1:##从第二层开始，把之前层修剪掉的mask保持不变
        for dd in range(len(params)):
            mask[dd][abs(params[dd]) == 0] = 0
            # print("全1矩阵：",mask)
    if layer<5:##前五层卷积层（没有BN参数）
        for mm in range(layer*2,(layer+1)*2):##每层只有w和b两个参数
            # sorted(Omega_v[mm].reshape([-1]))
            threshold = np.percentile(abs(Omega_v[mm]), percent)
            # print("阈值：", threshold)
            mask[mm][abs(Omega_v[mm]) < threshold] = 0
    elif layer==5:
        for mm in range(10,14):##每层只有w和b两个参数
            threshold = np.percentile(abs(Omega_v[mm]), percent)
            # print("阈值：", threshold)
            mask[mm][abs(Omega_v[mm]) < threshold] = 0
    elif layer==6:
        for mm in range(14,18):##每层只有w和b两个参数
            threshold = np.percentile(abs(Omega_v[mm]), percent)
            # print("阈值：", threshold)
            mask[mm][abs(Omega_v[mm]) < threshold] = 0
    else:
        for mm in range(18,20):##每层只有w和b两个参数
            threshold = np.percentile(abs(Omega_v[mm]), percent)
            # print("阈值：", threshold)
            mask[mm][abs(Omega_v[mm]) < threshold] = 0
    return mask

def get_omg_mask(params):
    omg_mask=deepcopy(params)
    for aa in range(len(params)):
        omg_mask[aa][abs(params[aa])>0]=1 ##重要性矩阵中可能有0值，没有剪到下一层的时候，即使下一层omega为0，也不应把该参数剪掉
    return omg_mask


def apply_prune_weights(params,mask):
    assign_op={}
    for dd in range(len(params)):
        assign_op["%d"%dd]=params[dd].assign(tf.multiply(mask[dd], params[dd]))
    return assign_op

def restore_params(params,ori_params):
    # reassign optimal weights for latest task
    restore_op={}
    for v in range(len(params)):
        restore_op["%d" %v]=params[v].assign(ori_params[v])
    return restore_op

def print_test_acc(sess,test_iter_nums,test_data,test_batch_size,out_dim,accuracy):
    # test
    test_accuracy_sum = 0
    for t in range(test_iter_nums):
        test_batch_x, test_batch_y = test_data[0][t * test_batch_size:(t + 1) * test_batch_size], \
                                       test_data[1][t * test_batch_size:(t + 1) * test_batch_size]
        test_batch_x = read_images(test_batch_x)
        test_batch_y = onehot(test_batch_y, out_dim)
        test_acc = sess.run(accuracy,feed_dict={x: test_batch_x, y_: test_batch_y, is_training:False})
        test_accuracy_sum += test_acc
    test_mean_acc = test_accuracy_sum / test_iter_nums
    return test_mean_acc

##load imgs
train_path=r'./UCM/train'
test_path=r'./UCM/test'
train_data = load_images_path(train_path)
test_data = load_images_path(test_path)

##define
epoches = 100
learning_lr = 1e-4
train_batch_size = 64
test_batch_size = 64
Omega_batch_size=64
train_iter_nums = len(train_data[0]) // train_batch_size
test_iter_nums = len(test_data[0]) // test_batch_size
Omega_iter_nums = len(train_data[0]) // Omega_batch_size
decay_epoches = 20 * train_iter_nums
# prune_percent=[30,50,70,90,93,95,97,99]##稀疏度：百分比
prune_percent=list(range(1,101))
# prune_percent=[30,50,70]

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="inputs")
y_ = tf.placeholder(tf.float32, shape=[None, 21], name="labels")
out_dim = int(y_.get_shape()[1])
is_training = tf.placeholder(tf.bool, name="is_training")
# dropout = tf.placeholder(tf.float32, name="dropout_keep_prob") ##每个神经元被保留的概率
lr_steps = tf.Variable(0, trainable=False)
model = Model(x, out_dim, is_training, True)

mask_matrix = [tf.placeholder(tf.float32) for kk in range(len(model.params))]
ori_params_matrix = [tf.placeholder(tf.float32) for dd in range(len(model.params))]
omg_mask_matrix=[tf.placeholder(tf.float32) for sj in range(len(model.params))]###这个掩膜矩阵用来添加到计算参数重要性上，来保证已经修剪掉的参数不会参与到重要性的计算
# loss
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=model.y))

##lr
with tf.name_scope('lr'):
    learning_rate = tf.train.exponential_decay(learning_lr, global_step=lr_steps, decay_steps=decay_epoches,decay_rate=0.9, staircase=True)  # lr decay=1

# optimizer
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads_vars = optimizer.compute_gradients(cross_entropy)
        train_op = optimizer.apply_gradients(grads_vars, global_step=lr_steps)

# accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(model.y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# calculate params importance
with tf.name_scope('params_importance'):
    probs = tf.nn.softmax(model.y)

    # # information entropy
    gradients = tf.gradients(2*tf.nn.l2_loss(probs), model.params)  # Gradient of the loss function for the current task
    # params_square = [tf.square(var) for var in model.params]
    gradient_square = [tf.square(g) for g in gradients]
    param_importance = [tf.abs(omg_mask_matrix[k]*gradients[k] + omg_mask_matrix[k]*gradient_square[k]) for k in range(len(gradients))]##计算出来的参数重要性可能有负数
    # # cal importance --max（0,a）
    # param_importance = [tf.abs(var_g_merge[n]) for n in range(len(var_g_merge))]##计算出来的参数重要性可能有负数

with tf.name_scope('prune_weights'):
    weights_prune = apply_prune_weights(model.params, mask_matrix)

with tf.name_scope('restore_weights'):
    weights_restore = restore_params(model.params, ori_params_matrix)

# only save trainable and bn variables：将BN参数保存进去
# var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list = model.params + bn_moving_vars
print(var_list)
print(len(var_list))

##set GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

with tf.Session(config=config) as sess:
    layer_pruning = 8  ###第几层剪枝
    print("第%d层剪枝：" % (layer_pruning))
    if layer_pruning == 1:
        model_path = r'./Alexnet-model-save/'
    else:
        model_path = r'./prune-Alexnet-model-save-%d/' % (layer_pruning - 1)

    sess.run(init)  ##Adam优化器或者BN之类的也存在变量，因此，需要初始化
    ckpt = tf.train.get_checkpoint_state(model_path)
    print(ckpt)
    saver.restore(sess, ckpt.model_checkpoint_path)

    original_params = sess.run(model.params)
    for weight in original_params:
        print("第1轮before pruning #non zero parameters： "  + str(np.sum(weight != 0)))

    # test
    test_mean_acc1 = print_test_acc(sess,test_iter_nums,test_data,test_batch_size,out_dim,accuracy)
    print("第1轮before pruning #test acc：" , test_mean_acc1)

    omg_mask=get_omg_mask(original_params)##获得参数重要性的掩膜矩阵
    # print(omg_mask)
    Omg_mask_feed_dict ={ff: hh for ff, hh in zip(omg_mask_matrix, omg_mask)}

    ori_params=deepcopy(original_params)
    # mask_omega = deepcopy(original_params)  ##让剪掉的参数重要性变为0，不参与计算，掩膜矩阵
    # Omega_v = deepcopy(original_params)
    # for pdd in range(len(original_params)):
    #     mask_omega[pdd][mask_omega[pdd] >= 0] = 1  ##重要性矩阵中可能有0值，没有剪到下一层的时候，即使下一层omega为0，也不应把该参数剪掉
    # if layer_pruning >= 2:  ##从第二层开始，把之前层修剪掉的mask保持不变
    #     for qdd in range(len(original_params)):
    #         mask_omega[qdd][abs(original_params[qdd]) == 0] = 0
            # print("全1矩阵：",mask)
    # calculate params importance
    omega_sum=0
    for Omega_iter in range(Omega_iter_nums):
        Omega_batch_imgs= get_omega_batch(train_data[0], Omega_batch_size,out_dim, data_aug=False)
        Omg_mask_feed_dict[x]=Omega_batch_imgs
        Omg_mask_feed_dict[is_training]=True
        gradient_order = sess.run(param_importance,feed_dict=Omg_mask_feed_dict)  ##计算参数重要性不需要标签
        omega_sum += np.array(gradient_order)  ##改成累加求和，最后求平均值
    Omega_v = omega_sum / Omega_iter_nums
    # for idd in range(len(Omega_p)):
    #     Omega_v[idd] = np.multiply(mask_omega[idd], Omega_p[idd])
    for Omega_v_cut in Omega_v:
        print("第0轮剪枝后Omega_v： " + str(np.sum(Omega_v_cut!= 0)))
    with open(r'./Omega_v_%d'%(layer_pruning),'wb') as f1:
        pickle.dump(Omega_v,f1,pickle.HIGHEST_PROTOCOL)


    ##迭代剪枝
    print("第%d层开始剪枝： "%(layer_pruning))
    f=open(r'./layer_%d_sensitivity.txt' % (layer_pruning), 'w')
    for prune_iter in range(len(prune_percent)):##每一层的剪枝比
        mask=get_single_layer_mask(Omega_v,prune_percent[prune_iter],layer_pruning-1,original_params)

        ##修剪权重：
        mask_weights_feed_dict = {mm: nn for mm, nn in zip(mask_matrix, mask)}  ##迭代传值：需要传mask、x和y_标签，x和y_可以添加到mask_feed_dict这个字典里
        ori_params_feed_dict={aa: bb for aa, bb in zip(ori_params_matrix, ori_params)}
        prune_params = sess.run(weights_prune, feed_dict=mask_weights_feed_dict)
        for key, weight_cut in prune_params.items():
            # print(key)
            print("layer_%d  #non zero parameters： " % (layer_pruning) + str(np.sum(weight_cut != 0)))
        #test
        after_prune_test_mean_acc = print_test_acc(sess,test_iter_nums,test_data,test_batch_size,out_dim,accuracy)
        print("layer_%d #test acc："%(layer_pruning),str(prune_percent[prune_iter])+'\t'+str(after_prune_test_mean_acc))
        f.write(str(prune_percent[prune_iter])+'\t'+str(after_prune_test_mean_acc)+'\n')

        ##控制单变量，还原初始权重，以便下一次剪枝
        params_ori=sess.run(weights_restore,feed_dict=ori_params_feed_dict)