#encoding:utf-8
import numpy as np
import scipy.misc as misc
import random
import os
import skimage
from skimage import color
# import imgaug.augmenters as iaa
def onehot(label, class_num):
    labels = np.zeros([len(label), class_num])
    labels[np.arange(len(label)), label] = 1
    labels = np.reshape(labels, [-1, class_num])
    return labels

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.choice([0,1])):  ##循环每次随机为True or False
            batch[i] = np.fliplr(batch[i])
    return batch
def _random_noise(batch):
    for i in range(len(batch)):
        if bool(random.choice([0,1])):
            batch[i]=skimage.util.random_noise(batch[i],mode='gaussian')##添加高斯噪声
    return batch
def load_images_path(dataset_path):
    classes = os.listdir(dataset_path)
    image=[]
    label=[]
    class_num=len(classes)
    for i in range(class_num):
        class_path = os.path.join(dataset_path, classes[i])
        imgs = os.listdir(class_path)
        img_num = len(imgs)
        for j in range(img_num):
            img = os.path.join(class_path, imgs[j])#图片路径
            image.append(img)
            label.append(i)
    return image,label

def read_images(batch_image):
    batch_x=[]
    for img in batch_image:
        img_arr=misc.imread(img)
        resize_image = misc.imresize(img_arr, [224,224,3])
        if resize_image.shape != (224,224, 3):
            resize_image = color.grey2rgb(resize_image)/255.0
        else:
            resize_image = resize_image/255.0
        batch_x.append(resize_image)
    return np.array(batch_x)
def get_next_batch(train_img,train_label,batch_size,out_dim,data_aug=True):
    index = np.random.choice(len(train_img), batch_size)
    # print(index)
    x_batch = np.array(train_img)[index]
    x_batch=read_images(x_batch)
    # seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    # ])
    if data_aug:
        # x_batch=seq.augment_images(x_batch)
        # print(x_batch)
        # print(x_batch.shape)
        x_batch=_random_flip_leftright(x_batch)
        x_batch=_random_noise(x_batch)
    y_batch =onehot(np.array(train_label)[index],out_dim)
    return x_batch,y_batch
def get_omega_batch(train_img,batch_size,out_dim,data_aug=True):
    index = np.random.choice(len(train_img), batch_size)
    # print(index)
    x_batch = np.array(train_img)[index]
    x_batch=read_images(x_batch)
    if data_aug:
        x_batch=_random_flip_leftright(x_batch)
        x_batch=_random_noise(x_batch)
    return x_batch