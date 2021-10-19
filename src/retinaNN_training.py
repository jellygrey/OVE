###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from lib.help_py import *

#function to obtain data for training/testing (validation)
from lib.extract_patches import get_data_training


# def set_GPU():
#     """GPU相关设置"""
#
#     # 打印变量在那个设备上
#     # tf.debugging.set_log_device_placement(True)
#     # 获取物理GPU个数
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     print('物理GPU个数为：', len(gpus))
#     # 设置内存自增长
#     # for gpu in gpus:
#     #     tf.config.experimental.set_memory_growth(gpu, True)
#     # print('-------------已设置完GPU内存自增长--------------')
#
#     # 设置哪个GPU对设备可见，即指定用哪个GPU
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     # 切分逻辑GPU
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],  # 指定要切割的物理GPU
#         # 切割的个数，和每块逻辑GPU的大小
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
#          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096), ]
#     )#最后一块物理GPU切分成两块，现在逻辑GPU个数为2
#     # 获取逻辑GPU个数
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print('逻辑GPU个数为：', len(logical_gpus))
#
# set_GPU()


#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
print(path_data)
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)
# print(patches_imgs_train)
print(patches_imgs_train.shape)
# print(patches_masks_train)
print(patches_masks_train.shape)

# patches_imgs_train=patches_imgs_train.reshape(190000,48,48,1)
# patches_masks_train=patches_masks_train.reshape(190000,48,48,1)

import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_last')
temp_x_train = []
bbb=[]
# for i in range(len(patches_imgs_train)):
#     new_x_train_row=np.moveaxis(patches_imgs_train[i],0,2)
#     temp_x_train.append(new_x_train_row)
#
#     new_bbb=np.moveaxis(patches_masks_train[i],0,2)
# #     bbb.append(new_bbb)
# patches_imgs_train = np.array(temp_x_train)
# patches_masks_train=np.array(bbb)


# print(patches_imgs_train)
print(patches_imgs_train.shape)
# print(patches_masks_train)
print(patches_masks_train.shape)

#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./DRIVE/'+name_experiment+'/'+"sample_input_imgs")#.show()
#sample_input_imgs.png
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./DRIVE/'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the     model arcitecture =====
n_ch = patches_imgs_train.shape[3]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[1]
model = get_unet(n_ch,patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
print(model.output_shape)
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./DRIVE/'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
print(patches_imgs_train.shape)
print(patches_masks_train.shape)
patches_imgs_train=tf.reshape(shape=(None,48,48,1))
model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
