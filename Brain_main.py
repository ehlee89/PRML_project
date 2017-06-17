import os
from datetime import datetime
import numpy as np
import Brain_def
import math
import model_3D

data_path = '/home/eh/Desktop/Data_PPMI/PRML_pro'
exp_dir = '/home/eh/Desktop/Data_PPMI/Experi_DAT'
save_dir = '/home/eh/Desktop/Data_PPMI/Extract_DAT'
lbl_dir = '/home/eh/Desktop/Data_PPMI/Extract_LBL'
info_dir = '/home/eh/Desktop/Data_PPMI/Extract_INFO'
output_dir = '/home/eh/Desktop/Data_PPMI/AP_CNN_result'

# Temp_dir = '/home/eh/Desktop/Data_PPMI/Temp_testD'

dataName = Brain_def.make_path_list(dir=data_path)

shape_path = info_dir + "/dataShape.npy"
if not os.path.exists(shape_path):
    dataShape = Brain_def.Voxel_extrac(data_path, save_dir, lbl_dir, dataName)
    np.save(info_dir + "/dataShape.npy", dataShape)
else:
    dataShape = np.load(info_dir + "/dataShape.npy")

# dataShape = Brain_def.Voxel_extrac(data_path, save_dir, lbl_dir, dataName)

CNN_Train_Flag = True

if CNN_Train_Flag:
    patch_name = Brain_def.make_path_list(dir=save_dir)
    label_name = Brain_def.make_path_list(dir=lbl_dir)
    model_def = model_3D.model_def()
    model_execute = model_3D.model_execute(data_path=data_path, patch_path=save_dir, label_path=lbl_dir, exp_path=exp_dir,
                                           output_path=output_dir, patch_name=patch_name, label_name=label_name, data_shape=dataShape)

    trainFlag_path = exp_dir + "/testLabel.lbl"

    if not os.path.exists(trainFlag_path):
        model_execute.mk_3D_voxel_data()

    # if not os.path.exists(trainFlag_path):
        # model_execute.mk_3D_voxel_data()

    # cross_entropy, softmax, layers, data_node, lbl_node = model_def.AP_CNN(train=True)
    # model_execute.train_AP_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node, label_node=lbl_node)

    cross_entropy, softmax, layers, data_node, lbl_node = model_def.AP_CNN(train=False)
    model_execute.test_AP_CNN(softmax=softmax, data_node=data_node)

    model_execute.code_test()