import os
import math
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from natsort import natsorted
import matplotlib.pyplot as plt


def print_aff(data_path):
    proxy_img = nib.load(data_path)
    print(proxy_img.affine)

    return

def make_path_list(dir):

    pathlist = []
    for root, _, fnames in natsorted(os.walk(dir)):
        for fname in natsorted(fnames):
            # if fname == filename:
            path = os.path.join(root, fname)
            pathlist.append(path)
    pathlist = np.asarray(pathlist)

    return pathlist

def normalize(img):
    max = np.max(img)
    min = np.min(img)
    normalized_img = (img-min)/(max-min)

    return normalized_img

def rescale(vol,scale):
    # The MRI dataset shape is w > h = d, so make rescaled mri isotropic
    h,w,d = vol.shape
    vol_rs = zoom(vol,zoom=(scale,scale*float(h)/w,scale),mode='nearest')
    return vol_rs

def get_mri(data_path):
    proxy_img = nib.load(data_path)
    data_array = np.asarray(proxy_img.dataobj)
    # data_array = rescale(data_array, scale)
    data_array = normalize(data_array)
    return data_array

def print_aff(data_path):
    proxy_img = nib.load(data_path)
    print(proxy_img.affine)

    return

def get_aff(dir, fname):
    f_list = make_path_list(dir, fname)
    proxy_img = nib.load(f_list[0])
    return proxy_img.affine

def preprocessing(data_dir, save_dir, fname, scale):

    print("Preprocessing Start")
    f_list = make_path_list(data_dir,fname)
    concat_mri = [get_mri(path,scale) for path in f_list]
    concat_mri = np.asarray(concat_mri,dtype=np.float32)
    np.save(save_dir,concat_mri)
    print("Concatenation Done")
    return

def rolling_window_lastaxis(a, window):
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a

def find_str(list_str):
    if "AD_0" in list_str:
        return 0

    elif "NC_P" in list_str:
        return 2

    elif "NC_0" in list_str:
        return 1

    elif "PD_P" in list_str:
        return 3

def Voxel_extrac(data_path, save_dir, lbl_dir, dataName):
    window = (33, 33, 33)
    ad = 0
    nc = 0
    pnc = 0
    ppd = 0
    dataShape = []

    # nad = 0
    # nnc = 0
    # npnc = 0
    # nppd = 0
    # nonz_num = np.zeros([1, 1])

    # for dd, ddata in enumerate(dataName):
    #     diseaseNum = find_str(ddata)
    #     MRI_image = get_mri(ddata)
    #     MRI_SW_image = rolling_window(MRI_image, window)
    #     dSize = math.floor(MRI_SW_image.shape[0] / 10)
    #
    #     coordinate = [(x, y, z) for x in range(10, MRI_SW_image.shape[0], 10) for y in
    #                   range(10, MRI_SW_image.shape[1], 10) for z in range(10, MRI_SW_image.shape[2], 10)]
    #
    #
    #     if diseaseNum == 0:
    #         if nad == 0:
    #             dataShape.append((20, dSize ** 3, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
    #
    #             nonz = 0
    #             for c, coord in enumerate(coordinate):
    #                 if sum(sum(sum(MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]))) != 0:
    #                     nonz_num[nad,nonz] = c
    #                     nonz +=1
    #         nad += 1
    #     elif diseaseNum == 1:
    #
    #         if nc == 0:
    #             dataShape.append((20, dSize ** 3, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
    #             NC_data = np.memmap(filename=save_dir + '/NC_data.dat', dtype=np.float32, mode="w+", shape=dataShape[1])
    #             nonz=0
    #             coordinate = [(x, y, z) for x in range(10, MRI_SW_image.shape[0], 10) for y in
    #                           range(10, MRI_SW_image.shape[1], 10) for z in range(10, MRI_SW_image.shape[2], 10)]
    #             for c, coord in enumerate(coordinate):
    #                 if not MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]:
    #                     nonz_num[nnc,nonz] = c
    #                     nonz +=1
    #         nc += 1
    #     elif diseaseNum == 2:
    #
    #         if pnc == 0:
    #             dataShape.append((11, dSize ** 3, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
    #             PNC_data = np.memmap(filename=save_dir + '/PNC_data.dat', dtype=np.float32, mode="w+", shape=dataShape[2])
    #
    #             coordinate = [(x, y, z) for x in range(10, MRI_SW_image.shape[0], 10) for y in
    #                           range(10, MRI_SW_image.shape[1], 10) for z in range(10, MRI_SW_image.shape[2], 10)]
    #             if not MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]:
    #                 nonz_num[npnc, nonz] = c
    #                 nonz += 1
    #         pnc += 1
    #     elif diseaseNum == 3:
    #
    #         if ppd == 0:
    #             dataShape.append((11, dSize ** 3, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
    #             PPD_data = np.memmap(filename=save_dir + '/PPD_data.dat', dtype=np.float32, mode="w+", shape=dataShape[3])
    #
    #             coordinate = [(x, y, z) for x in range(10, MRI_SW_image.shape[0], 10) for y in
    #                           range(10, MRI_SW_image.shape[1], 10) for z in range(10, MRI_SW_image.shape[2], 10)]
    #             if not MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]:
    #                 nonz_num[nppd, nonz] = c
    #                 nonz += 1
    #         ppd += 1

    for d, data in enumerate(dataName):

        diseaseNum = find_str(data)
        MRI_image = get_mri(data)
        MRI_SW_image = rolling_window(MRI_image, window)
        idx, coordinate = find_non_Zeros(MRI_SW_image)
        dSize = len(idx)
        if diseaseNum == 0:

            dataShape.append((dSize, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
            AD_data = np.memmap(filename=(save_dir + '/AAD_data_%d.dat') % ad, dtype=np.float32, mode="w+", shape=dataShape[d])
            AD_label = np.memmap(filename=(lbl_dir + '/AAD_data_%d.lbl') % ad, dtype=np.uint8, mode="w+", shape=(1, 1, dSize))

            for c, coord in enumerate(coordinate):
                AD_data[c, :, :, :] = MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]
            AD_label[:, :, :] = np.ones(shape=(1, 1, dSize), dtype=np.uint8)
            print('Make Test AD %d' % AD_label[0, 0, -1])

            ad += 1
        elif diseaseNum == 1:

            dataShape.append((dSize, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
            NC_data = np.memmap(filename=(save_dir + '/ANC_data_%d.dat') % nc, dtype=np.float32, mode="w+", shape=dataShape[d])
            NC_label = np.memmap(filename=(lbl_dir + '/ANC_data_%d.lbl') % nc, dtype=np.uint8, mode="w+", shape=(1, 1, dSize))

            for c, coord in enumerate(coordinate):
                NC_data[c, :, :, :] = MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]
            NC_label[:, :, :] = np.zeros(shape=(1, 1, dSize), dtype=np.uint8)
            print('Make Test NC %d' % NC_label[0, 0, -1])
            nc += 1
        elif diseaseNum == 2:

            dataShape.append((dSize, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
            PNC_data = np.memmap(filename=(save_dir + '/PNC_data_%d.dat') % pnc, dtype=np.float32, mode="w+", shape=dataShape[d])
            PNC_label = np.memmap(filename=(lbl_dir + '/PNC_data_%d.lbl') % pnc, dtype=np.uint8, mode="w+", shape=(1, 1, dSize))

            for c, coord in enumerate(coordinate):
                PNC_data[c, :, :, :] = MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]
            PNC_label[:, :, :] = np.zeros(shape=(1, 1, dSize), dtype=np.uint8)
            print('Make Test PNC %d' % PNC_label[0, 0, -1])

            pnc += 1
        elif diseaseNum == 3:

            dataShape.append((dSize, MRI_SW_image.shape[3], MRI_SW_image.shape[4], MRI_SW_image.shape[5]))
            PPD_data = np.memmap(filename=(save_dir + '/PPD_data_%d.dat') % ppd, dtype=np.float32, mode="w+", shape=dataShape[d])
            PPD_label = np.memmap(filename=(lbl_dir + '/PPD_data_%d.lbl') % ppd, dtype=np.uint8, mode="w+", shape=(1, 1, dSize))

            for c, coord in enumerate(coordinate):
                PPD_data[c, :, :, :] = MRI_SW_image[coord[0], coord[1], coord[2], :, :, :]
            PPD_label[:, :, :] = np.ones(shape=(1, 1, dSize), dtype=np.uint8)
            print('Make Test PPD %d' % PPD_label[0, 0, -1])
            ppd += 1

        print('# data: %d, # disease: %d, # AD: %d, # NC: %d, # PNC: %d, # PPD: %d' % (d, diseaseNum, ad, nc, pnc, ppd))

    return dataShape

def find_non_Zeros(data):
    coordinate = [(x, y, z) for x in range(0, data.shape[0], 10)
                    for y in range(0, data.shape[1], 10) for z in range(0, data.shape[2], 10)]
    idx = []
    coord = []
    for c, coo in enumerate(coordinate):
        if sum(sum(sum(data[coo[0], coo[1], coo[2], :, :, :]))) != 0:
            idx.append(c)
            coord.append(coo)

    return idx, coord
