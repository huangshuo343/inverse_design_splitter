"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    python register.py moving.nii.gz fixed.nii.gz --model model.h5 --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered, but if not, an initial linear registration
can be run by specifing an affine model with the --affine-model flag, which expects an affine model file as input.
If an affine model is provided, it's assumed that it was trained with images padded to (256, 256, 256) and resized
by a factor of 0.25 (the default affine model configuration).
"""

import os
import argparse
import numpy as np
import datetime

import sys

sys.path.append('C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design/voxelmorph'
                '-waveguide_structure_calculate/')

import voxelmorph as vxm
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--datalist', default='C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design'
                                          '/test_set_reduce_simulation_area.txt', help='data list text file')
parser.add_argument('--exp_name', default='C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design'
                                          '/register.log', help='experiment name for output dir')
parser.add_argument('--num_feature', type=int, default=1, help='number of feature components used for registration, '
                                                               'specify that data has multiple channel features')
# parser.add_argument('moving', help='moving image (source) filename')
# parser.add_argument('fixed', help='fixed image (target) filename')
parser.add_argument('--moved', help='registered image output filename')
parser.add_argument('--diff', help='difference image output filename')
parser.add_argument('--model', default='C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design'
                                       '/voxelmorph-waveguide_structure_calculate/scripts/tensorflow/'
                                       'models/0100 - 副本.h5',
                    help='run nonlinear registration - must specify keras model file')
parser.add_argument('--warp', help='output warp filename')
parser.add_argument('--affine-model', help='run intitial affine registration - must specify keras model file')
parser.add_argument('--affine', help='output affine filename (must be npz)')
parser.add_argument('-g', '--gpu', default = 0, help='GPU number(s) - if not supplied, CPU is used')
# parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# sanity check on the input
assert (args.model or args.affine_model), 'must provide at least a warp or affine model'

def read_complex_data(vol_name, **load_params_fea_complex):
    x_b_map_ima = vxm.py.utils.load_volfile(vol_name, **load_params_fea_complex)
    x_b_map_ima_shape = x_b_map_ima.shape
    #x_b_map_ima_shape[-1] = 1
    x_b_map_ima_complex = np.zeros((*x_b_map_ima_shape[ : -1], 1), dtype='complex_')
    x_b_map_ima_complex[0, : , : ,0] = x_b_map_ima[0, :, :, 0] + x_b_map_ima[0, :, :, 1] * 1j
    #x_b_map_ima_complex[0, :, :, 1] = x_b_map_ima[0, :, :, 0] + x_b_map_ima[0, :, :, 1] * 1j
    return x_b_map_ima_complex

# device handling
if args.gpu and (args.gpu != '-1'):
    device = '/gpu:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.keras.backend.set_session(tf.Session(config=config))
else:
    device = '/cpu:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
# add_feat_axis = not args.multichannel

# unet architecture
enc_nf = [16, 32, 32, 32]
# padding step to fit the network
sampling_num = 2**len(enc_nf)
train_vol_names = np.loadtxt(args.datalist,dtype='str',ndmin=2)
# fixed = vxm.py.utils.load_volfile(train_vol_names[0,0], add_feat_axis=add_feat_axis)
num_subject = len(train_vol_names)
for j in range(num_subject):
    # base_name = train_vol_names[j,0].split('data_after_topup')[0]
    # base_name = train_vol_names[j,0].split('data')[0]
    # output_dir = os.path.join(base_name,'eddy_unwarp',args.exp_name)
    #base_name = train_vol_names[j,0].split('Diffusion')[0]
    output_dir_name = train_vol_names[j,0].split('data')[2]#'data_train'
    output_dir_name_percentage = output_dir_name.split('/')[1]#output_dir_name[1:11]
    base_name = 'D:/data_ee604_final_project/results/test_data_reduce_simulation_area_wide_3size_100epoches_align_corner_smooth_spreatchannel12_learningrate_suit/'
    output_dir = os.path.join(base_name,output_dir_name_percentage)#'1_55_50_50_15epochstrain'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    warp_name = os.path.join(output_dir,'waveguide_structure_simulation_result.nii.gz')
    psi_tot_output_abs_name = os.path.join(output_dir, 'psi_tot_output_abs_image.nii.gz')
    diff_name = os.path.join(output_dir,'diff_image.nii.gz')
    jac_name = os.path.join(output_dir,'jacobian_determinant.nii.gz')
    std_txt = os.path.join(output_dir,'jd_std.txt')
    log_txt = os.path.join(output_dir,'jd_log.txt')

    if not os.path.exists(warp_name):
        a = datetime.datetime.now()
        # fixed = vxm.py.utils.load_volfile(train_vol_names[j,0], num_feature=args.num_feature)
        #
        # vol_size = fixed.shape
        # padding_num = []
        # pad_dim = len(vol_size)-1
        # if args.num_feature > 1:
        #     # 4D volume
        #     new_vol_size = list(vol_size)
        # else:
        #     new_vol_size = list(vol_size[:-1])
        #     print('new_vol_size',new_vol_size)
        # for i in range(pad_dim):
        #     divid_val = int(vol_size[i])/sampling_num
        #     tmp = int((np.ceil(divid_val) - divid_val)*sampling_num)
        #     padding_num.append(tmp)
        #     new_vol_size[i] = vol_size[i] + tmp
        fixed = vxm.py.utils.load_volfile(train_vol_names[0, 0], num_feature=args.num_feature)
        vol_size = fixed.shape
        fea_num_los = 1  # 2
        fea_num_complex = 2
        fea_num_complex_use = 1
        fixed_fea = vxm.py.utils.load_volfile(train_vol_names[0, 2],
                                              num_feature=args.num_feature)  # num_feature=args.num_feature #args.num_feature
        vol_size_fea = fixed_fea.shape

        fixed_fea_complex = vxm.py.utils.load_volfile(train_vol_names[0, 4],
                                              num_feature=args.num_feature)  # num_feature=args.num_feature #args.num_feature
        vol_size_fea_complex = fixed_fea_complex.shape

        # print('new_vol_size',new_vol_size)
        # print('vol_size',vol_size)
        # print('padding_num',padding_num)

        print('vol_size', vol_size)
        print('vol_size_fea', vol_size_fea)
        print('vol_size', vol_size)
        new_vol_size = vol_size[: -1]  # vol_size[ : -1]
        #new_vol_size_complex = vol_size[: -1]  # vol_size[ : -1]
        new_vol_size_fea = vol_size_fea[: -1]  # vol_size_fea[ : -1]
        new_vol_size_fea_complex = vol_size_fea_complex[: -1]
        #new_vol_size_fea_complex = vol_size_fea_complex

        forward_map_image = vxm.py.utils.load_volfile(train_vol_names[j,0], add_batch_axis=True,
                                           num_feature=args.num_feature,
                                           pad_shape=tuple(new_vol_size))
        inverse_map_image = vxm.py.utils.load_volfile(train_vol_names[j, 1], add_batch_axis=True,
                                                  num_feature=args.num_feature,
                                                  pad_shape=tuple(new_vol_size))

        #moving_features = vxm.py.utils.load_volfile(train_vol_names[j,1], add_batch_axis=True, num_feature=args.num_feature, pad_shape=tuple(new_vol_size)) #t1_data
        n_fix_map_image,fixed_affine  = vxm.py.utils.load_volfile(train_vol_names[j,2], add_batch_axis=True,
                                                        num_feature=args.num_feature,
                                                        pad_shape=tuple(new_vol_size),
                                                        ret_affine=True)## /
        simulation_region_map_image = vxm.py.utils.load_volfile(train_vol_names[j, 3], add_batch_axis=True,
                                                  num_feature=args.num_feature,
                                                  pad_shape=tuple(new_vol_size))
        #fixed, fixed_affine = vxm.py.utils.load_volfile(train_vol_names[j, 0], add_batch_axis=True,
                                                        #num_feature=args.num_feature, pad_shape=tuple(new_vol_size),
                                                        #ret_affine=True)
        load_params_fea_complex = dict(add_batch_axis=True, num_feature=fea_num_complex,
                                       pad_shape=tuple(new_vol_size_fea_complex))
        n_reference_b_map_image = read_complex_data(train_vol_names[j, 4], **load_params_fea_complex)
        #y_aim_map_image = read_complex_data(train_vol_names[j, 5], **load_params_fea_complex)

        offsets = [int((p - v) / 2) for p, v in zip(new_vol_size, vol_size)]
        slices  = tuple([slice(offset, l + offset) for offset, l in zip(offsets, vol_size)])
        if args.num_feature > 1:
            # take first 3 dimensions
            warp_slices = tuple(list(slices)[0:3])
        else:
            #warp_slices = slices + tuple([slice(0,3)])
            warp_slices = slices
        print('offsets', offsets)
        print('slices', slices)
        print('warp_slices', warp_slices)

        # padded[slices] = array

        if args.model:

            # moving = moving[np.newaxis]
            # fixed = fixed[np.newaxis]

            with tf.device(device):
                # load model and predict
                # warp = vxm.networks.DrC_net.load(args.model).register(moving, fixed)
                # net = vxm.networks.DrC_net.load(args.model)
                # net.summary()
                # warp, psi_tot_output_abs = vxm.networks.WaveguideOutputSimulate.load(args.model).predict([forward_map_image,
                #                                                                     inverse_map_image,
                #                                                                     n_fix_map_image,
                #                                                                     simulation_region_map_image,
                #                                                                     n_reference_b_map_image,
                #                                                                     y_aim_map_image])# #diff,warp = vxm.networks.DrC_net.load(args.model).predict([moving, t1_data, fixed]) diff, t1_data #, moving_features
                warp = vxm.networks.UnetWaveguideSimulate.load(args.model).predict(
                    [forward_map_image,
                     inverse_map_image,
                     n_fix_map_image,
                     simulation_region_map_image])
                # diff, warp = vxm.networks.DrC_net.load(args.model).predict([moving, moving])
                # moved = vxm.tf.utils.transform(moving, warp)
                #warp = warp.squeeze()
                #psi_tot_output_abs = psi_tot_output_abs.squeeze()
                # moved = moved.squeeze()
                #diff  = diff.squeeze()
                print(warp.shape)
                # print(moved.shape)
                #print(diff.shape)
                warp_unpad = warp[warp_slices]
                # psi_tot_output_abs_unpad = psi_tot_output_abs[warp_slices]
                # moved_unpad = moved[slices]
                #diff_unpad  = diff[slices]
                print(warp_unpad.shape)
                # print(moved_unpad.shape)

            # save warp
            vxm.py.utils.save_volfile(warp_unpad, warp_name, fixed_affine)#vxm.py.utils.save_volfile(warp_unpad * (-1), warp_name, fixed_affine) # * (-1)
            # vxm.py.utils.save_volfile(psi_tot_output_abs_unpad, psi_tot_output_abs_name, fixed_affine)
            #vxm.py.utils.save_volfile(diff_unpad, diff_name, fixed_affine)
            b = datetime.datetime.now()
            c= b-a
            print('The time elapsed for registration is', c.total_seconds(), 'seconds.')

            # calculate jacobian determinant
            #jd = vxm.py.utils.jacobian_determinant(warp_unpad)
            #vxm.py.utils.save_volfile(jd, jac_name, fixed_affine)
            # calculate std of jacobian determinant
            #pxstatisticsonimage = '/ifs/loni/faculty/shi/spectrum/yqiao/tools/ITKTools/bin/bin/pxstatisticsonimage '
            #mask = os.path.join(base_name,'Diffusion','eddy_unwarp','data_after_topup', 'Pos_RAS_Mask_mask.nii.gz')
            #cmdStr = pxstatisticsonimage + '-in ' + jac_name + ' -mask ' + mask  + ' -s arithmetic > ' + log_txt
            #if not os.path.exists(log_txt):
                #os.system(cmdStr)
            ## extract std
            #with open(std_txt,'w') as fout:
                #with open(log_txt,'r') as fin:
                    #for line in fin:
                        #if 'arithmetic stdev' in line:
                            #std = line.split('arithmetic stdev: ')[1]
                            #fout.write(std)
