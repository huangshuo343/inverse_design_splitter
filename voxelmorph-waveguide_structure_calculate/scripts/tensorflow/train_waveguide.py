"""
Example script to train a VoxelMorph model.

For the CVPR and MICCAI papers, we have data arranged in train, validate, and test folders. Inside each folder
are normalized T1 volumes and segmentations in npz (numpy) format. You will have to customize this script slightly
to accommodate your own data. All images should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed. Otherwise,
registration will be scan-to-scan.
"""

import os
import random
import argparse
import glob
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import keras.backend as K
import tensorflow.config as tfcon

import sys

sys.path.append('C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design/voxelmorph'
                '-waveguide_structure_calculate/')
# print(sys.path)
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--datadir', help='base data directory')
parser.add_argument('--datalist', default='C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design'
                                          '/training_set_reduce_simulate_area_symmetric_separatechannel12.txt',
                    help='data list text file')
parser.add_argument('--num_feature', type=int, default=1, help='number of feature components used for registration, '
                                                               'specify that data has multiple channel features')
parser.add_argument('--atlas', help='atlas filename')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
parser.add_argument("--phase_encoding", type=str, default='AP', help="phase encoding direction")
# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=150, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=15, help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 0.00001)')
#0 5e-5, 100 5e-6 #0d: 0 5e-5, 150 5e-6, epoch: 0-100, 50, 50

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=1, help='flow downsample factor for integration (default: 1)')#parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='transmission', help='image reconstruction loss - can be mse or nccc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.0001, help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--kl-lambda', type=float, default=10, help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')
args = parser.parse_args()

#tfcon.set_soft_device_placement(True)

#args.load_weights = 'C:/Users/DELL/Desktop/2023年上半年学习和生活的文件/科研的文件/waveguide_structure_design/voxelmorph' \
#                    '-waveguide_structure_calculate/scripts/tensorflow/models/0250.h5'

# load and prepare training data
# train_vol_names = glob.glob(os.path.join(args.datadir, '*.npz'))
train_vol_names = np.loadtxt(args.datalist,dtype='str',ndmin=2)
# random.shuffle(train_vol_names)  # shuffle volume list
# assert len(train_vol_names) > 0, 'Could not find any training data'

# no need to append an extra feature axis if data is multichannel
# if args.num_feature == 1:
#     add_feat_axis = True
# else:
#     add_feat_axis = False
# print('add_feat_axis',add_feat_axis)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# padding step to fit the network
sampling_num = 2**len(enc_nf)
fixed = vxm.py.utils.load_volfile(train_vol_names[0,0], num_feature=args.num_feature)
vol_size = fixed.shape
fea_num_los = 1#2
fea_num_complex = 2
fea_num_complex_use = 1
fixed_fea = vxm.py.utils.load_volfile(train_vol_names[0,2], num_feature=args.num_feature)#num_feature=args.num_feature #args.num_feature
vol_size_fea = fixed_fea.shape
fixed_fea_complex = vxm.py.utils.load_volfile(train_vol_names[0,4], num_feature=fea_num_complex)#num_feature=args.num_feature #args.num_feature
vol_size_fea_complex = fixed_fea_complex.shape
padding_num = []
pad_dim = len(vol_size)-1
padding_num_fea = []
if args.num_feature > 1:
    # 4D volume
    new_vol_size = list(vol_size)
else:
    new_vol_size = list(vol_size[:-1])
    print('new_vol_size',new_vol_size)
for i in range(pad_dim):
    divid_val = int(vol_size[i])/sampling_num
    tmp = int((np.ceil(divid_val) - divid_val)*sampling_num)
    padding_num.append(tmp)
    new_vol_size[i] = vol_size[i] + tmp
# if fea_num_los > 1:
#     # 4D volume
#     new_vol_size_fea = list(vol_size_fea)
# else:
#     new_vol_size_fea = list(vol_size_fea[:-1])
#     print('new_vol_size_fea',new_vol_size_fea)
# for i in range(pad_dim):
#     divid_val_fea = int(vol_size_fea[i])/sampling_num
#     tmp_fea = int((np.ceil(divid_val_fea) - divid_val_fea)*sampling_num)
#     padding_num_fea.append(tmp_fea)
#     new_vol_size_fea[i] = vol_size_fea[i] + tmp_fea
#print('new_vol_size',new_vol_size)
#print('new_vol_size_fea',new_vol_size_fea)
print('vol_size',vol_size)
print('vol_size_fea',vol_size_fea)
print('vol_size',vol_size)
new_vol_size = vol_size[ : -1]#vol_size[ : -1]
new_vol_size_fea = vol_size_fea[ : -1]#vol_size_fea[ : -1]
new_vol_size_fea_complex = vol_size_fea_complex
#new_vol_size_fea_complex_aim = vol_size_fea_complex[ : -1]
new_vol_size_fea_complex_aim = [vol_size_fea_complex[0], vol_size_fea_complex[1], vol_size_fea_complex[2] * 2]

generator = vxm.generators.scan_to_scan_FOD(train_vol_names, pad_shape_dat=tuple(new_vol_size),pad_shape_fea=tuple(new_vol_size_fea),
                                            pad_shape_complex=tuple(new_vol_size_fea_complex),pad_shape_complex_aim=tuple(new_vol_size_fea_complex_aim),
                                            batch_size=args.batch_size,
                                            bidir=args.bidir, num_feature_data=args.num_feature,
                                            num_feature_feature=fea_num_los, num_feature_complex=fea_num_complex)

# extract shape and number of features from sampled input
sample_shape = next(generator)[0][0].shape
feature_as_target = next(generator)[0][2]
print('sample for convolution shape: ', sample_shape ,'.')
inshape_simulation = sample_shape[1:-1]
print('inshape simulation n map is: ', inshape_simulation, '.')
nfeats = 1#nfeats = sample_shape[-1]

sample_shape_for_error = next(generator)[0][2].shape
print('sample for calculating error shape: ', sample_shape_for_error ,'.')
inshape_viewfield = sample_shape[1:-1]
print('inshape simulation for calculating error is: ', inshape_viewfield, '.')

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow gpu handling
device = '/gpu:' + args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.keras.backend.set_session(tf.Session(config=config))

# ensure valid batch size given gpu count
nb_gpus = len(args.gpu.split(','))
assert np.mod(args.batch_size, nb_gpus) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (args.batch_size, nb_gpus)


# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

print('args.phase_encoding',args.phase_encoding)

with tf.device(device):

    # build the model
    model = vxm.networks.WaveguideOutputSimulate(
        inshape_simulation=inshape_simulation,
        inshape_viewfield=inshape_viewfield,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
        fea_feats=fea_num_complex_use,#fea_num_complex
        phase_encoding=args.phase_encoding#,
        #feature_data=feature_as_target
    )#model = vxm.networks.VxmDenseFeatSemiSupervised

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)

    # prepare image loss

    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE(args.image_sigma).loss
    elif args.image_loss == 'transmission':
        image_loss_func = vxm.losses.Transmit_percentage(args.image_sigma).loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc" or "transmission", but found "%s".' % args.image_loss)

    print('args.image_loss is:',args.image_loss)

    # image_loss_func = lambda _, y_p: K.mean(K.square(y_p))
    # need two image loss functions if bidirectional
    if args.bidir:
        losses  = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses  = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    if args.use_probs:
        flow_shape = model.outputs[-1].shape[1:-1]
        losses += [vxm.losses.KL(args.kl_lambda, flow_shape).loss]
    else:
        #losses += [vxm.losses.Grad('l2').loss]
        losses += [vxm.losses.Binary().loss]

    weights += [args.lambda_weight]
    #
    # multi-gpu support
    if nb_gpus > 1:
        save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
        model = tf.keras.utils.multi_gpu_model(model, gpus=nb_gpus)
    else:
        save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # Get the custom layer instance from the model
    custom_layer = None
    for layer in model.layers:
        if isinstance(layer, vxm.tf.layers.WaveguideValueDesignOutput):
            custom_layer = layer
            break

    #print(model.layers)

    # Ensure the custom layer is found
    if custom_layer is None:
        raise ValueError("Custom layer not found in the model.")

    # Create an instance of the custom callback with the custom layer
    change_parameter_callback = vxm.tf.layers.ChangeBinaryWeightCallback(change_interval=50, layer=custom_layer)
    #change_parameter_callback = vxm.tf.layers.ContinuousChangeBinaryWeightCallback(initial_weight=1, enlarge_rate = 1.015,
    #                                                                     change_learning_steps=150, layer=custom_layer)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))
    model.summary()
    model.fit_generator(generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[save_callback,change_parameter_callback],
        verbose=2
    )

    # save final model weights
    model.save(save_filename.format(epoch=args.epochs))
