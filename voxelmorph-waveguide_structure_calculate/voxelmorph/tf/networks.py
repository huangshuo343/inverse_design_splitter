import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import tensorflow.linalg as tensorflow_linalg
from tensorflow.python.ops import io_ops
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections.abc import Iterable

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI
import tensorflow.keras as keras
from tensorflow.python.framework import ops

from .. import default_unet_features
from . import layers
from . import neuron as ne
from .modelio import LoadableModel, store_config_args
from .utils import gaussian_blur, value_at_location, point_spatial_transformer

# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel


def get_feature_layer(y_feature):
    return y_feature


def get_diff_data(y_source, y_target):
    y_diff = ne.layers.SpatialDiff(name='y_diff')([y_source, y_target])
    return y_diff


class UnetWaveguideSimulate(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape=None,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 inshape_simulation=None,
                 inshape_viewfield=None,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=2,
                 trg_feats=2,
                 fea_feats=1,
                 input_model=None,
                 phase_encoding='RL',
                 feature_data=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
            phase_encoding: phase encoding direction, default is RL.
        """

        # ensure correct dimensionality
        if inshape is None:
            inshape = [48, 48]
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        fea_feats = trg_feats
        if input_model is None:
            # configure default input layers if an input model is not provided
            forward_map = tf.keras.Input(shape=(*inshape, src_feats), name='forward_map_input')
            inverse_map = tf.keras.Input(shape=(*inshape, src_feats), name='inverse_map_input')
            n_fix = tf.keras.Input(shape=(*inshape, trg_feats), name='waveguide_fix_input')  # source_quality_value
            simulation_region = tf.keras.Input(shape=(*inshape, trg_feats), name='simulation_region_input')
            # simulation_region = tf.keras.Input(shape=inshape, name='simulation_region_input')
            input_model = tf.keras.Model(inputs=[forward_map, inverse_map, n_fix, simulation_region],
                                         outputs=[forward_map, inverse_map, n_fix,
                                                  simulation_region])  # tensor先从GPU挪到CPU上，去掉前面两项，分别是batch shize和channel通道数，得到3维的体数据，再转换成numpy，再用itk-snap这个库，转换成图像。 # # / # image_t1,

        else:
            forward_map, inverse_map, n_fix, simulation_region = input_model.outputs[:4]
            # source, target = input_model.outputs[:2]#修改成只用初始时的前两行。

        u_net_model_input = tf.keras.Model(inputs=input_model.inputs,
                                           outputs=input_model.outputs[:2])
        print('Simulation area\'s shape is: ', forward_map.shape, '.')
        # build core unet model and grab inputs
        unet_model = Unet(
            input_model=u_net_model_input,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        n_pred_allresults_1size = Conv(kernel_size=3, filters=3, strides=[1, 1], use_bias=True, padding='same',
                                 kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                                 #activation='relu',
                                 name='waveguide_predict_allresults_1size')(
            unet_model.output)  # ndims, / https://blog.csdn.net/cjhxydream/article/details/114492870
        n_pred_allresults = Conv(kernel_size=3, filters=1, strides=[3, 3], use_bias=True, padding='same',
                                 kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                                 #activation='relu',
                                 name='waveguide_predict_allresults')(
            n_pred_allresults_1size)
        print('The shape of n_pred_alresults is: ', n_pred_allresults.shape, '.')

        n_pred = layers.WaveguideValueDesignOutput(name = 'waveguide_predict_result_output')([n_pred_allresults,
                                                                                              simulation_region,
                                                                                              n_fix])

        outputs = [n_pred]  # neg_flow #y_source #[source_b0, flow_mean] source_b0, #

        # del target
        super().__init__(name='waveguide_net', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        # self.references.y_source = y_source#不能每次变化。 #y_source source
        # self.references.y_target = y_target#y_target #source target /
        self.references.forward_map = forward_map
        self.references.inverse_map = inverse_map
        self.references.n_fix = n_fix
        self.references.simulation_region = simulation_region
        self.references.n_pred_allresults = n_pred_allresults
        self.references.n_pred = n_pred

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class WaveguideOutputSimulate(LoadableModel):
    """
    VoxelMorph network for (semi-supervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape_simulation,
                 inshape_viewfield,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=1,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 fea_feats=1,
                 input_model=None,
                 phase_encoding='RL',
                 feature_data=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_labels: Number of labels used for ground truth segmentations.
            nb_unet_features: Unet convolutional features. See VxmDense documentation for more information.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            seg_downsize: Interger specifying the downsampled factor of the segmentations. Default is 2.
        """

        # configure base voxelmorph network
        waveguide_net_model = UnetWaveguideSimulate(
            inshape=inshape_simulation,
            nb_unet_features=nb_unet_features,
            bidir=bidir,
            use_probs=use_probs,
            int_steps=int_steps,
            int_downsize=int_downsize,
            src_feats=src_feats,
            trg_feats=trg_feats,
            phase_encoding=phase_encoding)  # ,
        # feature_data=feature_data

        x_b = tf.keras.Input(shape=(*inshape_viewfield, fea_feats), dtype=tf.complex64, name='input_signal_b_input')
        y_aim = tf.keras.Input(shape=(*inshape_viewfield, fea_feats * 2), dtype=tf.complex64,
                               name='output_signal_aim_input')

        error_value = layers.WaveguideOutputSimulation(name='output_error_value')(
            [waveguide_net_model.references.n_pred, waveguide_net_model.references.n_pred_allresults,
             x_b, y_aim, waveguide_net_model.references.simulation_region])

        inputs = waveguide_net_model.inputs + [x_b, y_aim]

        outputs = [error_value, waveguide_net_model.references.n_pred]

        super().__init__(inputs=inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.x_b = x_b
        self.references.y_aim = y_aim
        # self.references.simulation_error = simulation_error#
        self.references.error_value = error_value

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


def conv_block(x, nfeat, strides=1, name=None):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    Conv = getattr(KL, 'Conv%dD' % ndims)

    convolved = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(
        x)
    print('Conv%dD' % ndims, 'convolved x shape', convolved.shape)
    name = name + '_activation' if name else None
    return KL.LeakyReLU(0.2, name=name)(convolved)


def upsample_block(x, connection, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    upsampled = UpSampling(name=name)(x)
    print('Conv%dD' % ndims, 'upsampled x shape', upsampled.shape)
    name = name + '_concat' if name else None
    return KL.concatenate([upsampled, connection], name=name)


class Unet(tf.keras.Model):
    """
    A unet architecture that builds off of an input keras model. Layer features can be specified directly
    as a list of encoder and decoder features or as a single integer along with a number of unet levels.
    The default network features per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self, input_model, nb_features=None, nb_levels=None, feat_mult=1, nb_conv_per_level=1):
        """
        Parameters:
            input_model: Input model that feeds directly into the unet before concatenation.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
        """

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        # configure encoder (down-sampling path)
        enc_layers = [KL.concatenate(input_model.outputs, name='unet_input_concat')]
        last = enc_layers[0]
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                strides = 2 if conv == (nb_conv_per_level - 1) else 1
                name = 'unet_enc_conv_%d_%d' % (level, conv)
                last = conv_block(last, nf, strides=strides, name=name)
            enc_layers.append(last)

        # configure decoder (up-sampling path)
        last = enc_layers.pop()
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                name = 'unet_dec_conv_%d_%d' % (real_level, conv)
                last = conv_block(last, nf, name=name)
            name = 'unet_dec_upsample_' + str(real_level)
            last = upsample_block(last, enc_layers.pop(), name=name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            name = 'unet_dec_final_conv_' + str(num)
            last = conv_block(last, nf, name=name)

        return super().__init__(inputs=input_model.inputs, outputs=last)
