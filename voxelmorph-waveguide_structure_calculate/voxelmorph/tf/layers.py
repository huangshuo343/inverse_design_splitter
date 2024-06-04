import numpy as np
import neuron as ne
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as tensorflow_keras_layers
import tensorflow.linalg as tensorflow_linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from tensorflow.python.framework import ops
from tensorflow import sparse as tensorflow_sparse
import time

from .utils import is_affine, extract_affine_ndims, affine_shift_to_identity, affine_identity_to_shift

# make the following neuron layers directly available from vxm
SpatialTransformer = ne.layers.SpatialTransformer
LocalParam = ne.layers.LocalParam


class ChangeBinaryWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, change_interval=50, layer=None, **kwargs):
        super(ChangeBinaryWeightCallback, self).__init__(**kwargs)
        self.change_interval = change_interval
        self.layer = layer

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.change_interval == 0:
            binary_weight = self.change_parameter(epoch)
            print(f'Epoch {epoch}: Changing layer parameter to {binary_weight}')
            self.layer.update_parameter(binary_weight)

    def change_parameter(self, epoch):
        # Define how you want to change the parameter
        # For example, let's change the parameter value
        all_weights = np.array([5.0, 25.0, 50.0, 100.0, 150.0], dtype = float)
        #new_parameter_value = 1.0 * (0.1 ** (epoch // self.change_interval))
        binary_weight = all_weights[int(epoch//self.change_interval)]
        return binary_weight


class ContinuousChangeBinaryWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_weight=5, enlarge_rate = 1.015, initial_learning = 5e-5, change_learning_steps=150, layer=None, **kwargs):
        super(ContinuousChangeBinaryWeightCallback, self).__init__(**kwargs)
        self.initial_weight = initial_weight
        self.enlarge_rate = enlarge_rate
        self.initial_learning = initial_learning
        self.change_learning_steps = change_learning_steps
        self.layer = layer

    def on_epoch_begin(self, epoch, logs=None):
        #if epoch % self.change_interval == 0:
        binary_weight, learning_rate_newvalue = self.change_parameter(epoch)
        print(f'Epoch {epoch}: Changing layer parameter to {binary_weight} and learning rate to {learning_rate_newvalue}')
        self.layer.update_parameter(binary_weight)

    def change_parameter(self, epoch):
        # Define how you want to change the parameter
        # For example, let's change the parameter value
        #all_weights = np.array([5, 25, 50, 100, 150], dtype = float)
        #new_parameter_value = 1.0 * (0.1 ** (epoch // self.change_interval))
        #binary_weight = all_weights[int(epoch//self.change_interval)]
        binary_weight =  self.initial_weight * (self.enlarge_rate ** epoch)
        #learning_rate_newvalue = self.initial_learning
        #if epoch == self.change_learning_interval:
        #    learning_rate_newvalue = self.initial_learning / 10
        #    tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate_newvalue)
        #if epoch > self.change_learning_interval:
        #    learning_rate_newvalue = self.initial_learning / 10
        learning_rate_newvalue = self.initial_learning * (0.1 ** float(epoch // self.change_learning_steps))
        #print(epoch // self.change_learning_steps)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate_newvalue)
        #time.sleep(3)
        return binary_weight, learning_rate_newvalue


class WaveguideValueDesignOutput(Layer):
    '''
    Acquire the differences using the attention method by multiply feature data with attention weight.
    '''

    def __init__(self, **kwards):
        super(WaveguideValueDesignOutput, self).__init__(**kwards)
        #self.binary_weight = binary_weight
        self.binary_weight = tf.Variable(5.0, trainable=False)

    def build(self, input_shape):
        super(WaveguideValueDesignOutput, self).build(input_shape)

    def call(self, inputs):
        n_predict_allresults, simulation_region, n_fix = inputs[0], inputs[1], inputs[2]
        scale_factor = 3
        image_actual_shape = n_predict_allresults.shape

        one_data = tf.ones_like(n_fix)
        zero_data = tf.zeros_like(n_fix)
        n_predict_allresults = (n_predict_allresults - tf.reduce_min(n_predict_allresults)) / \
                               (tf.reduce_max(n_predict_allresults) - tf.reduce_min(n_predict_allresults))

        n_predict_allresults = (n_predict_allresults - 0.5) * self.binary_weight
                                                                 #5 25 50 100 150

        n_predict_upsample = tf.image.resize_images(n_predict_allresults,
                                                    [image_actual_shape[1] * scale_factor,
                                                     image_actual_shape[2] * scale_factor],
                                                    method=tf.image.ResizeMethod.BILINEAR,
                                                    align_corners=True)
        n_predict_sigmoid = tf.sigmoid(n_predict_upsample)
        n_predict_sigmoid = (n_predict_sigmoid - tf.reduce_min(n_predict_sigmoid)) / \
                            (tf.reduce_max(n_predict_sigmoid) - tf.reduce_min(n_predict_sigmoid))
        n_predict_sigmoid = (n_predict_sigmoid - 1) * 2.5
        # n_predict_allresults = (tf.sigmoid(n_predict_allresults) - 1) * 2.5
        n_predict = n_predict_sigmoid * simulation_region + n_fix
        n_predict = tf.where(n_predict <= 1.0, one_data, n_predict)
        n_predict = tf.where(n_predict >= 3.5, one_data * 3.5, n_predict)

        return n_predict

    def update_parameter(self, new_value):
        tf.keras.backend.set_value(self.binary_weight, new_value)

    def compute_output_shape(selfself, input_shape):
        return input_shape


class WaveguideOutputSimulation(Layer):
    '''
    Acquire the differences using the attention method by multiply feature data with attention weight.
    '''

    def __init__(self, **kwards):
        super(WaveguideOutputSimulation, self).__init__(**kwards)

    def build(self, input_shape):
        super(WaveguideOutputSimulation, self).build(input_shape)

    def call(self, inputs):
        [n_prediction, n_predict_allresults, x_b, y_aim, simulation_region] = inputs[0], inputs[1], inputs[2], \
                                                                              inputs[3], inputs[4]

        nx = 48  # 162
        ny = 48  # 162
        k0dx = 0.2026833935483871
        n_PML = 8
        regulation_connection_factor = 1.0 / (nx * ny * 4 * 10)  # Divided by 4 is to make the maximum value
        # be less than 4.

        A1 = self.build_laplacian_2d_PML(nx, ny, n_PML, n_PML, n_PML, n_PML)
        A1_tensor = tf.convert_to_tensor(A1.todense(dtype='complex64'), name='A1_tensor', dtype='complex64')
        n_pred_complex64 = tf.cast(n_prediction, dtype=tf.complex64,
                                   name='n_pred_tensor_complex64')
        print('shape of n_pred_complex64 is: ', n_pred_complex64.shape, '.')
        n_tensor_transpose = tf.transpose(n_pred_complex64, conjugate=False, perm=[0, 2, 1, 3],
                                          name='n_tensor_transpose')
        n_tensor_reshape = tf.reshape(n_tensor_transpose, [-1], name='n_tensor_reshape_flatten')
        A2 = tensorflow_linalg.diag(k0dx ** 2 * n_tensor_reshape ** 2, name='A2_tensor', padding_value=0,
                                    num_rows=nx * ny,
                                    num_cols=nx * ny)
        A = tf.add(A1_tensor, A2, name='A_tensor')
        # A = self.build_laplacian_2d_PML(nx, ny, n_PML, n_PML, n_PML, n_PML) + tf.matrix_diag(
        #     k0dx ** 2 * n_fix_num.flatten('F') ** 2, 0, nx * ny, nx * ny)
        x_b_complex64 = tf.cast(x_b, dtype=tf.complex64, name='x_b_tensor_complex64')
        b_tensor_transpose = tf.transpose(x_b_complex64, conjugate=False, perm=[0, 2, 1, 3], name='b_tensor_transpose')
        b_tensor_reshape = tf.reshape(b_tensor_transpose, [nx * ny, 1], name='b_tensor_reshape_flatten')
        psi_tot_data_reshape = tf.reshape(tensorflow_linalg.solve(A, b_tensor_reshape, name='psi_solve'),
                                          [1, 48, 48, 1], name='psi_out_reshape')
        psi_tot = tf.transpose(psi_tot_data_reshape, conjugate=False, perm=[0, 2, 1, 3], name='psi_tot_data_transpose')

        error_value_output_channel1 = tf.multiply(psi_tot, tf.reshape(y_aim[:, :, :, 0], [1, 48, 48, 1]),
                                                  name='error_value_transmission_channel1')  #
        error_value_output_channel2 = tf.multiply(psi_tot, tf.reshape(y_aim[:, :, :, 1], [1, 48, 48, 1]),
                                                  name='error_value_transmission_channel2')  #
        energy_value_input_channel1 = tf.reshape(tf.multiply(y_aim[:, :, :, 0], y_aim[:, :, :, 0]),
                                                 [1, 48, 48, 1])
        energy_value_input_channel2 = tf.reshape(tf.multiply(y_aim[:, :, :, 1], y_aim[:, :, :, 1]),
                                                 [1, 48, 48, 1])

        print("The shape of error_value_output_channel1 is: ", error_value_output_channel1.shape, ".")
        print("The shape of error_value_output_channel2 is: ", error_value_output_channel2.shape, ".")

        error_value_output = KL.concatenate([error_value_output_channel1, error_value_output_channel2,
                                                   energy_value_input_channel1, energy_value_input_channel2])

        return error_value_output

    def fun_s(self, u):
        v = []
        pi = 3.1415926
        power_kappa = 3
        kappa_max = 15
        power_sigma = 3
        lambda_over_dx = 40  # use a common choice for lambda/dx/n_bg
        sigma_over_omega_max = 0.8 * (power_sigma + 1) * lambda_over_dx / (2 * pi)
        kappa = lambda u: 1 + (kappa_max - 1) * u ** power_kappa
        sigma_over_omega = lambda u: sigma_over_omega_max * u ** power_sigma
        for i in range(len(u)):
            # print([i, u[i]])
            v.append(complex(kappa(u[i]), sigma_over_omega(u[i])))
        return v

    def build_laplacian_1d_PML(self, N, N_PML_L, N_PML_R):
        # Builds dx^2 times the Laplacian in 1D with effective outgoing boundary
        # condition implemented using PML on the two sides.
        # N = number of grid points
        # N_PML_L = number of PML pixels on the left; zero means no PML
        # N_PML_R = number of PML pixels on the right; zero means no PML

        # f = [f(1), ...., f(N)]
        # df = [df(0.5), ..., df(N+0.5)]
        # d2f = [d2f(1), ...., d2f(N)]
        # put Dirichlet boundary condition behind PML: f(0) = f(N+1) = 0
        # ddx_1*f = df
        # This is dx times (d/dx) operator
        pi = 3.1415926
        ddx_1 = sparse.spdiags([np.ones(N), -np.ones(N)], [0, -1], N + 1, N)

        # ddx_2*df = d2f
        # This is dx times (d/dx) operator
        ddx_2 = -np.transpose(ddx_1)

        if N_PML_L == 0 and N_PML_R == 0:
            A = ddx_2 * ddx_1
            return

        # define coordinate-stretching profile s(u)
        power_kappa = 3
        kappa_max = 15
        power_sigma = 3
        lambda_over_dx = 40  # use a common choice for lambda/dx/n_bg
        sigma_over_omega_max = 0.8 * (power_sigma + 1) * lambda_over_dx / (2 * pi)
        # kappa = lambda u: 1 + (kappa_max-1)*u**power_kappa;
        # sigma_over_omega = lambda u: sigma_over_omega_max*u**power_sigma;
        # fun_s = lambda u: complex(kappa(u), sigma_over_omega(u));

        # assign s(u) on interger and half-interger grid points
        # u = 0 on the integer site before PML
        # u = 1 on the integer site after PML (where we put Dirichlet BC)
        s_half = np.array([1 + 0j for i in range(N + 1)])  # column vector
        s_int = np.array([1 + 0j for i in range(N)])  # column vector
        if N_PML_R > 0:
            s_half[(N - N_PML_R):(N + 1)] = np.transpose(
                self.fun_s(np.arange(0.5, (N_PML_R + 1), 1) / (N_PML_R + 1)))
            s_int[(N - N_PML_R):N] = np.transpose(self.fun_s(np.arange(1, N_PML_R + 1, 1) / (N_PML_R + 1)))
        if N_PML_L > 0:
            s_half[(N_PML_L)::-1] = np.transpose(self.fun_s(np.arange(0.5, (N_PML_L + 1), 1) / (N_PML_L + 1)))
            s_int[N_PML_L - 1::-1] = np.transpose(self.fun_s(np.arange(1, N_PML_L + 1, 1) / (N_PML_L + 1)))

        # dx^2 times the Laplacian with PML on the two sides
        A = sparse.spdiags(1. / s_int, 0, N, N) * ddx_2 * sparse.spdiags(1. / s_half, 0, N + 1, N + 1) * ddx_1

        return A

    def build_laplacian_2d_PML(self, nx, ny, N_PML_L, N_PML_R, N_PML_B, N_PML_T):
        # Builds dx^2 times the Laplacian in 2D with effective outgoing boundary
        # condition implemented using PML on all four sides.
        # nx = number of grid points in x
        # ny = number of grid points in y
        # N_PML_L = number of PML pixels on the left; zero means no PML
        # N_PML_R = number of PML pixels on the right; zero means no PML
        # N_PML_B = number of PML pixels on the bottom; zero means no PML
        # N_PML_T = number of PML pixels on the top; zero means no PML

        # A = [(d^2/dx^2) + (d^2/dy^2)]*dx^2
        A = sparse.kron(self.build_laplacian_1d_PML(nx, N_PML_L, N_PML_R), sparse.eye(ny)) + \
            sparse.kron(sparse.eye(nx), self.build_laplacian_1d_PML(ny, N_PML_B, N_PML_T))
        return A

    def calculate_output_error(self, n_pred, x_b, y_aim,
                               nx, ny, k0dx, n_PML):
        # n_fix, simulation_region,  start_x, start_y, length_x, length_y,
        # n_pred_num = K.eval(n_pred)
        # n_fix_num = K.eval(n_fix)
        # n_x_b_num = K.eval(x_b)
        # n_x_b_num_complex = n_x_b_num[:,:,0] + n_x_b_num[:,:,1]*1j
        # n_y_aim_num = K.eval(y_aim)
        # n_y_aim_num_complex = n_y_aim_num[:,:,0] + n_y_aim_num[:,:,1]*1j
        # n_pred_num[n_pred_num < 0] = 0
        # n_fix_num[start_y: start_y + length_y, start_x: start_x + length_x] = n_fix_num[start_y: start_y + length_y,
        #                                                                       start_x: start_x + length_x] + n_pred_num
        A1 = self.build_laplacian_2d_PML(nx, ny, n_PML, n_PML, n_PML, n_PML)
        A1_tensor = tf.convert_to_tensor(A1.todense(), name='A1_tensor', dtype='complex64')
        n_tensor_transpose = tf.transpose(n_pred, conjugate=False, name='n_tensor_transpose')
        n_tensor_reshape = tf.reshape(n_tensor_transpose, [-1], name='n_tensor_reshape_flatten')
        A2 = linalg.diag(k0dx ** 2 * n_tensor_reshape ** 2, name='A2_tensor', padding_value=0, num_rows=nx * ny,
                         num_cols=nx * ny)
        A = tf.add(A1_tensor, A2, name='A_tensor')
        # A = self.build_laplacian_2d_PML(nx, ny, n_PML, n_PML, n_PML, n_PML) + tf.matrix_diag(
        #     k0dx ** 2 * n_fix_num.flatten('F') ** 2, 0, nx * ny, nx * ny)
        b_tensor_transpose = tf.transpose(x_b, conjugate=False, name='b_tensor_transpose')
        b_tensor_reshape = tf.reshape(b_tensor_transpose, [nx * ny, 1], name='b_tensor_reshape_flatten')
        psi_tot = tf.reshape(tf.linalg.solve(A, b_tensor_reshape, name='psi_solve'), [nx, ny],
                             name='psi_out_reshape')
        # psi_tot = np.reshape(linalg.spsolve(A, n_x_b_num_complex.flatten('F')), [ny, nx], order='F')
        # y_output_num = psi_tot[:, n_PML]
        # y_aim_num = n_y_aim_num_complex[:, n_PML]
        error_value = tf.subtract(tf.abs(y_aim, name='y_aim_amplitude'), tf.abs(psi_tot, name='psi_tot_amplitude'),
                                  name='error_value_small')
        error_value_large = tf.multiply(error_value, nx * ny, name='error_value_large')
        return error_value_large

    def compute_output_shape(selfself, input_shape):
        return input_shape


