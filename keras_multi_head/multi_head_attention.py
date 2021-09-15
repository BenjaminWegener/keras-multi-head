from .backend import keras, TF_KERAS
from .backend import backend as K
from keras_self_attention import ScaledDotProductAttention

class MultiHeadAttention(keras.layers.Layer):
    """
    Implements https://arxiv.org/abs/2006.04768
    """

    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')
        d_model = input_shape[-1]
        input_size = input_shape[-2]
        linformer_depth = input_shape[-1]

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        def create_projection_model():
            inputs = keras.models.Input(shape = (input_size, linformer_depth))
            outputs = keras.layers.Dense(linformer_depth, name = 'proj', kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal')(inputs)
            return keras.models.Model(inputs, outputs)

        projection_model = create_projection_model()
        
        def create_attention_model():
            queries = keras.models.Input(shape = (input_size, linformer_depth))
            keys = keras.models.Input(shape = (input_size, linformer_depth))
            values = keras.models.Input(shape = (input_size, linformer_depth))
            causal_mask = keras.models.Input(shape = (input_size, linformer_depth))
            infinity_mask = keras.models.Input(shape = (input_size, linformer_depth))

            keys_transposed = keras.layers.Permute((2, 1), name = 'keys_transposed')(keys) # 0, 64, 1024
            values_transposed = keras.layers.Permute((2, 1), name = 'values_transposed')(values) # 0, 64, 1024

            e_proj = projection_model(keys_transposed) # 0, 64, 64
            f_proj = projection_model(values_transposed)# 0, 64, 64

            e_proj_transposed = keras.layers.Permute((2, 1), name = 'e_proj_transpose')(e_proj) # 0, 64, 64
            f_proj_transposed = keras.layers.Permute((2, 1), name = 'f_proj_transpose')(f_proj) # 0, 64, 64
            qk = keras.layers.Dot(name = 'qk', axes = -1)([queries, e_proj_transposed]) # 0, 1024, 64
            p_bar = keras.layers.Lambda(lambda x: x * K.constant(1 / np.sqrt(linformer_depth)), name = 'p_bar', output_shape = lambda s: (s))(qk) # 0, 1024, 64
            p_bar = keras.layers.Multiply(name = 'multiply_causal')([p_bar, causal_mask]) # 0, 1024, 64
            p_bar = keras.layers.Add(name = 'add_infinitymask')([p_bar, infinity_mask])
            p_bar = keras.layers.Activation(activation = 'softmax', name = 'softmax')(p_bar) # 0, 1024, 64
            p_bar = keras.layers.Dropout(0.1, name = 'dropout')(p_bar)
            p_bar = keras.layers.Dot(axes = -1, name = 'p_bar-f_proj_matmul')([p_bar, f_proj_transposed]) # 0, 1024, 64

            return keras.models.Model([queries, keys, values, causal_mask, infinity_mask], p_bar)

        attention_model = create_attention_model()
        
        
        queries = keras.layers.Dense(linformer_depth)(inputs)
        keys = keras.layers.Dense(linformer_depth)(inputs)
        values = keras.layers.Dense(linformer_depth)(inputs)
        causal_mask = keras.layers.Lambda(lambda x: K.constant(np.expand_dims(np.transpose(np.triu(np.ones([linformer_depth, input_size]), 0)), 0)), output_shape = lambda s: (None, s[1], s[2]), name = 'causal_mask')(queries) # 0, 64, 1024 #only once compute
        infinity_mask = keras.layers.Lambda(lambda x: (x - 1) * 1e16, output_shape = lambda s: (None, s[1], s[2]), name = 'infinity_mask')(causal_mask) # 0, 1024, 64
        
        result = attention_model([queries, keys, values, causal_mask, infinity_mask])
        return keras.layers.Dense(depth)(result)
        
    def compute_output_shape(self, input_shape):
        return input_shape
