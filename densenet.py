import tensorflow as tf
from tensorflow import keras


def dense_layer(x, k=32, stage=0, name='', l=0):
    x_skip = x
    x1 = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.ReLU(name=f'{stage}_relu{l}1')(x1)
    x1 = keras.layers.Conv2D(filters=4*k, kernel_size=(1,1), padding='same', name=f'{stage}_conv{name}_{l}1')(x1)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.ReLU(name=f'{stage}_relu{l}3')(x1)
    x1 = keras.layers.Conv2D(filters=k, kernel_size=(3,3), padding='same', name=f'{stage}_conv{name}_{l}3')(x1)

    return keras.layers.concatenate([x, x1], axis=-1, name=f'{stage}_concat{name}_{l}')

def dense_block(x, blocks, k, stage):
    name = 97
    for i in range(blocks):
        x = dense_layer(x, k, stage, chr(name+i), i)
    return x
    
def transition_layer(x, k, reduction, name):
    bn_axis = x.shape[-1]
    x = keras.layers.BatchNormalization(name=f'transBatch_{name}')(x)
    x = keras.layers.ReLU(name=f'relu_{name}')(x)
    x = keras.layers.Conv2D(filters=reduction*bn_axis, kernel_size=(1,1), padding='same', name=f'transConv_{name}')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2,2), strides=2, name=f'transAvpool_{name}')(x)

    return x
    

def DenseNet(inputs=None, blocks=[6, 12, 24, 16], k=32, reduction=0.5, include_top=True, input_shape=(224,224,3)):
    
    inputs = keras.layers.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', name='conv0')(inputs)
    x = keras.layers.BatchNormalization(name='batch0')(x)
    x = keras.layers.ReLU(name='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same', strides=2, name='maxpool0')(x)

    x = dense_block(x, blocks[0], k, 'a')
    x = transition_layer(x, k, reduction, 'a')
    x = dense_block(x, blocks[1], k, 'b')
    x = transition_layer(x, k, reduction, 'b')
    x = dense_block(x, blocks[2], k, 'c')
    x = transition_layer(x, k, reduction, 'c')
    x = dense_block(x, blocks[3], k, 'd')

    x = keras.layers.BatchNormalization(name='batch1')(x)
    x = keras.layers.ReLU(name='relu2')(x)
    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avpool0')(x)
        x = keras.layers.Dense(1000, activation='softmax', name='FullyConnected0')(x)
        
    model = keras.Model(inputs, x, name='densenet')
    
    return model
