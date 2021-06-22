#%%
import tensorflow as tf
import utils
import matplotlib.pyplot as plt
print(tf.__version__)
# %%
(x_train, y_train),(x_test, y_test) = utils.load_data()
utils.plot_random_examples(x_train, y_train).show()
# %%
class ParametricRelu(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(ParametricRelu, self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.alpha = self.add_weight(name='minimum',shape=(1,),initializer='zeros',trainable=True)
        super(ParametricRelu, self).build(input_shape)

    def call(self,x):
        return tf.maximum(0.,x) + self.alpha * tf.minimum(0.,x)
#%%
def create_model(use_prelu=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64,input_shape=(784,)))
    if not use_prelu:
        model.add(tf.keras.layers.ReLU())
    else:
        model.add(ParametricRelu())
    model.add(tf.keras.layers.Dense(10,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.add(tf.keras.layers.Dense(10,activation='tanh'))
    model.compile(loss = 'categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])
    return model

model = create_model()
model.summary()
# %%
print('Initial Alpha: ',model.layers[1].get_weights())
h = model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=5)
print('Final Alpha: ',model.layers[1].get_weights())
# %%
utils.plot_results(h).show()
# %%
model = create_model(use_prelu=False)
model.summary()
# %%
h = model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=5)
utils.plot_results(h).show()
# %%
