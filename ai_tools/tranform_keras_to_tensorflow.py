import tensorflow as tf
from tensorflow import keras

model_path = 'models/covid19_3.model'

model = keras.models.load_model(model_path)
MODEL_EXPORT_PATH = 'models/servable/2'


# We'll need to create an input mapping, and name each of the input tensors.
# In the VGG16 Keras model, there is only a single input and we'll name it 'image'
input_names = ['image']
name_to_input = {name: t_input for name, t_input in zip(input_names, model.inputs)}

# Save the model to the MODEL_EXPORT_PATH
# this saves the model in tensorflow format to be served using tensorflow model server
keras.models.save_model(model, MODEL_EXPORT_PATH, save_format='tf')