import tensorflow as tf
from main import *

model = tf.keras.models.load_model('handwritten.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

