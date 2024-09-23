import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

color_tensors = np.load('./color_tensors.npy')
bw_tensors = np.load('./bw_tensors.npy')

print(color_tensors.shape)

color_tensors = color_tensors.astype(np.float32)
bw_tensors = bw_tensors.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    bw_tensors, color_tensors, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

batch_size = 16
train_dataset = (train_dataset
                 .shuffle(buffer_size=len(X_train))
                 .batch(batch_size)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

def build_colorization_model(input_shape=(400, 400, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    
    # Decoder
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(x) 
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

print("starting model")
model = build_colorization_model()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset, epochs=1, validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

model.save('colorization_model.h5')