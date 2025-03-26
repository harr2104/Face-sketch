import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, ReLU, Dropout, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load Paired Dataset 
def load_paired_images(data_dir, img_size=(256, 256)):
    images = []
    sketches = []
    
    img_dir = os.path.join(data_dir, 'images')
    sketch_dir = os.path.join(data_dir, 'sketches')

    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        sketch_path = os.path.join(sketch_dir, filename)

        img = cv2.imread(img_path)
        sketch = cv2.imread(sketch_path)

        if img is not None and sketch is not None:
            img = cv2.resize(img, img_size).astype(np.float32) / 127.5 - 1  # Normalize to [-1,1]
            sketch = cv2.resize(sketch, img_size).astype(np.float32) / 127.5 - 1
            images.append(img)
            sketches.append(sketch)

    return np.array(images, dtype=np.float32), np.array(sketches, dtype=np.float32)


image_data, sketch_data = load_paired_images("C:\\Users\\sh103\\Downloads\\portraits-sketches")
def build_generator():
    inputs = Input(shape=(256, 256, 3))

    
    down1 = Conv2D(64, (4,4), strides=2, padding='same', use_bias=False)(inputs)
    down1 = LeakyReLU(negative_slope=0.2)(down1)  # ✅ FIXED

    down2 = Conv2D(128, (4,4), strides=2, padding='same', use_bias=False)(down1)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(negative_slope=0.2)(down2)  # ✅ FIXED

    down3 = Conv2D(256, (4,4), strides=2, padding='same', use_bias=False)(down2)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(negative_slope=0.2)(down3)  # ✅ FIXED

    
    up1 = Conv2DTranspose(128, (4,4), strides=2, padding='same', use_bias=False)(down3)
    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)
    up1 = Concatenate()([up1, down2])

    up2 = Conv2DTranspose(64, (4,4), strides=2, padding='same', use_bias=False)(up1)
    up2 = BatchNormalization()(up2)
    up2 = ReLU()(up2)
    up2 = Concatenate()([up2, down1])

    outputs = Conv2DTranspose(3, (4,4), strides=2, activation="tanh", padding='same')(up2)
    
    return Model(inputs, outputs)


def build_discriminator():
    inputs = Input(shape=(256, 256, 3))
    target = Input(shape=(256, 256, 3))

    combined = Concatenate(axis=-1)([inputs, target])
    
    down1 = Conv2D(64, (4,4), strides=2, padding='same', use_bias=False)(combined)
    down1 = LeakyReLU(negative_slope=0.2)(down1)  # ✅ FIXED
    
    down2 = Conv2D(128, (4,4), strides=2, padding='same', use_bias=False)(down1)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(negative_slope=0.2)(down2)  # ✅ FIXED
    
    down3 = Conv2D(256, (4,4), strides=2, padding='same', use_bias=False)(down2)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(negative_slope=0.2)(down3)  # ✅ FIXED
    
    outputs = Conv2D(1, (4,4), strides=1, activation="sigmoid", padding='same')(down3)
    
    return Model([inputs, target], outputs)



generator = build_generator()
discriminator = build_discriminator()


loss_object = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output, generated_image, target_image):
    adv_loss = loss_object(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target_image - generated_image))  # L1 loss
    return adv_loss + (100 * l1_loss)  # Weighted loss

def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


epochs = 100
batch_size = 1

for epoch in range(epochs):
    for i in range(len(image_data)):
        real_image = np.expand_dims(image_data[i], axis=0)  # (1, 256, 256, 3)
        real_sketch = np.expand_dims(sketch_data[i], axis=0)  # (1, 256, 256, 3)

        real_image = tf.convert_to_tensor(real_image, dtype=tf.float32)
        real_sketch = tf.convert_to_tensor(real_sketch, dtype=tf.float32)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_sketch = generator(real_image, training=True)
            
            real_output = discriminator([real_image, real_sketch], training=True)
            fake_output = discriminator([real_image, generated_sketch], training=True)

            g_loss = generator_loss(fake_output, generated_sketch, real_sketch)
            d_loss = discriminator_loss(real_output, fake_output)

        gradients_gen = gen_tape.gradient(g_loss, generator.trainable_variables)
        gradients_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} - Generator Loss: {g_loss:.4f} | Discriminator Loss: {d_loss:.4f}")

    
    if epoch % 20 == 0:
        generator.save(f"pix2pix_generator_epoch{epoch}.h5")
        discriminator.save(f"pix2pix_discriminator_epoch{epoch}.h5")


generator.save("pix2pix_generator_final.h5")
discriminator.save("pix2pix_discriminator_final.h5")
