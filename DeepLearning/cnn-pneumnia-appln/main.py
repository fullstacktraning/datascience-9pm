# tensorflow - used to build deep learning application
import tensorflow as tf

# ImageDataGenerator is the predefined class
# ImageDataGenerator, used to implement the augumentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# these classes helps to create hidden layers while building model
from tensorflow.keras import layers, models

# MobileNetV2 is the trained model
# 94.4%
# USA - Hospital
# Edges, Curves, Textures, Objects
from tensorflow.keras.applications import MobileNetV2

# EarlyStopping - used to stop the training, if no improvement detected
# ReduceLROnPlateau, used for microlevel management, to monitor accuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# class_weight, used to balance the weights between normal and pneoumia data
from sklearn.utils.class_weight import compute_class_weight

# numpy - used to perform matrix operations
import numpy as np

# plot the graphs
import matplotlib.pyplot as plt

# 📁 Paths
train_dir = "data/chest_xray/train"
val_dir = "data/chest_xray/val"
test_dir = "data/chest_xray/test"

IMG_SIZE = 224
BATCH_SIZE = 32 
   

# 🔄 Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_test_datagen = ImageDataGenerator(rescale=1./255)


train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# AUGUMENTATION COMPLETED SUCCESSFULLY #

# ⚖️ CLASS WEIGHTS (VERY IMPORTANT)
class_weights = compute_class_weight(
    class_weight='balanced',                               
    classes=np.unique(train_data.classes),                   
    y=train_data.classes                                    
)
class_weights = dict(enumerate(class_weights))              
print("Class Weights:", class_weights)

# CLASS WEIGHTS EXPALNATION DONE #


# 🧠 Base Model
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)


# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# 🧠 Final Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
]) 


# ⚙️ Compile (LOW LR)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 🛑 Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# 🏋️ Train
history = model.fit(
    train_data,
    epochs=25,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# 📊 Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()

# 📊 Plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()

# 🧪 Test Evaluation
test_loss, test_acc = model.evaluate(test_data)
print("✅ Final Test Accuracy:", test_acc)

# 💾 Save Model
model.save("pneumonia_model.h5") 

# 🔮 Prediction Function
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        print("🦠 Pneumonia Detected")
    else:
        print("✅ Normal")


predict_image("data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")