import os
import math
import random
import cv2
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

list_face_expression = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']
img_shape = (96, 96, 3)
batch_size = 128

# データ読み込み

def load_data_from_path(base_path, emotion_list):
    img_list, label_list = [], []
    for i, expr in enumerate(emotion_list):
        dir_path = os.path.join(base_path, expr)
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            path = os.path.join(dir_path, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, img_shape[:2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
            label_list.append(i)
    return np.array(img_list), np.array(label_list)

# OneCycleLRコールバック
class OneCycleLRCallback(Callback):
    def __init__(self, max_lr, total_steps, pct_start=0.3, final_div_factor=500):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor
        self.lr_schedule = self.build_schedule()
        self.step = 0

    def build_schedule(self):
        peak_step = int(self.total_steps * self.pct_start)
        lrs = []
        for t in range(peak_step):
            lrs.append(self.max_lr * t / peak_step)
        for t in range(peak_step, self.total_steps):
            lrs.append(self.max_lr * ((self.total_steps - t) / (self.total_steps - peak_step)) / self.final_div_factor)
        return lrs

    def on_train_batch_begin(self, batch, logs=None):
        if self.step < len(self.lr_schedule):
            lr = self.lr_schedule[self.step]
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            self.step += 1

# メイン処理

data_root = '/home/eito/codes/pattern/dataset'
train_base_path = os.path.join(data_root, 'Train')
test_base_path = os.path.join(data_root, 'Test')

X_train_full, y_train_full_idx = load_data_from_path(train_base_path, list_face_expression)
X_test, y_test_idx = load_data_from_path(test_base_path, list_face_expression)

y_train_full = to_categorical(y_train_full_idx, num_classes=len(list_face_expression))
y_test = to_categorical(y_test_idx, num_classes=len(list_face_expression))
X_test = X_test.astype('float32') / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full_idx)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9,1.1],
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# モデル構築

def build_model():
    input_tensor = Input(shape=img_shape)
    base = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    base.trainable = False
    x = Flatten()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(list_face_expression), activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    return model, base

def lr_schedule_stage1(epoch, lr):
    return 2e-4 if epoch < 5 else 1e-4

weight_patterns = [
    (1.0, 1.0, 1.0),
    (1.1, 1.1, 1.1),
    (1.1, 1.2, 1.1),
    (1.1, 1.3, 1.1),
    (1.1, 1.4, 1.1),
]

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
csv_log_path = os.path.join(results_dir, 'tuning_log.csv')
with open(csv_log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fear_w', 'surprise_w', 'anger_w', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'timestamp'])

for w_fear, w_surprise, w_anger in weight_patterns:
    tf.keras.backend.clear_session()
    model, base_model = build_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.012)

    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED)
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    total_steps = steps_per_epoch * 80

    X_val_normalized = X_val.astype('float32') / 255.0

    # Stage 1
    model.compile(optimizer=Adam(learning_rate=3e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=12,
        validation_data=(X_val_normalized, y_val),
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule_stage1)],
        verbose=1
    )

    # Stage 2
    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block_conv1':
            set_trainable = True
        layer.trainable = set_trainable

    model.compile(optimizer=Adam(learning_rate=1e-6), loss=loss_fn, metrics=['accuracy'])
    class_weights = { 0: 1.0, 1: 1.0, 2: 1.0, 3: w_fear, 4: w_surprise, 5: w_anger }
    scheduler_callback = OneCycleLRCallback(max_lr=5e-4, total_steps=total_steps, pct_start=0.3, final_div_factor=500)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=80,
        class_weight=class_weights,
        validation_data=(X_val_normalized, y_val),
        callbacks=[
            scheduler_callback,
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val_normalized, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"fear{w_fear}_surp{w_surprise}_ang{w_anger}".replace('.', '_')

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(list_face_expression))
    plt.xticks(tick_marks, list_face_expression, rotation=45)
    plt.yticks(tick_marks, list_face_expression)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'cm_{param_str}_{timestamp}.png'))
    plt.close()

    with open(csv_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([w_fear, w_surprise, w_anger, val_acc, val_loss, test_acc, test_loss, timestamp])

print("All hyperparameter tuning runs are complete!")
