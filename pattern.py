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

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split

from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix



# === 乱数シード固定 ===

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)



# === 設定項目 ===

list_face_expression = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']

img_shape = (96, 96, 3)

batch_size = 128



data_root = '/home/eito/codes/pattern/dataset'

train_base_path = os.path.join(data_root, 'Train')

test_base_path = os.path.join(data_root, 'Test')



# === 画像とラベルを読み込む関数 ===

def load_data_from_path(base_path, emotion_list):

    img_list, label_list = [], []

    for i, expr in enumerate(emotion_list):

        dir_path = os.path.join(base_path, expr)

        if not os.path.isdir(dir_path):

            print(f"警告: ディレクトリが見つかりません: {dir_path}")

            continue

        for fname in os.listdir(dir_path):

            path = os.path.join(dir_path, fname)

            img = cv2.imread(path)

            if img is None:

                print(f"警告: 画像が読み込めませんでした: {path}")

                continue

            img = cv2.resize(img, img_shape[:2])

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append(img)

            label_list.append(i)

    return np.array(img_list), np.array(label_list)



# === データ読み込みと前処理 ===

X_train_full, y_train_full_idx = load_data_from_path(train_base_path, list_face_expression)

X_test, y_test_idx = load_data_from_path(test_base_path, list_face_expression)

print(f"Total training samples: {X_train_full.shape[0]}, Test samples: {X_test.shape[0]}")



y_train_full = to_categorical(y_train_full_idx, num_classes=len(list_face_expression))

y_test = to_categorical(y_test_idx, num_classes=len(list_face_expression))

X_test = X_test.astype('float32') / 255.0



X_train, X_val, y_train, y_val = train_test_split(

    X_train_full,

    y_train_full,

    test_size=0.1,

    random_state=SEED,

    stratify=y_train_full_idx

)

print(f"-> Split into {X_train.shape[0]} training samples and {X_val.shape[0]} validation samples.")



# === データ拡張設定 ===

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



# === モデル構築関数 ===

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



#step1の学習率

def lr_schedule_stage1(epoch, lr):

    return 2e-4 if epoch < 5 else 1e-4



# === パラメータチューニングのメインロジック ===



weight_patterns = [

    (1.0, 1.0, 1.0),

    (1.1, 1.1, 1.1),

    (1.1, 1.2, 1.1),

]



results_dir = 'results'

os.makedirs(results_dir, exist_ok=True)



# ★★★ ここからが変更点 ★★★

session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_log_path = os.path.join(results_dir, f'tuning_log_{session_timestamp}.csv')

# ★★★ ここまでが変更点 ★★★



with open(csv_log_path, 'w', newline='') as f:

    writer = csv.writer(f)

    writer.writerow(['fear_w', 'surprise_w', 'anger_w', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'run_timestamp'])



for w_fear, w_surprise, w_anger in weight_patterns:

    print(f"\n{'='*40}")

    print(f"  STARTING RUN FOR WEIGHTS: fear={w_fear}, surprise={w_surprise}, anger={w_anger}")

    print(f"{'='*40}\n")



    tf.keras.backend.clear_session()

    model, base_model = build_model()

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.012)



    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED)

    steps_per_epoch = math.ceil(len(X_train) / batch_size)

    X_val_normalized = X_val.astype('float32') / 255.0



    print("\n--- Stage 1: Training Head ---")

    model.compile(optimizer=Adam(learning_rate=3e-4), loss=loss_fn, metrics=['accuracy'])

    model.fit(

        train_gen,

        steps_per_epoch=steps_per_epoch,

        epochs=12,

        validation_data=(X_val_normalized, y_val),

        callbacks=[LearningRateScheduler(lr_schedule_stage1, verbose=0)],

        verbose=1

    )



    print("\n--- Stage 2: Fine-tuning ---")

    base_model.trainable = True

    freeze_until = 'block4_conv1' # 例としてblock4からファインチューニング

    set_trainable = False

    for layer in base_model.layers:

        if layer.name == freeze_until:

            set_trainable = True

        layer.trainable = set_trainable



    model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=['accuracy'])

    class_weights = { 0: 1.0, 1: 1.0, 2: 1.0, 3: w_fear, 4: w_surprise, 5: w_anger }

    print("\nApplying custom class weights:")

    print(class_weights)



    history = model.fit(

        train_gen,

        steps_per_epoch=steps_per_epoch,

        epochs=80,

        class_weight=class_weights,

        validation_data=(X_val_normalized, y_val),

        callbacks=[

            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),

            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        ],

        verbose=1

    )



    print(f"\n--- Final Evaluation for weights ({w_fear}, {w_surprise}, {w_anger}) ---")

    val_loss, val_acc = model.evaluate(X_val_normalized, y_val, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"  Validation -> Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

    print(f"  Test       -> Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")



    y_pred_prob = model.predict(X_test)

    y_pred = np.argmax(y_pred_prob, axis=1)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    param_str = f"fear{w_fear}_surp{w_surprise}_ang{w_anger}".replace('.', '_')



    print("\n--- Classification Report ---")

    print(classification_report(y_test_idx, y_pred, target_names=list_face_expression))



    cm = confusion_matrix(y_test_idx, y_pred)

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title(f'Confusion Matrix (weights: {w_fear}, {w_surprise}, {w_anger})')

    plt.colorbar()

    tick_marks = np.arange(len(list_face_expression))

    plt.xticks(tick_marks, list_face_expression, rotation=45)

    plt.yticks(tick_marks, list_face_expression)

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(j, i, format(cm[i, j], 'd'),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    confusion_matrix_filename = os.path.join(results_dir, f'cm_{param_str}_{run_timestamp}.png')

    plt.savefig(confusion_matrix_filename)

    plt.close()



    training_history = history.history

    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)

    plt.plot(training_history['accuracy'], marker='o', label='train_accuracy')

    plt.plot(training_history['val_accuracy'], marker='x', label='val_accuracy')

    plt.title('Model Accuracy')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.grid(True)

    plt.subplot(1, 2, 2)

    plt.plot(training_history['loss'], marker='o', label='train_loss')

    plt.plot(training_history['val_loss'], marker='x', label='val_loss')

    plt.title('Model Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plot_filename = os.path.join(results_dir, f'history_{param_str}_{run_timestamp}.png')

    plt.savefig(plot_filename)

    plt.close()



    model_filename = os.path.join(results_dir, f'model_{param_str}_{run_timestamp}.h5')

    model.save(model_filename)



    with open(csv_log_path, 'a', newline='') as f:

        writer = csv.writer(f)

        writer.writerow([w_fear, w_surprise, w_anger, val_acc, val_loss, test_acc, test_loss, run_timestamp])



    print(f"--- Finished run for weights ({w_fear}, {w_surprise}, {w_anger}). Results saved. ---")



print("\nAll hyperparameter tuning runs are complete!")

print(f"Check the summary at: {csv_log_path}")
