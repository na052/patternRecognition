import os
import math
import matplotlib
import numpy as np
import tensorflow as tf
# GUIがない環境でもエラーが出ないようにするための設定
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# === 設定項目 ===
list_face_expression = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']
img_shape = (224, 224, 3)
batch_size = 32  # メモリ節約のため小さめに設定
train_base_path = '/home/eito/codes/pattern/dataset/Train'
test_base_path  = '/home/eito/codes/pattern/dataset/Test'


#=== データジェネレータの準備 ===
# 学習用: データ拡張＋preprocess_input
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    #brightness_range=[0.9, 1.1],
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # 学習/検証に分割
)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★【修正点】検証用は前処理のみを行うジェネレータを別途用意 ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1 # 学習用と同じデータ分割を行うためsplit設定は必要
)


train_generator = train_datagen.flow_from_directory(
    train_base_path,
    target_size=img_shape[:2],
    color_mode='rgb',
    classes=list_face_expression,
    class_mode='categorical',
    batch_size=batch_size,
    subset='training',
    shuffle=True
)

# ★★★【修正点】validation_datagen から検証用ジェネレータを作成 ★★★
validation_generator = validation_datagen.flow_from_directory(
    train_base_path,
    target_size=img_shape[:2],
    color_mode='rgb',
    classes=list_face_expression,
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation',
    shuffle=False
)

# テスト用: 前処理のみ
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_base_path,
    target_size=img_shape[:2],
    color_mode='rgb',
    classes=list_face_expression,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

steps_per_epoch = math.ceil(train_generator.n / batch_size)
validation_steps = math.ceil(validation_generator.n / batch_size)
test_steps = math.ceil(test_generator.n / batch_size)

#=== モデル構築 ===
input_tensor = Input(shape=img_shape)
base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.003))(x)
predictions = Dense(len(list_face_expression), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#=== ステージ1: 分類器ヘッドのみ訓練 ===
base_model.trainable = False
model.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_stage1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=12,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1
)

#=== ステージ2: ファインチューニング ===
base_model.trainable = True
# 末尾20層を学習対象に
fine_tune_from = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

#=== 評価と保存 ===
scores = model.evaluate(test_generator, steps=test_steps, verbose=1)
print(f'Test loss: {scores[0]:.4f}, Test accuracy: {scores[1]:.4f}')

result_dir = 'results_efficientnet'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 学習曲線のプロット
# (前回の提案の通り、historyを結合するとより良いグラフになりますが、ここでは元のロジックのままにしておきます)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], marker='o', label='Train Acc')
plt.plot(history.history['val_accuracy'], marker='x', label='Val Acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], marker='o', label='Train Loss')
plt.plot(history.history['val_loss'], marker='x', label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'training_history.png'))

# モデル保存
model_path = os.path.join(result_dir, 'emotion_model_efficientnet.h5')
model.save(model_path)
print(f"Model saved to '{model_path}'")