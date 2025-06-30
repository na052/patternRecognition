import os

import math

import cv2

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation #BN追加

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split # ★ train_test_splitをインポート

from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix # ★★★ 追加 ★★★



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



# ★ 学習データを90%の訓練データと10%の検証データに分割

X_train, X_val, y_train, y_val = train_test_split(

    X_train_full,

    y_train_full,

    test_size=0.1,      # 10%を検証データとして使用

    random_state=42,    # 再現性のための乱数シード

    stratify=y_train_full_idx # 元のラベル比率を維持して分割

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

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x) #256-128



    # Batch Normalizationを適用

    x = BatchNormalization()(x)

    # 活性化関数を適用

    x = Activation('relu')(x)

    # ★★★ ここまでが変更部分 ★★★



    x = Dropout(0.5)(x)#変更する可能性あり

    output = Dense(len(list_face_expression), activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)

    return model, base



# === Stage1用 学習率スケジュール ===

def lr_schedule_stage1(epoch, lr):

    return 2e-4 if epoch < 5 else 1e-4



# ★★★ 交差検証ループを削除し、単一の学習プロセスに変更 ★★★



# === 学習準備 ===

train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)

steps_per_epoch = math.ceil(len(X_train) / batch_size)

X_val_normalized = X_val.astype('float32') / 255.0 # 検証データは正規化のみ



model, base_model = build_model()

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.012)



# === Stage 1: ヘッド部分の学習 ===

print("\n--- Stage 1: Training Head ---")

model.compile(optimizer=Adam(learning_rate=3e-4), loss=loss_fn, metrics=['accuracy'])

model.fit(

    train_gen,

    steps_per_epoch=steps_per_epoch,

    epochs=10,

    validation_data=(X_val_normalized, y_val),

    callbacks=[LearningRateScheduler(lr_schedule_stage1, verbose=1)],

    verbose=1

)



# === Stage 2: ファインチューニング ===

print("\n--- Stage 2: Fine-tuning ---")

base_model.trainable = True

freeze_until = 'block5_conv1'

set_trainable = False

for layer in base_model.layers:

    if layer.name == freeze_until:

        set_trainable = True

    layer.trainable = set_trainable



model.compile(optimizer=Adam(learning_rate=1e-5), loss= loss_fn, metrics=['accuracy'])

history = model.fit(

    train_gen,

    steps_per_epoch=steps_per_epoch,

    epochs=80,

    validation_data=(X_val_normalized, y_val),

    callbacks=[

        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),

        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    ],

    verbose=1

)





# === 最終評価と保存 ===

print(f"\n{'='*20} Final Evaluation {'='*20}")



# 検証データでの最終評価

val_loss, val_acc = model.evaluate(X_val_normalized, y_val, verbose=0)

print(f"Final Validation Accuracy: {val_acc:.4f}")



# テストデータでの総合的な最終評価

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print(f"\nFinal Test Accuracy (Overall): {test_acc:.4f}")



# テストデータに対する予測を実行

y_pred_prob = model.predict(X_test)

y_pred = np.argmax(y_pred_prob, axis=1) # 確率が最も高いクラスのインデックスを取得



# --- 感情ごとの性能評価レポート ---

print("\n--- Classification Report (Emotion-wise Performance) ---")

# y_test_idx は one-hotエンコード前の整数ラベル

# recall が各感情の「正答率」に相当します

print(classification_report(y_test_idx, y_pred, target_names=list_face_expression))





# --- 混同行列の作成と可視化 ---

cm = confusion_matrix(y_test_idx, y_pred)



plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Confusion Matrix')

plt.colorbar()

tick_marks = np.arange(len(list_face_expression))

plt.xticks(tick_marks, list_face_expression, rotation=45)

plt.yticks(tick_marks, list_face_expression)



# 行列内に数値を書き込む

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(j, i, format(cm[i, j], 'd'),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()



# 結果保存ディレクトリの作成

os.makedirs('results', exist_ok=True)



# 現在時刻を "YYYYMMDD_HHMMSS" の形式で取得

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



# 混同行列のプロットを保存

confusion_matrix_filename = f'results/confusion_matrix_{timestamp}.png'

plt.savefig(confusion_matrix_filename)

print(f"\nConfusion matrix plot saved to {confusion_matrix_filename}")





# モデルファイル名とプロットファイル名にタイムスタンプを追加

model_filename = f'results/emotion_model_{timestamp}.h5'

plot_filename = f'results/training_history_{timestamp}.png'



model.save(model_filename)

print(f"Model saved to {model_filename}")



# 学習履歴のプロット

training_history = history.history

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(training_history['accuracy'], marker='o', label='train_acc')

plt.plot(training_history['val_accuracy'], marker='x', label='val_acc')

plt.title('Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)



plt.subplot(1,2,2)

plt.plot(training_history['loss'], marker='o', label='train_loss')

plt.plot(training_history['val_loss'], marker='x', label='val_loss')

plt.title('Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)



plt.tight_layout()

plt.savefig(plot_filename)

print(f"Training history plot saved to {plot_filename}")
