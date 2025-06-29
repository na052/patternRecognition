import os
import math
import cv2
import numpy as np
import matplotlib
import tensorflow as tf
# GUIがない環境でもエラーが出ないようにするための設定
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 # ★★★ L2正則化のためにインポート ★★★

# === 設定項目 ===
# 1. 必要なライブラリをインストールしてください:
#    conda install -c conda-forge tensorflow opencv matplotlib numpy scipy
#
# 2. このスクリプトは、指定された絶対パスからデータを読み込みます。
# =================


#--- 変数宣言 ---
# 感情ラベルのリスト (6クラス)
list_face_expression = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']
img_shape = (144, 144, 3)
batch_size = 128 # バッチサイズを変数として定義

# データセットのベースパス (ユーザー指定の絶対パス)
# 環境に合わせて変更してください
train_base_path = '/home/eito/codes/pattern/dataset/Train'
test_base_path = '/home/eito/codes/pattern/dataset/Test'


#--- 画像とラベルを読み込む関数 ---
def load_data_from_path(base_path, emotion_list):
    """指定されたベースパスから画像とラベルを読み込む"""
    img_list = []
    label_list = []
    print(f"--- Loading data from: {base_path} ---")
    for i, expression in enumerate(emotion_list):
        # 各感情のディレクトリパス
        expression_dir = os.path.join(base_path, expression)
        
        # ディレクトリが存在するかチェック
        if not os.path.isdir(expression_dir):
            print(f"警告: ディレクトリが見つかりません: {expression_dir}。スキップします。")
            continue

        paths = os.listdir(expression_dir)
        print(f"Found {len(paths)} images for '{expression}'")
        
        for path in paths:
            img_path = os.path.join(expression_dir, path)
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                print(f"警告: 画像が読み込めませんでした: {img_path}。スキップします。")
                continue
            
            # 画像のリサイズと色チャンネルの並び替え (BGR -> RGB)
            bgr_img = cv2.resize(bgr_img, img_shape[:2])
            b, g, r = cv2.split(bgr_img)
            rgb_img = cv2.merge([r, g, b])
            
            img_list.append(rgb_img)
            label_list.append(i)
            
    return (np.array(img_list), np.array(label_list))

#--- データ読み込みと前処理 ---
try:
    X_train, y_train = load_data_from_path(train_base_path, list_face_expression)
    X_test, y_test = load_data_from_path(test_base_path, list_face_expression)
except Exception as e:
    print(f"データ読み込み中にエラーが発生しました: {e}")
    exit()

print("\n--- Data Summary ---")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 正解ラベルをone-hotエンコーディング
y_train = to_categorical(y_train, num_classes=len(list_face_expression))
y_test = to_categorical(y_test, num_classes=len(list_face_expression))


#--- データ拡張の準備 ---
print("\nSetting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
X_test_normalized = X_test.astype('float32') / 255.0


#--- モデル構築 ---
print("\nBuilding model...")
input_tensor = Input(shape=img_shape)
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

sequential_model = Sequential()
sequential_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
# ★★★ ここにL2正則化を適用 ★★★
sequential_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
sequential_model.add(Dropout(rate=0.5))
sequential_model.add(Dense(len(list_face_expression), activation='softmax'))

model = Model(inputs=vgg16.input, outputs=sequential_model(vgg16.output))

# VGG16のblock5以降を再学習可能にする（ファインチューニング）
for layer in model.layers[:15]:
    layer.trainable = False
for layer in model.layers[15:]:
    layer.trainable = True

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


#--- コールバックの定義 ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

#--- 学習 ---
print("\nStarting training with data augmentation...")
steps_per_epoch = math.ceil(train_generator.n / batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=(X_test_normalized, y_test),
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)


#--- 評価と結果保存 ---
print("\nEvaluating model...")
scores = model.evaluate(X_test_normalized, y_test, verbose=1)
print(f'Test loss: {scores[0]}')
print(f'Test accuracy: {scores[1]}')

result_dir = 'results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="x")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss", marker="o")
plt.plot(history.history["val_loss"], label="Validation Loss", marker="x")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'training_history.png'))
print(f"Training history graph saved to '{result_dir}/training_history.png'")

model_path = os.path.join(result_dir, 'emotion_model_vgg16.h5')
model.save(model_path)
print(f"Model saved to '{model_path}'")


#--- 学習済みモデルを使った予測関数の例 ---
def predict_emotion(model, img_array, classes):
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_probabilities = model.predict(img_array)[0]
    pred_index = np.argmax(pred_probabilities)
    return classes[pred_index], pred_probabilities[pred_index]

print("\n--- Prediction Example ---")
# テストデータが存在する場合のみ予測を実行
if len(X_test) > 0:
    random_index = random.randint(0, len(X_test) - 1)
    sample_image = X_test[random_index]
    true_label_index = np.argmax(y_test[random_index])
    true_label_name = list_face_expression[true_label_index]

    predicted_label, confidence = predict_emotion(model, sample_image, list_face_expression)

    print(f"Predicted emotion: {predicted_label} (Confidence: {confidence:.2%})")
    print(f"True emotion: {true_label_name}")

    plt.figure()
    plt.imshow(sample_image)
    plt.title(f"Predicted: {predicted_label} | True: {true_label_name}")
    plt.axis('off')
    # plt.show()の代わりにファイルに保存
    plt.savefig(os.path.join(result_dir, 'prediction_example.png'))
    print(f"Prediction example image saved to '{result_dir}/prediction_example.png'")
else:
    print("No test data to run prediction example.")
