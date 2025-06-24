import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model

# === 設定項目 ===
# 1. 必要なライブラリをインストールしてください:
#    pip install tensorflow opencv-python matplotlib numpy
#
# 2. このスクリプトは、指定された絶対パスからデータを読み込みます。
# =================

#--- 変数宣言 ---
# 感情ラベルのリスト (6クラス)
list_face_expression = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']
img_shape = (48, 48, 3)

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
        
        # (任意) 読み込むデータ件数を制限したい場合は、以下の行のコメントを解除
        # paths = paths[:500] # 例: 各フォルダ500件に制限

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
            label_list.append(i) # ラベルとして感情リストのインデックスを使用
            
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
# 例) 5 -> [0, 0, 0, 0, 0, 1]
y_train = to_categorical(y_train, num_classes=len(list_face_expression))
y_test = to_categorical(y_test, num_classes=len(list_face_expression))

#--- モデル構築 ---
print("\nBuilding model...")
# VGG16モデル構築
input_tensor = Input(shape=img_shape)
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# 全結合層モデル構築
sequential_model = Sequential()
sequential_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
sequential_model.add(Dense(256, activation='relu')) # 活性化関数をReLUに変更
sequential_model.add(Dropout(rate=0.5))
sequential_model.add(Dense(len(list_face_expression), activation='softmax')) # 出力層のユニット数をクラス数に合わせる

# VGG16モデルと全結合層モデルを結合
model = Model(inputs=vgg16.input, outputs=sequential_model(vgg16.output))

# VGG16モデルの重みを固定 (転移学習)
for layer in model.layers[:19]:
    layer.trainable = False

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# モデル構造表示
model.summary()

#--- 学習 ---
print("\nStarting training...")
history = model.fit(X_train, y_train, verbose=1, batch_size=64, epochs=25, validation_data=(X_test, y_test))

#--- 評価と結果保存 ---
print("\nEvaluating model...")
# モデルの汎化精度評価
scores = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {scores[0]}')
print(f'Test accuracy: {scores[1]}')

# resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# エポック毎のモデル精度推移をプロットして保存
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="x")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)

# エポック毎の損失をプロットして保存
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
#plt.show()


# 重みを保存
model_path = os.path.join(result_dir, 'emotion_model_vgg16.h5')
model.save(model_path)
print(f"Model saved to '{model_path}'")

#--- 学習済みモデルを使った予測関数の例 ---
def predict_emotion(model, img_array, classes):
    # モデルの入力に合わせて画像を4次元配列に変換 (1, 48, 48, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # 予測確率を取得
    pred_probabilities = model.predict(img_array)[0]
    # 最も確率の高いインデックスを取得
    pred_index = np.argmax(pred_probabilities)
    # 感情名と確率を返す
    return classes[pred_index], pred_probabilities[pred_index]

# テストセットからランダムに画像を選んで予測を試す
print("\n--- Prediction Example ---")
random_index = random.randint(0, len(X_test) - 1)
sample_image = X_test[random_index]
true_label_index = np.argmax(y_test[random_index])
true_label_name = list_face_expression[true_label_index]

# 予測実行
predicted_label, confidence = predict_emotion(model, sample_image, list_face_expression)

print(f"Predicted emotion: {predicted_label} (Confidence: {confidence:.2%})")
print(f"True emotion: {true_label_name}")

# 画像を表示
plt.figure()
plt.imshow(sample_image)
plt.title(f"Predicted: {predicted_label} | True: {true_label_name}")
plt.axis('off')
plt.show()
