import os
import cv2
import numpy as np
import matplotlib
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 設定項目 ---
# ▼▼▼ 評価したい学習済みモデルのパスをここに指定してください ▼▼▼
SAVED_MODEL_PATH = 'results_resnet/model_surp1_0_ang1_0_20250702_195557.h5'
# ▲▲▲ 評価したい学習済みモデルのパスをここに指定してください ▲▲▲


list_face_expression = ['happy', 'sad', 'neutral','surprise', 'anger']
img_shape = (144, 144, 3)
batch_size = 8  # メモリに応じて調整

# テストデータのパス
data_root = '/home/eito/codes/pattern/dataset/Mangatest'
test_base_path = os.path.join(data_root)
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)

# === 画像とラベルを読み込む関数 (元のスクリプトから流用) ===
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

# === メイン処理 ===
if __name__ == '__main__':
    # 1. テストデータの読み込み
    print(f"--- Loading test data from: {test_base_path} ---")
    X_test, y_test_idx = load_data_from_path(test_base_path, list_face_expression)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_idx, num_classes=len(list_face_expression))
    print(f"Found {len(X_test)} test samples.")

    # 2. データの正規化
    print("--- Preprocessing test data ---")
    X_test_processed = preprocess_input(X_test.copy())

    # 3. 学習済みモデルの読み込み
    print(f"--- Loading saved model from: {SAVED_MODEL_PATH} ---")
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"[エラー] モデルファイルが見つかりません: {SAVED_MODEL_PATH}")
        exit()
    model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    model.summary()

    # 4. モデルの評価
    print("\n--- Evaluating model on test data ---")
    test_loss, test_acc = model.evaluate(X_test_processed, y_test_categorical, batch_size=batch_size, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # 5. モデルによる予測
    print("\n--- Generating predictions ---")
    y_pred_prob = model.predict(X_test_processed, batch_size=batch_size)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)

    # 6. Classification Report の表示
    print("\n--- Classification Report ---")
    print(classification_report(y_test_idx, y_pred_idx, target_names=list_face_expression, zero_division=0))

    # 7. 混同行列 (Confusion Matrix) のプロットと保存
    print("--- Generating and saving confusion matrix ---")
    cm = confusion_matrix(y_test_idx, y_pred_idx)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Data)')
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
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confusion_matrix_filename = os.path.join(results_dir, f'test_confusion_matrix_{run_timestamp}.png')
    plt.savefig(confusion_matrix_filename)
    plt.close()
    print(f"Confusion matrix saved to: {confusion_matrix_filename}")

    print("\n--- Test complete ---")
