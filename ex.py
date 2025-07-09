import os
import math
import random
import cv2
import csv
import numpy as np
import matplotlib
import gc
matplotlib.use('Agg')  # GUIバックエンドがない環境でも動作するように設定
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# === 設定クラス ===
class Config:
    """設定を管理するクラス"""
    SEED = 42
    IMG_SHAPE = (128, 128, 3)
    BATCH_SIZE = 16
    DATA_ROOT = '/home/eito/codes/pattern/dataset'
    EMOTIONS = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']
    
    # 学習設定
    STAGE1_EPOCHS = 18
    STAGE2_EPOCHS = 50
    VALIDATION_SPLIT = 0.1
    
    # 学習率設定
    STAGE1_INITIAL_LR = 3e-4
    STAGE2_INITIAL_LR = 1e-5
    STAGE2_MAX_LR = 7e-6
    STAGE2_MIN_LR = 1e-6
    
    # データ拡張設定
    NOISE_PROBABILITY = 0.3
    BRIGHTNESS_PROBABILITY = 0.5
    NOISE_SCALE = 25.0
    BRIGHTNESS_RANGE = (0.9, 1.1)

# === 乱数シード固定 ===
def set_seeds():
    """乱数シードを設定"""
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    tf.random.set_seed(Config.SEED)

# === GPU設定 ===
def setup_gpu():
    """GPU設定を行う"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Info] Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"[Error] Could not set memory growth: {e}")
    else:
        print("[Warning] No GPU found, running on CPU.")

# === データ読み込みクラス ===
class DataLoader:
    """データ読み込みを管理するクラス"""
    
    def __init__(self, data_root):
        self.data_root = data_root
        self.train_base_path = os.path.join(data_root, 'Train')
        self.test_base_path = os.path.join(data_root, 'Test')
    
    def load_data_from_path(self, base_path, emotion_list):
        """指定されたパスから画像とラベルを読み込む"""
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
                img = cv2.resize(img, Config.IMG_SHAPE[:2])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_list.append(img)
                label_list.append(i)
        return np.array(img_list), np.array(label_list)
    
    def load_all_data(self):
        """全データを読み込む"""
        X_train_full, y_train_full_idx = self.load_data_from_path(
            self.train_base_path, Config.EMOTIONS
        )
        X_test, y_test_idx = self.load_data_from_path(
            self.test_base_path, Config.EMOTIONS
        )
        
        print(f"Total training samples: {X_train_full.shape[0]}, Test samples: {X_test.shape[0]}")
        
        # ワンホットエンコーディング
        y_train_full = to_categorical(y_train_full_idx, num_classes=len(Config.EMOTIONS))
        y_test = to_categorical(y_test_idx, num_classes=len(Config.EMOTIONS))
        
        # 訓練データと検証データに分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=Config.VALIDATION_SPLIT, 
            random_state=Config.SEED, 
            stratify=y_train_full_idx
        )
        
        print(f"-> Split into {X_train.shape[0]} training samples and {X_val.shape[0]} validation samples.")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, y_train_full_idx, y_test_idx)

# === データ前処理クラス ===
class DataPreprocessor:
    """データ前処理を管理するクラス"""
    
    @staticmethod
    def custom_preprocessing(image):
        """カスタム前処理関数"""
        # 30%の確率で「ノイズ or ブラー」のどちらかを適用
        if np.random.rand() < Config.NOISE_PROBABILITY:
            if np.random.rand() < 0.5:
                # ガウシアンノイズ
                noise = np.random.normal(loc=0.0, scale=Config.NOISE_SCALE, size=image.shape)
                image = image + noise
                image = np.clip(image, 0, 255)
            else:
                # ガウシアンブラー（カーネルサイズ3,5,7）
                ksize = random.choice([3, 5, 7])
                image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # 明るさ補正（50%）
        if np.random.rand() < Config.BRIGHTNESS_PROBABILITY:
            factor = np.random.uniform(*Config.BRIGHTNESS_RANGE)
            image = np.clip(image * factor, 0, 255)

        # ResNet50の前処理
        return preprocess_input(image)
    
    @staticmethod
    def create_data_generator():
        """データ拡張用のImageDataGeneratorを作成"""
        return ImageDataGenerator(
            preprocessing_function=DataPreprocessor.custom_preprocessing,
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

# === モデルクラス ===
class EmotionClassifier:
    """感情認識モデルを管理するクラス"""
    
    def __init__(self):
        self.model = None
        self.base_model = None
    
    def build_model(self):
        """モデルを構築"""
        input_tensor = Input(shape=Config.IMG_SHAPE)
        base = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
        base.trainable = False  # まずはベースモデルを凍結
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, kernel_regularizer=l2(0.008))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, kernel_regularizer=l2(0.007))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(len(Config.EMOTIONS), activation='softmax')(x)
        
        self.model = Model(inputs=base.input, outputs=output)
        self.base_model = base
        return self.model, self.base_model

# === 学習率スケジューラークラス ===
class LearningRateSchedulers:
    """学習率スケジューラーを管理するクラス"""
    
    @staticmethod
    def lr_schedule_stage1(epoch, lr):
        """ステージ1の学習率スケジュール"""
        if epoch < 7:
            return 2e-4
        elif epoch < 13:
            return 1e-4
        else:
            return 5e-5
    
    @staticmethod
    def create_one_cycle_lr(total_steps):
        """OneCycle学習率スケジューラーを作成"""
        return OneCycleLR(
            max_lr=Config.STAGE2_MAX_LR, 
            total_steps=total_steps, 
            pct_start=0.3, 
            min_lr=Config.STAGE2_MIN_LR
        )

# OneCycleLRコールバッククラス（既存のまま）
class OneCycleLR(Callback):
    def __init__(self, max_lr, total_steps, pct_start=0.3, min_lr=1e-6):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.min_lr = min_lr
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        pct = self.step / self.total_steps
        if pct < self.pct_start:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (pct / self.pct_start)
        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * ((pct - self.pct_start) / (1 - self.pct_start))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.step += 1

# === 結果保存クラス ===
class ResultSaver:
    """結果保存を管理するクラス"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_csv_files()
    
    def setup_csv_files(self):
        """CSVファイルを初期化"""
        self.csv_log_path = os.path.join(self.results_dir, f'tuning_log_{self.session_timestamp}.csv')
        self.emotion_results_csv_path = os.path.join(self.results_dir, f'emotion_detailed_results_{self.session_timestamp}.csv')
        
        # メインログファイル
        with open(self.csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'run_timestamp'])
        
        # 感情ごとの詳細結果ファイル
        with open(self.emotion_results_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['emotion', 'precision', 'recall', 'f1_score', 'support', 'weight', 'run_timestamp'])
    
    def save_emotion_results(self, report, class_weights):
        """感情ごとの結果を保存"""
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(self.emotion_results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for emotion in Config.EMOTIONS:
                if emotion in report:
                    precision = report[emotion]['precision']
                    recall = report[emotion]['recall']
                    f1_score = report[emotion]['f1-score']
                    support = report[emotion]['support']
                    weight = class_weights[Config.EMOTIONS.index(emotion)]
                    writer.writerow([emotion, precision, recall, f1_score, support, weight, run_timestamp])
    
    def save_main_results(self, val_acc, val_loss, test_acc, test_loss):
        """メイン結果を保存"""
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([val_acc, val_loss, test_acc, test_loss, run_timestamp])
    
    def save_model(self, model, param_str=""):
        """モデルを保存"""
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(self.results_dir, f'model_{param_str}_{run_timestamp}.h5')
        model.save(model_filename)
        print(f"Model saved to {model_filename}")
        return model_filename
    
    def save_plots(self, history, confusion_matrix_data, y_test_idx, y_pred, param_str=""):
        """プロットを保存"""
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 混合行列生成
        cm = confusion_matrix(y_test_idx, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(Config.EMOTIONS))
        plt.xticks(tick_marks, Config.EMOTIONS, rotation=45)
        plt.yticks(tick_marks, Config.EMOTIONS)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        confusion_matrix_filename = os.path.join(self.results_dir, f'cm_{param_str}_{run_timestamp}.png')
        plt.savefig(confusion_matrix_filename)
        plt.close()
        print(f"Confusion matrix saved to {confusion_matrix_filename}")
        
        # 学習曲線生成
        training_history = history.history
        plt.figure(figsize=(12, 5))
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
        plot_filename = os.path.join(self.results_dir, f'history_{param_str}_{run_timestamp}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Learning curves saved to {plot_filename}")

# === メイン実行関数 ===
def main():
    """メイン実行関数"""
    print(f"\n{'='*60}")
    print(f"学習開始")
    print(f"{'='*60}\n")
    
    # 初期設定
    set_seeds()
    setup_gpu()
    
    # データ読み込み
    data_loader = DataLoader(Config.DATA_ROOT)
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     y_train_full_idx, y_test_idx) = data_loader.load_all_data()
    
    # データ前処理
    X_test_processed = preprocess_input(X_test.copy())
    X_val_processed = preprocess_input(X_val.copy())
    
    # データ拡張
    train_datagen = DataPreprocessor.create_data_generator()
    train_gen = train_datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE, shuffle=True, seed=Config.SEED)
    steps_per_epoch = math.ceil(len(X_train) / Config.BATCH_SIZE)
    
    # 結果保存設定
    results_dir = 'results_resnet'
    result_saver = ResultSaver(results_dir)
    
    # メモリクリア
    tf.keras.backend.clear_session()
    gc.collect()
    
    # モデル構築
    emotion_classifier = EmotionClassifier()
    model, base_model = emotion_classifier.build_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.012)
    
    # クラス重み設定
    class_weights = {i: 1.0 for i in range(len(Config.EMOTIONS))}
    print("Applying custom class weights:", class_weights)
    
    # Stage 1: 転移学習
    print("\n--- Stage 1: Training Head ---")
    model.compile(optimizer=Adam(learning_rate=Config.STAGE1_INITIAL_LR), loss=loss_fn, metrics=['accuracy'])
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.STAGE1_EPOCHS,
        validation_data=(X_val_processed, y_val),
        callbacks=[LearningRateScheduler(LearningRateSchedulers.lr_schedule_stage1)],
        verbose=1
    )
    
    # Stage 2: ファインチューニング
    print("\n--- Stage 2: Fine-tuning ---")
    base_model.trainable = True
    freeze_until = 'conv4_block1_out'
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == freeze_until:
            set_trainable = True
        layer.trainable = set_trainable
    
    print(f"Fine-tuning from layer: {freeze_until}")
    
    model.compile(optimizer=Adam(learning_rate=Config.STAGE2_INITIAL_LR), loss=loss_fn, metrics=['accuracy'])
    
    total_steps = steps_per_epoch * Config.STAGE2_EPOCHS
    one_cycle = LearningRateSchedulers.create_one_cycle_lr(total_steps)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.STAGE2_EPOCHS,
        class_weight=class_weights,
        validation_data=(X_val_processed, y_val),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            one_cycle
        ],
        verbose=1
    )
    
    # 評価
    print(f"\n--- Final Evaluation ---")
    val_loss, val_acc = model.evaluate(X_val_processed, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"  Validation -> Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
    print(f"  Test       -> Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    
    # 予測
    y_pred_prob = model.predict(X_test_processed, batch_size=4)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # 分類レポート
    report = classification_report(y_test_idx, y_pred, target_names=Config.EMOTIONS, output_dict=True)
    
    # 結果保存
    result_saver.save_emotion_results(report, class_weights)
    result_saver.save_main_results(val_acc, val_loss, test_acc, test_loss)
    result_saver.save_model(model)
    result_saver.save_plots(history, None, y_test_idx, y_pred)
    
    # 結果表示
    print(f"\n--- Emotion-wise Results ---")
    for emotion in Config.EMOTIONS:
        if emotion in report:
            print(f"{emotion}: Precision={report[emotion]['precision']:.4f}, "
                  f"Recall={report[emotion]['recall']:.4f}, F1={report[emotion]['f1-score']:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test_idx, y_pred, target_names=Config.EMOTIONS))
    
    # メモリクリーンアップ
    print("\n--- Cleaning up memory ---")
    del model, base_model, history, y_pred_prob, y_pred
    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()
    print("Memory cleaned up.")
    
    print(f"\nTraining completed!")
    print(f"Check the summary at: {result_saver.csv_log_path}")
    print(f"Check the emotion-wise results at: {result_saver.emotion_results_csv_path}")

if __name__ == "__main__":
    main()
