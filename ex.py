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
# <<< 修正点: 不要なインポートを削除
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tensorflow.keras.regularizers import l2
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# === 乱数シード固定 ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === 設定項目 ===
list_face_expression = ['happy', 'sad', 'neutral', 'fear', 'surprise', 'anger']
img_shape = (144, 144, 3)
batch_size = 32

data_root = '/home/eito/codes/pattern/dataset'
train_base_path = os.path.join(data_root, 'Train')
test_base_path = os.path.join(data_root, 'Test')

# === カスタム前処理関数 ===
def custom_preprocessing(image):
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            noise = np.random.normal(loc=0.0, scale=25.0, size=image.shape)
            image = np.clip(image + noise, 0, 255)
        else:
            ksize = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.9, 1.1)
        image = np.clip(image * factor, 0, 255)
    return preprocess_input(image)

# === モデル構築関数 ===
def build_model():
    input_tensor = Input(shape=img_shape)
    base = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, kernel_regularizer=l2(0.008))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(list_face_expression), activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    return model, base

# === コールバック定義 ===
def lr_schedule_stage1(epoch, lr):
    if epoch < 7: return 2e-4
    elif epoch < 13: return 1e-4
    else: return 5e-5

class OneCycleLR(Callback):
    def __init__(self, max_lr, total_steps, pct_start=0.3, min_lr=1e-6):
        super().__init__()
        self.max_lr, self.total_steps, self.pct_start, self.min_lr = max_lr, total_steps, pct_start, min_lr
        self.step = 0
    def on_train_batch_begin(self, batch, logs=None):
        pct = self.step / self.total_steps
        if pct < self.pct_start:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (pct / self.pct_start)
        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * ((pct - self.pct_start) / (1 - self.pct_start))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.step += 1

# === パラメータチューニングのメインロジック ===
weight_patterns = [(1.0, 1.0, 1.0), (1.1, 1.2, 1.1)]
results_dir = 'results_resnet'
os.makedirs(results_dir, exist_ok=True)
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_log_path = os.path.join(results_dir, f'tuning_log_{session_timestamp}.csv')

# --- データジェネレータの定義 (設計図) ---
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
    fill_mode='nearest', validation_split=0.1
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

with open(csv_log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fear_w', 'surprise_w', 'anger_w', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'run_timestamp'])

for w_fear, w_surprise, w_anger in weight_patterns:
    print(f"\n{'='*60}")
    print(f"STARTING RUN FOR WEIGHTS: fear={w_fear}, surprise={w_surprise}, anger={w_anger}")
    print(f"{'='*60}\n")
    
    tf.keras.backend.clear_session()
    gc.collect()

    # --- ジェネレータの生成 (実体化) ---
    print("--- Creating new data generators for this run ---")
    train_generator = train_datagen.flow_from_directory(
        directory=train_base_path, target_size=img_shape[:2],
        batch_size=batch_size, class_mode='categorical', subset='training', seed=SEED
    )
    validation_generator = train_datagen.flow_from_directory(
        directory=train_base_path, target_size=img_shape[:2],
        batch_size=batch_size, class_mode='categorical', subset='validation', seed=SEED
    )
    test_generator = test_datagen.flow_from_directory(
        directory=test_base_path, target_size=img_shape[:2],
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    y_test_idx = test_generator.classes

    model, base_model = build_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.012)
    steps_per_epoch = train_generator.samples // batch_size + (1 if train_generator.samples % batch_size else 0)
    validation_steps = validation_generator.samples // batch_size + (1 if validation_generator.samples % batch_size else 0)

    # --- Stage 1: Training Head ---
    print("\n--- Stage 1: Training Head ---")
    model.compile(optimizer=Adam(learning_rate=3e-4), loss=loss_fn, metrics=['accuracy'])
    model.fit(
        train_generator, steps_per_epoch=steps_per_epoch, epochs=18,
        validation_data=validation_generator, validation_steps=validation_steps,
        callbacks=[LearningRateScheduler(lr_schedule_stage1)], verbose=1
    )

    # --- Stage 2: Fine-tuning ---
    print("\n--- Stage 2: Fine-tuning ---")
    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'conv4_block1_out': set_trainable = True
        layer.trainable = set_trainable
    
    print(f"Fine-tuning from layer: conv4_block1_out")
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=['accuracy'])
    class_weights = {i: 1.0 for i in range(len(list_face_expression))}
    class_weights.update({
        list_face_expression.index('fear'): w_fear,
        list_face_expression.index('surprise'): w_surprise,
        list_face_expression.index('anger'): w_anger
    })
    print("Applying custom class weights:", class_weights)
    total_steps = steps_per_epoch * 50
    one_cycle = OneCycleLR(max_lr=7e-6, total_steps=total_steps, pct_start=0.3, min_lr=1e-6)
    history = model.fit(
        train_generator, steps_per_epoch=steps_per_epoch, epochs=50,
        class_weight=class_weights, validation_data=validation_generator, validation_steps=validation_steps,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), one_cycle], verbose=1
    )

    # --- Final Evaluation ---
    print(f"\n--- Final Evaluation for weights ({w_fear}, {w_surprise}, {w_anger}) ---")
    val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"  Validation -> Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
    print(f"  Test       -> Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # --- Reporting and Saving ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"fear{w_fear}_surp{w_surprise}_ang{w_anger}".replace('.', '_')
    print("\n--- Classification Report ---")
    print(classification_report(y_test_idx, y_pred, target_names=list_face_expression, zero_division=0))

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
    plt.ylabel('True label'), plt.xlabel('Predicted label'), plt.tight_layout()
    confusion_matrix_filename = os.path.join(results_dir, f'cm_{param_str}_{run_timestamp}.png')
    plt.savefig(confusion_matrix_filename)
    plt.close()

    training_history = history.history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_history['accuracy'], label='train_accuracy'), plt.plot(training_history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy'), plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.legend(), plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(training_history['loss'], label='train_loss'), plt.plot(training_history['val_loss'], label='val_loss')
    plt.title('Model Loss'), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(results_dir, f'history_{param_str}_{run_timestamp}.png')
    plt.savefig(plot_filename)
    plt.close()

    with open(csv_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([w_fear, w_surprise, w_anger, val_acc, val_loss, test_acc, test_loss, run_timestamp])

    model_filename = os.path.join(results_dir, f'model_{param_str}_{run_timestamp}.h5')
    model.save(model_filename)
    print(f"Model and plots saved for run with weights: {w_fear}, {w_surprise}, {w_anger}")

    # --- Final Cleanup ---
    print("\n--- Cleaning up memory ---")
    del model, base_model, history, y_pred_prob, y_pred, cm, training_history, train_generator, validation_generator, test_generator
    gc.collect()
    print("Memory cleaned up for the next run.")

print("\nAll hyperparameter tuning runs are complete!")
print(f"Check the summary at: {csv_log_path}")
