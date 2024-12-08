import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import os
    


def create_siamese_model(input_shape=(224, 224, 3), num_classes_1=10, num_classes_2=2):
    def create_base_network(input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        return models.Model(inputs, x)
    
    
    base_network = create_base_network(input_shape)
    
  
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    combined = layers.Concatenate()([processed_a, processed_b])
    
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    output_1 = layers.Dense(num_classes_1, activation='softmax', name='output_1')(x)
    output_2 = layers.Dense(num_classes_2, activation='softmax', name='output_2')(x)
    
    model = models.Model(inputs=[input_a, input_b], outputs=[output_1, output_2])
    model.compile(
        optimizer='adam',
        loss={
            'output_1': 'categorical_crossentropy',
            'output_2': 'categorical_crossentropy'
        },
        metrics={
            'output_1': 'accuracy',
            'output_2': 'accuracy'
        }
    )
    
    return model

def load_and_preprocess_data(image_dir, label_file, img_height=224, img_width=224):
    if not tf.io.gfile.exists(image_dir):
        raise ValueError(f"Directory {image_dir} does not exist")
    if not tf.io.gfile.exists(label_file):
        raise ValueError(f"Label file {label_file} does not exist")

    labels_df = pd.read_csv(label_file)
 
    images_1 = []
    images_2 = []
    labels_1 = []  
    labels_2 = []  
    
    for idx, row in labels_df.iterrows():
        img1_path = tf.io.gfile.glob(f"{image_dir}/{row['filename1']}")[0]
        img1 = tf.keras.preprocessing.image.load_img(
            img1_path, target_size=(img_height, img_width)
        )
        img1 = tf.keras.preprocessing.image.img_to_array(img1) / 255.0

        img2_path = tf.io.gfile.glob(f"{image_dir}/{row['filename2']}")[0]
        img2 = tf.keras.preprocessing.image.load_img(
            img2_path, target_size=(img_height, img_width)
        )
        img2 = tf.keras.preprocessing.image.img_to_array(img2) / 255.0
        
        images_1.append(img1)
        images_2.append(img2)
        labels_1.append(row['label_name'])  
        labels_2.append(row['Gender_Name'])  

    images_1 = np.array(images_1)
    images_2 = np.array(images_2)
    labels_1 = np.array(labels_1)
    labels_2 = np.array(labels_2)
    
    label_to_idx_1 = {label: idx for idx, label in enumerate(np.unique(labels_1))}
    label_to_idx_2 = {label: idx for idx, label in enumerate(np.unique(labels_2))}
    
    numeric_labels_1 = np.array([label_to_idx_1[label] for label in labels_1])
    numeric_labels_2 = np.array([label_to_idx_2[label] for label in labels_2])
    
    one_hot_labels_1 = tf.keras.utils.to_categorical(numeric_labels_1)
    one_hot_labels_2 = tf.keras.utils.to_categorical(numeric_labels_2)
    
    indices = np.random.permutation(len(images_1))
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = [images_1[train_indices], images_2[train_indices]]
    train_labels = {
        'output_1': one_hot_labels_1[train_indices],
        'output_2': one_hot_labels_2[train_indices]
    }
    val_data = [images_1[val_indices], images_2[val_indices]]
    val_labels = {
        'output_1': one_hot_labels_1[val_indices],
        'output_2': one_hot_labels_2[val_indices]
    }
    
    return (train_data, train_labels), (val_data, val_labels), (label_to_idx_1, label_to_idx_2)

def train_model(model, train_data, train_labels, val_data, val_labels, epochs=25):
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=32
    )
    return history

def visualize_metrics(history, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['loss', 'accuracy']
    outputs = ['output_1', 'output_2']

    plt.figure(figsize=(12, 5))
    for output in outputs:
        plt.plot(history.history[f'{output}_loss'], label=f'Training (Word Recognition)')
        plt.plot(history.history[f'val_{output}_loss'], label=f'Validation (Gender Recognition)')
    
    plt.title('Model Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/losses1.png')
    plt.close()

 
    plt.figure(figsize=(12, 5))
    for output in outputs:
        plt.plot(history.history[f'{output}_accuracy'], label=f'Training (Word Recognition)')
        plt.plot(history.history[f'val_{output}_accuracy'], label=f'Validation (Gender Recognition)')
    
    plt.title('Model Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/accuracies1.png')
    plt.close()


    print("\nFinal Epoch Metrics:")
    print("=" * 50)
    
    for output in outputs:
        print(f"\n{output.upper()} Metrics:")
        print(f"Training Loss: {history.history[f'{output}_loss'][-1]:.4f}")
        print(f"Validation Loss: {history.history[f'val_{output}_loss'][-1]:.4f}")
        print(f"Training Accuracy: {history.history[f'{output}_accuracy'][-1]:.4f}")
        print(f"Validation Accuracy: {history.history[f'val_{output}_accuracy'][-1]:.4f}")

def evaluate_model(model, val_data, val_labels):
    
    y_pred = model.predict(val_data)
    
    outputs = ['output_1', 'output_2']
    metrics = {}
    
    for i, output in enumerate(outputs):
        y_pred_classes = np.argmax(y_pred[i], axis=1)
        y_true = np.argmax(val_labels[output], axis=1)

        metrics[output] = {
            'accuracy': accuracy_score(y_true, y_pred_classes),
            'f1': f1_score(y_true, y_pred_classes, average='weighted'),
            'precision': precision_score(y_true, y_pred_classes, average='weighted'),
            'recall': recall_score(y_true, y_pred_classes, average='weighted')
        }
        
    
        print(f"\n{output.upper()} Metrics:")
        print("=" * 30)
        for metric_name, value in metrics[output].items():
            print(f"{metric_name.capitalize()}: {value:.4f}")
    
    return metrics

def show_predictions(model, val_data, val_labels, label_dicts):
 
    y_pred = model.predict(val_data)

    pred_probs_1 = y_pred[0]
    pred_probs_2 = y_pred[1]
    

    y_true_1 = np.argmax(val_labels['output_1'], axis=1)
    y_true_2 = np.argmax(val_labels['output_2'], axis=1)

    label_dict_1, label_dict_2 = label_dicts
   
    idx_to_label_1 = {idx: label for label, idx in label_dict_1.items()}
    idx_to_label_2 = {idx: label for label, idx in label_dict_2.items()}

    print("\nOutput 1 - Predictions vs True Labels:")
    print("=" * 80)
    print(f"{'Predicted Label':<25} {'Confidence':<15} {'True Label':<25} {'Correct?'}")
    print("-" * 80)
    
    for probs, true in zip(pred_probs_1, y_true_1):
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        pred_label = idx_to_label_1[pred_idx]
        true_label = idx_to_label_1[true]
        correct = "✓" if pred_idx == true else "✗"
        print(f"{pred_label:<25} {confidence:>6.2f}%      {true_label:<25} {correct}")
    
    print("\nOutput 2 - Predictions vs True Labels:")
    print("=" * 80)
    print(f"{'Predicted Label':<25} {'Confidence':<15} {'True Label':<25} {'Correct?'}")
    print("-" * 80)
    
    for probs, true in zip(pred_probs_2, y_true_2):
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        pred_label = idx_to_label_2[pred_idx]
        true_label = idx_to_label_2[true]
        correct = "✓" if pred_idx == true else "✗"
        print(f"{pred_label:<25} {confidence:>6.2f}%      {true_label:<25} {correct}")


if __name__ == "__main__":
    try:
        model = create_siamese_model()
      
        (train_data, train_labels), (val_data, val_labels), label_dicts = load_and_preprocess_data(
            'Full_Dataset/', 
            'Full_Dataset/Full_Label.csv'
        )
        history = train_model(model, train_data, train_labels, val_data, val_labels)
        visualize_metrics(history)
        metrics = evaluate_model(model, val_data, val_labels)
        show_predictions(model, val_data, val_labels, label_dicts)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
