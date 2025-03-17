import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data():
    with open("D:/Programs/Projects/Lazy bot/data/annotations/processed_data.json", "r") as f:
        data = json.load(f)

    X = []
    y = []
    for entry in data:
        spine_angle = entry["spine_angle"]
        label = entry["label"]
        X.append([spine_angle])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_model():
    X, y, scaler = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_dim=1, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save model and scaler
    model.save("D:/Programs/Projects/Lazy bot/models/trained/posture_model.h5")
    print("Model saved to D:/Programs/Projects/Lazy bot/models/trained/posture_model.h5")
    with open("D:/Programs/Projects/Lazy bot/models/trained/scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    train_model()