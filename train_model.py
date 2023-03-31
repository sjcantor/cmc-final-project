import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

loss = []
accuracy = []

def train_lstm_model(x_train_file='x_train.npy', y_train_file='y_train.npy',
                     epochs=100, batch_size=32, model_file='model.h5', clip_length=10):

    # Load the data
    X = np.load(x_train_file)
    y = np.load(y_train_file)

    # print(f'shapes: x: {X.shape}, y: {y.shape}')

    # Normalize y to [0-1]
    non_zero_indices = np.nonzero(y)
    # print(f'y: {non_zero_indices}')

    y_norm = (y - y.min()) / (y.max() - y.min())


    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(X, y_norm, test_size=0.2)

    # Define the model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(clip_length, 10)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(88, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define the checkpoint to save the best model weights
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(x_val, y_val), callbacks=[checkpoint])
    
    loss.append(history.history['loss'])
    accuracy.append(history.history['accuracy'])

    # Return the trained model
    return model

if __name__ == '__main__':
    print('Training LSTM model...')
    train_lstm_model('./fingertips.npy', './midiroll.npy')

    # Plot the loss
    plt.plot(loss)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.show()

    # Plot the accuracy
    plt.plot(accuracy)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')
    plt.show()