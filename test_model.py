import numpy as np
from tensorflow.keras.models import load_model
from hand_landmarks import get_input_from_video_file
from output_to_mid import *
import matplotlib.pyplot as plt

import music21 as m21


def test_lstm_model(model_file='model.h5', x=None, threshold=0.5):
    # Load the model
    model = load_model(model_file)

    # Check that x is not None
    if x is None:
        raise ValueError('Input x cannot be None')

    # Check that x has the correct shape
    # if len(x.shape) != 3 or x.shape[1:] != (10, 10):
    #     raise ValueError('Input x must have shape (batch_size, 10, 10)')

    # Make predictions on the input
    y_pred = model.predict(x)

    # Apply the threshold to the predictions
    # y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Convert back to velocity [0-127]
    y_pred_vel = y_pred

    # Flatten the first two dimensions of the binary predictions
    y_pred_flat = y_pred_vel.reshape(-1, y_pred_vel.shape[-1])

    # Return the flattened binary predictions
    return y_pred_flat

# x = get_input_from_video_file('../data/test_clip/super_short.mp4')
# pred = test_lstm_model(x=x)
# binary_array_to_midi(pred, 'test_model_output.mid')
# fig, ax = plt.subplots()
# im = ax.imshow(pred[:10,:])
# plt.show()
# final_try(pred, 'yeye.mid')


# solved bad versions with this:
# https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal