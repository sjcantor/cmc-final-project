# from the tutorial https://google.github.io/mediapipe/solutions/hands.html

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from hand_landmarks import get_input_from_video_file
from test_model import test_lstm_model
from output_to_mid import binary_array_to_midi
import matplotlib.pyplot as plt
from rand_select import randomly_select

import os
import ffmpeg

def compress_video_output(filename):
    ''' cv2.VideoWriter() writes a HUGE file...
        We can compress this using ffmpeg '''
        
    input = ffmpeg.input(filename)
    new_output_name = +filename[:-4]+'_tracked.mp4' # TODO - clean using OS
    output = ffmpeg.output(input, new_output_name, format='mp4') 
    output.run()

    # cleanup
    os.remove(filename)

    return new_output_name


def track():
    # For webcam input:
    cap = cv2.VideoCapture(0)

    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    print(framespersecond)
    size = (int(cap.get(3)), int(cap.get(4)))

    # TODO - is this cleaner using OS?
    output_path = 'live_tracked.avi' #remove ".mp4" and add a little bit

    output = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        framespersecond,
        size
    )

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            output.write(image)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                cap.release()
                break
    cap.release()
    return output_path

if __name__ == '__main__':
    print('tracking')
    output_path = track()
    print('parsing video')
    x = get_input_from_video_file(output_path, DEMO=True, clip_length=10)
    print('predicting')
    pred = test_lstm_model(x=x)

    # fig, ax = plt.subplots()
    # im = ax.imshow(pred.T)
    # plt.show()

    # enchanced = randomly_select(pred, 5)

    fig, ax = plt.subplots()
    im = ax.imshow(pred.T)
    plt.show()

    midi_name = 'live_demo.mid'
    print('saving to midi: {midi_name}')
    binary_array_to_midi(pred, midi_name)
    print('done')
    
