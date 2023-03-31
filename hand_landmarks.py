import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import math

# The distance for Rousseau hands
HAND_DISTANCE_CONSTANT = 0.14354457979708476

def extract_relevant_landmarks(landmarks, DEMO=False):
    '''
    We love simple helper functions!
    Takes: hands.process(image).multi_hand_landmarks
    Landmarks to extract: each fingertip (5 per hand)
    '''
    global HAND_DISTANCE_CONSTANT
    fingertips = []

    one_time = DEMO

    if landmarks is None:
        if DEMO:
            print(f'---------- Uh oh! This frame has 0 hands...')
            fingertips.extend([0]*5*2) # TODO - is this a good solution? add padding zeros
        else:
            return fingertips
    
    for hand_landmarks in landmarks:
        # Extract iterable list from weird object thing
        lms = hand_landmarks.landmark
        # Fingertips are index 4, 8, 12, 16, and 20
        fingertips.extend([
            lms[4].x,
            lms[8].x,
            lms[12].x,
            lms[16].x,
            lms[20].x
        ])
        if one_time:
            x_distance = abs(lms[0].x - lms[5].x)
            y_distance = abs(lms[0].y - lms[5].y)
            HAND_DISTANCE_CONSTANT = math.sqrt(x_distance**2 + y_distance**2)
            one_time = False # to only run this once

    if DEMO:
        print(f'len {len(landmarks)}')
        if len(landmarks) != 2:
            print(f'---------- Uh oh! This frame has {len(landmarks)} hands...')
            if len(landmarks) < 2:
                fingertips.extend([0]*5*len(landmarks)) # TODO - is this a good solution? add padding zeros (doesn't consider left/right hand)
            elif len(landmarks) > 2:
                print(f'---- Shit... it has {len(landmarks)} hands... this whole video should be ignored')
                # return 'STOP ANALYZING ENTIRE VIDEO'
                fingertips = fingertips[:10]

    # normalize by hand size
    normalized_fingertips = [i / HAND_DISTANCE_CONSTANT for i in fingertips]
    return normalized_fingertips

def get_input_from_video_file(video_file, DEMO=False, clip_length=10):
    cap = cv2.VideoCapture(video_file)
    # OpenCV > 3.1
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fingertips data
    dim0 = int(total_frames_count / 10) + 1 # will delete the last one later...
    dim1 = clip_length # TODO make sure this still works if you add cliplength*fps
    fingers_dim2 = 10 # number of fingers we're tracking
    fingertips = np.zeros((dim0, dim1, fingers_dim2))
    print(fingertips.shape)

    frame = 0
    current_arr_index = -1

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while True: #cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            if frame % dim1 == 0: # TODO - anotha one
                current_arr_index += 1

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                # Fingers
                extracted = extract_relevant_landmarks(results.multi_hand_landmarks, DEMO)
                fingertips[current_arr_index][frame%dim1] = extracted # TODO - hardcoded 100
            frame += 1
    cap.release()
    # The last row won't have a full 10 frames, just discard it...
    fingertips = fingertips[:-1]
    return fingertips