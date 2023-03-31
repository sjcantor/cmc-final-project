# inspired from https://www.geeksforgeeks.org/python-writing-to-video-with-opencv/
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import argparse
import os
import ffmpeg
import numpy as np
from tqdm import tqdm
from change_framerate import change_framerate
import pandas as pd
import glob
import pretty_midi
import pdb

from hand_landmarks import extract_relevant_landmarks


def compress_video_output(filename, output_filename):
    ''' cv2.VideoWriter() writes a HUGE file...
        We can compress this using ffmpeg '''
    print(f'-----INPUT: {filename}')
    input = ffmpeg.input(filename)
    output = ffmpeg.output(input, output_filename, format='mp4') 
    output.run()

    # cleanup
    os.remove(filename)

    return output_filename

def get_midi_from_file(filename):
    print(f'Getting midi info from {filename}...')
    pm = pretty_midi.PrettyMIDI(filename)
    print(len(pm.get_piano_roll(100)[0]))
    return pm

def check_video_validity(cap):
    '''
    Checks for anything that makes the entire video not valid.
    For now, the only case is >2 hands.
    '''
    # TODO - this is very inefficient, is it needed? Is there a better way?
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
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) > 2:
                    return False
    return True
            


def track_video(input_video_file, downsample, clip_length, midi, output_dir):
    print(f'Tracking video {input_video_file}...')
    cap = None
    if downsample is not None:
        cap = change_framerate(input_video_path=input_video_file, fps=downsample)
    else:
        cap = cv2.VideoCapture(input_video_file)

    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    if framespersecond != downsample:
        print('Something went wrong with the downsampling')
    # pdb.set_trace(header='after downsample')
    # valid = check_video_validity(cap)
    # if not valid:
    #     print('This video is not valid, skipping...')
    #     return None, None
    # pdb.set_trace(header='after validity')
    


    # Changing clip length to represent number of frames per clip
    # (instead of just seconds)
    # clip_length = clip_length * framespersecond

    # OpenCV > 3.1
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fingertips data
    # dim0 = int(total_frames_count / 10) + 1 # will delete the last one later...
    # dim1 = clip_length
    # fingers_dim2 = 10 # number of fingers we're tracking
    # fingertips = np.zeros((dim0, dim1, fingers_dim2))
    fingertips = []
    # fingertips shape: (number of clips, frames per clip, 10 fingers)

    # MIDI data
    # midi_dim2 = 88 # number of midi notes per frame
    # midi_roll = np.zeros((dim0, dim1, midi_dim2))
    midi_roll = []
    # midi_roll shape: (number of clips, frames per clip, 88 piano notes)

    frame = 0
    # current_frame_multiple = 0
    current_arr_index = -1

    midi_length = len(midi.get_piano_roll(framespersecond)[0])

    midi_streak = 0
    fingertips_streak = []
    midi_roll_streak = []


    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while frame < midi_length: 

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            piano_roll = midi.get_piano_roll(framespersecond)[21:109, frame] # 21:109 is MIDI range for piano keys 1:89
            if len(np.nonzero(piano_roll)[0]) > 0:
                midi_streak += 1

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
            
                if results.multi_hand_landmarks:
                    # Fingers
                    extracted = extract_relevant_landmarks(results.multi_hand_landmarks)
                    if extracted == 'STOP ANALYZING ENTIRE VIDEO': # too many hands!
                        break
                    if len(extracted) == 10: # streak also dependent on tracking both hands
                        fingertips_streak.append(extracted)
                        midi_roll_streak.append(piano_roll.tolist())
                    else: # break streak if not 2 hands
                        fingertips_streak = []
                        midi_roll_streak = []
                        midi_streak = 0
                else:
                    # print(f"Didn't detect any landmarks for video {input_video_file} frame {frame}...")
                    # TODO - extract anyways, is this okay to do?
                    # extracted = extract_relevant_landmarks(None)
                    fingertips_streak = []
                    midi_roll_streak = []
                    midi_streak = 0
                
                if midi_streak == clip_length:
                    # print(f'Reached {clip_length} in a row!')
                    midi_streak = 0
                    fingertips.append(fingertips_streak)
                    midi_roll.append(midi_roll_streak)
                    fingertips_streak = []
                    midi_roll_streak = []

            else:
                fingertips_streak = []
                midi_roll_streak = []
                midi_streak = 0

            frame += 1


            # TODO - Right now there is no overlap

    cap.release()
    print('cap released')

    # The last row won't have a full 10 frames, just discard it...
    # fingertips = fingertips[:-1]
    # midi_roll = midi_roll[:-1]

    fingertips = np.array(fingertips)
    midi_roll = np.array(midi_roll)



    return fingertips, midi_roll

def create_dataset(downsample, output_dir, clip_length):

    DIR = '/Users/sam/upf/thesis/thesis-testing'

    FULL_PLAYLIST = os.path.join(DIR, 'data/full_playlist/trimmed_videos')
    if output_dir is None:
        output_dir = os.path.join(DIR, 'data/full_playlist/only_hands')
    MIDI_DIR = os.path.join(DIR, 'data/midis')

    # Match dimensions with video numpy arrays
    dim0 = 1 # will concatenate to this dimension then later remove the first row of zeros
    dim1 = clip_length # TODO - for now with new streak implementation
    fingers_dim2 = 10 # number of fingers we're tracking
    fingertips = np.zeros((dim0, dim1, fingers_dim2))

    # MIDI data
    midi_dim2 = 88 # number of midi notes per frame
    midi_roll = np.zeros((dim0, dim1, midi_dim2))

    match_count = 0
    no_match_count = 0
    for filename in tqdm(os.listdir(FULL_PLAYLIST)):
        f = os.path.join(FULL_PLAYLIST, filename)
        if os.path.isfile(f) and f.endswith('.mp4'):
            # Find MIDI file
            match = glob.glob(MIDI_DIR + '/*' + filename[:-4] + '*.mid')
            if match:
                match_count += 1
                print(f'{match_count} * Tracking hands for video: {filename}')

                # Get MIDI file
                midi = get_midi_from_file(match[0])

                # Get fingertips and MIDI frames for a single video, split into clips
                video_fingertips, video_midi_roll = track_video(f, downsample, clip_length, midi, output_dir)
                if video_fingertips is None: # video not valid
                    pass
                elif len(video_fingertips.shape) == 3 and len(video_midi_roll.shape) == 3:
                    print(f'fingertips shape: {video_fingertips.shape} \nmidi shape: {video_midi_roll.shape}')
                    print(f'big array shape before: {fingertips.shape}, {midi_roll.shape}')
                    fingertips = np.concatenate((fingertips, video_fingertips))
                    midi_roll = np.concatenate((midi_roll, video_midi_roll))
                    print(f'big array shape after: {fingertips.shape}, {midi_roll.shape}')
                else:
                    print(f'Shape is weird? \ningertips shape: {video_fingertips.shape} \nmidi shape: {video_midi_roll.shape}')
            else:
                no_match_count += 1
                print(f'{no_match_count} * no match for {filename}, not adding to dataset')
    print(f'fingertips shape: {fingertips.shape} \nmidi shape: {midi_roll.shape}')

    # Remove first dummy row
    fingertips = fingertips[1:]
    midi_roll = midi_roll[1:]
    np.save('fingertips.npy', fingertips)
    np.save('midiroll.npy', midi_roll)
    return fingertips, midi_roll

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--downsample', default=10, help='downsample to given fps')
parser.add_argument('-o', '--output_dir', default=None, help='output folder to save video(s)')
parser.add_argument('-c', '--clip_length', default=10, help='separate videos into clips of c seconds')

if __name__ == '__main__':
    # Arguments
    args = parser.parse_args()
    print(f'args: {args}')
    # TODO - is there a cleaner way to write this next line?
    downsample, output_dir, clip_length = int(args.downsample), args.output_dir, int(args.clip_length)

    print(f'Creating dataset for all videos...')
    create_dataset(downsample, output_dir, clip_length)