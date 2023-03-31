import mido
import numpy as np
import pdb

def binary_array_to_midi(binary_array, filename):
    print(f'input size: {binary_array.shape}')
    # Set up MIDI file with a single track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add a program change event to specify the instrument
    # program_change = mido.Message('program_change', program=0, time=0)
    # track.append(program_change)

    # Iterate over the binary array and create MIDI events
    for time, state in enumerate(binary_array):
        for note, value in enumerate(state):
            if value == 1:
                note_on = mido.Message('note_on', note=note+21, velocity=64, time=time)
                track.append(note_on)
            elif value == 0:
                note_off = mido.Message('note_off', note=note+21, velocity=0, time=time)
                track.append(note_off)


    # Save the MIDI file
    mid.save(filename)

def array2midi(arr, tempo=500000):
    """Converts a numpy array to a MIDI file"""
    # Adapted from: https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    new_arr = np.concatenate([np.array([[0] * 128]), np.array(arr)], axis=0)
    changes = new_arr[1:] - new_arr[:-1]
    midi_file = mido.MidiFile()  # create a midi file with an empty track
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    last_time = 0
    for ch in changes:  # add difference in the empty track
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=n, velocity=50, time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    return midi_file


def pad(arr):
    start_array = np.zeros((arr.shape[0], 21)) * 21
    end_array = np.zeros((arr.shape[0], 19)) * 19

    # concatenate the start array, input array, and end array along the second dimension
    output_array = np.concatenate((start_array, arr, end_array), axis=1)
    return output_array

def final_try(arr, output_file):
    padded = pad(arr)
    mid = array2midi(padded)
    mid.save(output_file)




# input_array = np.random.randint(2, size=(100, 88))
# binary_array_to_midi(input_array, 'pls_work.mid')