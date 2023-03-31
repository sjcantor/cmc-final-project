import os
import subprocess
import cv2

def change_framerate(input_video_path, fps):
    '''
    Returns: cv2.VideoCapture of new video file
    '''

    output_file_name = input_video_path[:-4] + f'_{fps}_temp.mp4'
    command = ['ffmpeg', '-i', input_video_path, '-filter:v', f'fps=fps={fps}', output_file_name]
    subprocess.call(command)

    cap = cv2.VideoCapture(output_file_name)
    os.remove(output_file_name) # cleanup

    return cap

if __name__ == '__main__':
    test_input = '../data/test_clip/test.mp4'

    cap_before = cv2.VideoCapture(test_input)
    fps_before = int(cap_before.get(cv2.CAP_PROP_FPS))

    cap_after = change_framerate(test_input, 10)
    fps_after = int(cap_after.get(cv2.CAP_PROP_FPS))

    print(f'FPS before: {fps_before} and after: {fps_after}')