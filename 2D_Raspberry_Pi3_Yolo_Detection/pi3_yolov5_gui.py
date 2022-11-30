#!/usr/bin/python3
import cv2
import numpy as np
import time
from datetime import datetime
import PySimpleGUI as sg
import threading
import sys
from pathlib import Path

# Cheeky include for YOLO models and files
path_to_elements = Path(__file__).resolve().parent.parent / "2D_Jetson_Yolo_Detection"
sys.path.insert(0, str(path_to_elements))

from elements.yolo import OBJ_DETECTION

def drawText(frame, txt, location, color = (50, 172, 50)):
  cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

class GUIState():
    def __init__(self, window):
        self.window       = window
        self.exit         = False
        self.play_video   = False
        self.detect_frame = False
        
    def update_window_state(self):
        if self.play_video:
            if self.detect_frame:
                self.window['STATUS'].update('Object detection inference...')
            else:
                self.window['STATUS'].update('Playing video...')
                self.window['toggle_video'].update('Stop video')
                self.window['detect'].update(button_color='black on green')
        else:
            self.window['STATUS'].update('Paused')
            self.window['toggle_video'].update('Start video')
            self.window['detect'].update(button_color='black on black')

class Model():
    def __init__(self, classes_file, colors, weights):
        self.classes  = open(classes_file).read().strip().split('\n')
        self.colors   = colors
        self.detector = OBJ_DETECTION(weights, self.classes)

def video_processing(state, model):
    camera_needs_init = True
    
    while not state.exit:
        while state.play_video and not state.exit:
            if camera_needs_init:
                camera_needs_init = False
                cap = cv2.VideoCapture(0)
                window_handle = cv2.namedWindow("Pi Camera", cv2.WINDOW_AUTOSIZE)
            ret, frame = cap.read()
            if ret:
                if state.detect_frame:
                    old = time.perf_counter()
                    objs = model.detector.detect(frame)
                    inference_time = time.perf_counter() - old
                    state.detect_frame = False
                    state.update_window_state()
                    
                    print(f"Num detections: {len(objs)}\tinference time: {inference_time}")
                    
                    num_detections = {}
                    # plotting
                    for obj in objs:
                        print(f"obj: {obj}")
                        label = obj['label']
                        if label not in num_detections:
                            num_detections[label] = 1
                        else:
                            num_detections[label] += 1
                        score = obj['score']
                        [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                        color = model.colors[model.classes.index(label)]
                        frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                        frame = cv2.putText(frame, f'{label} ({str(score)})',
                                           (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX ,
                                            0.75, color, 1, cv2.LINE_AA)
                    if len(objs) > 0:
                        drawText(frame, f"Inference time: {round(inference_time,2)}s", (80, 100))
                        cv2.imshow("Detections", frame)
                        now = datetime.now()
                        current_time = now.strftime("%Y_%m_%d-%H_%M_%S")
                        det_desc = ""
                        for label, count in num_detections.items():
                            det_desc += (f"{label}{count}_")
                        filename = f"detections/{current_time}-{det_desc[:-1]}.jpg"
                        print(f'Saving as {filename}')
                        cv2.imwrite(filename, frame)
                else:
                    cv2.imshow("Pi Camera", frame)
            keyCode = cv2.waitKey(30)
            if keyCode == ord('q'):
                break
        if not camera_needs_init:
            cap.release()
            cv2.destroyAllWindows()
            camera_needs_init = True
        time.sleep(0.5)

def main():
    sg.theme('DarkAmber')
    layout = [[sg.Text('Controls', size=(50,2), font=("Arial", 20), justification='center')],
              [sg.StatusBar('Initializing...', text_color='blue', key='STATUS', size=(10,1), font=("Arial", 14), background_color='gray')],
              [sg.Button('Start video', button_color='black on black', font=("Arial", 14), key='toggle_video')],
              [sg.Button('Detect frame', button_color='black on black', font=("Arial", 14), key='detect')],
              [sg.Button('Close')]]
    window = sg.Window('Pi Detector', layout, size=(300, 300), finalize=True)
    
    # State variables
    state = GUIState(window)
    
    model = Model(path_to_elements / 'weights/coco.names', list(np.random.rand(80,3)*255), path_to_elements / 'weights/yolov5s.pt')
    
    # Initialized
    window['STATUS'].update('Ready')
    window['toggle_video'].update(button_color='black on green')
    window['detect'].update(button_color='black on green')
    
    # Video control loop
    video_thread = threading.Thread(target=video_processing, args=(state, model,))
    video_thread.start()
    
    while True:
        event, values = window.read()
        # Handle event
        if event == sg.WIN_CLOSED or event == 'Close':
            break
        elif event == 'toggle_video':
            state.play_video = not state.play_video
        elif event == 'detect':
            if state.play_video:
                state.detect_frame = True
        
        # Update GUI
        state.update_window_state()

    state.exit = True
    window.close()
    video_thread.join()
            

if __name__ == "__main__":
    main()
