import PySimpleGUI as sg

import cv2

import numpy as np

from contoursBackgroundRemoval import removeBG
from deepNNBackgroundRemoval import initiateModel
from deepNNBackgroundRemoval import segment



def main():

    sg.theme("LightGreen")


    # Define the window layout
    
    webcam_column = [
    
        [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],

        [sg.Image(filename="", key="-IMAGE-")],
    
    ]
    
    controls_column = [
    
        [sg.Radio("None", "Radio", True, size=(10, 1))],

        [

            sg.Radio("threshold", "Radio", size=(10, 1), key="-THRESH-"),

            sg.Slider(

                (0, 255),

                128,

                1,

                orientation="h",

                size=(40, 15),

                key="-THRESH SLIDER-",

            ),

        ],

        [

            sg.Radio("Contour BG remove", "Radio", size=(10, 1), key="-CANNY-"),

            sg.Slider(

                (0, 255),

                128,

                1,

                orientation="h",

                size=(20, 15),

                key="-CANNY SLIDER A-",

            ),

            sg.Slider(

                (0, 255),

                128,

                1,

                orientation="h",

                size=(20, 15),

                key="-CANNY SLIDER B-",

            ),
            sg.Slider(

                (1, 21),

                3,

                1,

                orientation="h",

                size=(20, 15),

                key="-GAUSS SLIDER-",

            ),

        ],

        [

            sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),

            sg.Slider(

                (1, 11),

                1,

                1,

                orientation="h",

                size=(40, 15),

                key="-BLUR SLIDER-",

            ),

        ],

        [

            sg.Radio("hue", "Radio", size=(10, 1), key="-HUE-"),

            sg.Slider(

                (0, 225),

                0,

                1,

                orientation="h",

                size=(40, 15),

                key="-HUE SLIDER-",

            ),

        ],

        [

            sg.Radio("enhance", "Radio", size=(10, 1), key="-ENHANCE-"),

            sg.Slider(

                (1, 255),

                128,

                1,

                orientation="h",

                size=(40, 15),

                key="-ENHANCE SLIDER-",

            ),

        ],
        [sg.Button("deep BG removal", size=(10, 1))],

        [sg.Button("Exit", size=(10, 1))],

    
    ]

    layout = [
        [
            sg.Column(webcam_column),
            sg.VSeperator(),
            sg.Column(controls_column),
        ]
        
    ]


    # Create the window and show it without the plot

    window = sg.Window("OpenCV Integration", layout, location=(50, 50))
    dlab = initiateModel()

    #cap = cv2.VideoCapture(0)
    #cap = cv2.imread("C:\ImageApp\testimg.png")

    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:

            break
        
        if event == "deep BG removal":
            print("begin segmentation")
            frame = segment(dlab, frame)#TODO: figure out why imshow works but pysimplegui doesn't update
            print("complete segmentation")
            

        #ret, frame = cap.read()
        frame = cv2.imread("./testimg.png") 

        if values["-THRESH-"]:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]

            frame = cv2.threshold(

                frame, values["-THRESH SLIDER-"], 255, cv2.THRESH_BINARY

            )[1]

        elif values["-CANNY-"]:

            frame = removeBG(

                frame, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"], values["-GAUSS SLIDER-"]

            )

        elif values["-BLUR-"]:

            frame = cv2.GaussianBlur(frame, (21, 21), values["-BLUR SLIDER-"])

        elif values["-HUE-"]:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            frame[:, :, 0] += int(values["-HUE SLIDER-"])

            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        elif values["-ENHANCE-"]:

            enh_val = values["-ENHANCE SLIDER-"] / 40

            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            lab[:, :, 0] = clahe.apply(lab[:, :, 0])

            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        '''
        elif values["-DEEPNP-"]:
            print("begin segmentation")
            frame = segment(dlab, frame)
            print("complete segmentation")
        '''

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        window["-IMAGE-"].update(data=imgbytes)


    window.close()


main()