import cv2 as cv
import PySimpleGUI as sg
import numpy as np
import time


# Example of a Simple AI that converts Short Term Memory to Working Memory
# It consists of a very basic number detector using OPENCV
# templates and buffers for STM and WM.
# The working Memory buffer adds/sums the last
# detected number via the STMToWM function.
# Consult the Medium post for further details
# GUI is pysimpleGUI


# VARS:

_VARS = {'cellCount': 240,
         'gridSize': 480,  # param with opencv
         'window': False,
         'accumulator1': 0,
         'accumulator2': 0,
         'accumulator3': 0,
         'accumulator4': 0,
         'accumulator5': 0,
         'accumulator6': 0,
         'accumulator7': 0,
         'accumulator8': 0,
         'accumulator9': 0,
         'attentionThreshold': 15,
         'STM': [],
         'WM': 0}

templateThreshold = 0.38
cellSize = _VARS['gridSize'] / _VARS['cellCount']
frameWH = _VARS['gridSize']
last_recorded_time = time.time()

# PySimpleGUI:
AppFont = 'Any 16'
AppFont2 = 'Any 32'
sg.theme('DarkGrey5')
fps = 24

# Import Number Templates:

template_1_50 = np.load("STMToWM_Medium/templates/num_1.npy")
template_2_50 = np.load("STMToWM_Medium/templates/num_2.npy")
template_3_50 = np.load("STMToWM_Medium/templates/num_3.npy")
template_4_50 = np.load("STMToWM_Medium/templates/num_4.npy")
template_5_50 = np.load("STMToWM_Medium/templates/num_5.npy")
template_6_50 = np.load("STMToWM_Medium/templates/num_6.npy")
template_7_50 = np.load("STMToWM_Medium/templates/num_7.npy")
template_8_50 = np.load("STMToWM_Medium/templates/num_8.npy")
template_9_50 = np.load("STMToWM_Medium/templates/num_9.npy")


# RED Tamplates for detection:

def makeColorTemplate(template):
    color = [0, 0, 255]
    white = [255, 255, 255]
    colorTemplate = []
    # print(template)
    for index, item in enumerate(template):
        colorTemplate.append([white if x == 1 else color for x in item])
    return colorTemplate


colTemp_1 = makeColorTemplate(template_1_50)
colTemp_2 = makeColorTemplate(template_2_50)
colTemp_3 = makeColorTemplate(template_3_50)
colTemp_4 = makeColorTemplate(template_4_50)
colTemp_5 = makeColorTemplate(template_5_50)
colTemp_6 = makeColorTemplate(template_6_50)
colTemp_7 = makeColorTemplate(template_7_50)
colTemp_8 = makeColorTemplate(template_8_50)
colTemp_9 = makeColorTemplate(template_9_50)


tDims = np.shape(colTemp_1)[0]
templateW, templateH = template_1_50.shape[::-1]

# PySimpleGUI Init:
col1row1 = [[sg.Text('Raw Camera')],
            [sg.Image(filename='', key='raw')]]
col2row1 = [[sg.Text(' v Downsampled - Template : Detected (RED) ')],
            [sg.Image(filename='', key='template_1')]]

layout = [[sg.Column(col1row1), sg.Column(col2row1)],
          [sg.Text('STM Buffer:[]', font=AppFont2, key='STM'), sg.Text(
              'WM:0', font=AppFont2, key='WM'), sg.Stretch()],
          [sg.Stretch(), sg.Exit(font=AppFont)]]


window = sg.Window('OpenCV PySimpleGUI', layout,
                   resizable=True, location=(100, 300), finalize=True,
                   element_justification='r')

# OPENCV Init:
cap = cv.VideoCapture(0)

# OpenCV
capW = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
capH = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
minHW = min(capW, capH)
maxHW = max(capW, capH)
cropLeft = int((maxHW-minHW)/2)
cropTop = 0
downsamplePixels = _VARS['cellCount']


def matchTemplate(frameOutput=None, template=None):
    res = cv.matchTemplate(frameOutput, template.astype(
        np.uint8), cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= templateThreshold)
    return loc


def STMToWM(STM):
    # Insert detected number into STM
    _VARS['STM'].insert(0, STM)
    # Remove last element from STM if it's too long
    if len(_VARS['STM']) > 2:
        _VARS['STM'].pop()
    # Calculate WM
    if len(_VARS['STM']) == 2:
        _VARS['WM'] = int(sum(_VARS['STM']))
    # if  WM exists move it to the front of the STM
    # and discard the last element
    if _VARS['WM'] > 0:
        _VARS['STM'].insert(0, _VARS['WM'])
        _VARS['STM'].pop()

    # Update WM display

    window['WM'].update('WM:' + str(_VARS['WM']))


while True:                     # The PSG "Event Loop"
    event, values = window.Read(timeout=fps, timeout_key='timeout')
    if event in (None, 'Exit'):
        break

    # Frame Operations
    ret, frame = cap.read()
    height, width = 300, 300
    scaleCoords = height/downsamplePixels
    frame = frame[cropTop:cropTop+minHW, cropLeft:cropLeft+minHW]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_1 = cv.resize(frame, (200, 200),
                        interpolation=cv.INTER_NEAREST)
    temp = cv.resize(gray, (downsamplePixels, downsamplePixels),
                     interpolation=cv.INTER_LINEAR)
    thresh = cv.adaptiveThreshold(temp, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                  cv.THRESH_BINARY, 11, 2)
    pixelate = cv.resize(thresh, (downsamplePixels, downsamplePixels),
                         interpolation=cv.INTER_LINEAR)

    # Get matching locations for each template : TO GLOBAL ?
    locs_1 = matchTemplate(pixelate, template_1_50)
    locs_2 = matchTemplate(pixelate, template_2_50)
    locs_3 = matchTemplate(pixelate, template_3_50)
    locs_4 = matchTemplate(pixelate, template_4_50)
    locs_5 = matchTemplate(pixelate, template_5_50)
    locs_6 = matchTemplate(pixelate, template_6_50)
    locs_7 = matchTemplate(pixelate, template_7_50)
    locs_8 = matchTemplate(pixelate, template_8_50)
    locs_9 = matchTemplate(pixelate, template_9_50)

    # Make Frame for Template Display
    colorPixelate_01 = cv.cvtColor(pixelate, cv.COLOR_GRAY2BGR)
    colorPixelate_01[np.where((colorPixelate_01 == [0, 0, 0]).all(axis=2))] = [
        80, 80, 80]

    # colorPixelate_01[0:tDims, 0:tDims] = colTemp_2 # showTemplate

    # Time Keeping/Substitute for attention :
    current_time = time.time()
    if current_time - last_recorded_time > 2:
        if _VARS['accumulator1'] > _VARS['attentionThreshold']:
            print('ONE DETECTED !')
            STMToWM(1)
        elif _VARS['accumulator2'] > _VARS['attentionThreshold']:
            print('TWO DETECTED !')
            STMToWM(2)
        elif _VARS['accumulator3'] > _VARS['attentionThreshold']:
            print('THREE DETECTED !')
            STMToWM(3)
        elif _VARS['accumulator4'] > _VARS['attentionThreshold']:
            print('FOUR DETECTED !')
            STMToWM(4)
        elif _VARS['accumulator5'] > _VARS['attentionThreshold']:
            print('FIVE DETECTED !')
            STMToWM(5)
        elif _VARS['accumulator6'] > _VARS['attentionThreshold']:
            print('SIX DETECTED !')
            STMToWM(6)
        elif _VARS['accumulator7'] > _VARS['attentionThreshold']:
            print('SEVEN DETECTED !')
            STMToWM(7)
        elif _VARS['accumulator8'] > _VARS['attentionThreshold']:
            print('EIGHT DETECTED !')
            STMToWM(8)
        elif _VARS['accumulator9'] > _VARS['attentionThreshold']:
            print('NINE DETECTED !')
            STMToWM(9)

        last_recorded_time = current_time
        _VARS['accumulator1'] = 0
        _VARS['accumulator2'] = 0
        _VARS['accumulator3'] = 0
        _VARS['accumulator4'] = 0
        _VARS['accumulator5'] = 0
        _VARS['accumulator6'] = 0
        _VARS['accumulator7'] = 0
        _VARS['accumulator8'] = 0
        _VARS['accumulator9'] = 0

    if len(locs_1[0]) > 0:
        _VARS['accumulator1'] += 1
        for pt in zip(*locs_1[::-1]):
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_1

    if len(locs_2[0]) > 0:
        _VARS['accumulator2'] += 1
        for pt in zip(*locs_2[::-1]):
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_2

    if len(locs_3[0]) > 0:  # Detect 3
        _VARS['accumulator3'] += 1
        for pt in zip(*locs_3[::-1]):  # Detect 3
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_3  # Detect 3

    if len(locs_4[0]) > 0:  # Detect 4
        _VARS['accumulator4'] += 1
        for pt in zip(*locs_4[::-1]):  # Detect 4
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_4  # Detect 4

    if len(locs_5[0]) > 0:  # Detect 5
        _VARS['accumulator5'] += 1
        for pt in zip(*locs_5[::-1]):  # Detect 5
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_5  # Detect 5

    if len(locs_6[0]) > 0:  # Detect 6
        _VARS['accumulator6'] += 1
        for pt in zip(*locs_6[::-1]):  # Detect 6
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_6  # Detect 6

    if len(locs_7[0]) > 0:  # Detect 7
        _VARS['accumulator7'] += 1
        for pt in zip(*locs_7[::-1]):  # Detect 7
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_7  # Detect 7

    if len(locs_8[0]) > 0:  # Detect 8
        _VARS['accumulator8'] += 1
        for pt in zip(*locs_8[::-1]):  # Detect 8
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_8  # Detect 8

    if len(locs_9[0]) > 0:  # Detect 9
        _VARS['accumulator9'] += 1
        for pt in zip(*locs_9[::-1]):  # Detect 9
            # Insert colorTemplate
            colorPixelate_01[pt[1]:pt[1]+tDims,
                             pt[0]:pt[0]+tDims] = colTemp_9  # Detect 9

    window['raw'].Update(
        data=cv.imencode('.png', frame_1)[1].tobytes())
    window['template_1'].Update(
        data=cv.imencode('.png', colorPixelate_01)[1].tobytes())
    window['STM'].update('STM:' + str(_VARS['STM']))

    locs_1 = []
    locs_2 = []
    locs_3 = []
    locs_4 = []
    locs_5 = []
    locs_6 = []
    locs_7 = []
    locs_8 = []
    locs_9 = []

window.close()
