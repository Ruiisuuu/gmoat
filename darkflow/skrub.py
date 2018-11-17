from darkflow.net.build import TFNet
import cv2
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def players_to_frames(video_filename):
    player1 = pd.DataFrame()
    player2 = pd.DataFrame()
    cap = cv2.VideoCapture(video_filename)
    cfps = int(cap.get(cv2.CAP_PROP_FPS))
    dfps = 10
    totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_length = totalframes/cfps
    print("this will take " + str(totalframes/dfps) + " seconds...")
    count = 0
    success, image = cap.read()
    while count % dfps == 0: #for each frame get json from darkflow, add to table
        frame = np.array(image)
        json = tfnet.return_predict(frame)
        framedata = json_normalize(json)
        framedata['x'] = (framedata['bottomright.x'] + framedata['topleft.x'])/2
        framedata['y'] = (framedata['bottomright.y'] + framedata['topleft.y'])/2
        print('logged players count: '+ str(count))

        framedata = framedata.drop(['bottomright.x', 'bottomright.y', 'topleft.x', 'topleft.y'], 1)
        framedata = framedata[framedata.label == 'person']
        framedata1 = framedata[framedata.y >= 250]
        framedata2 = framedata[framedata.y < 250]
        framedata1 = framedata1.loc[framedata1.groupby('label')['confidence'].idxmax()]
        framedata2 = framedata2.loc[framedata2.groupby('label')['confidence'].idxmax()]
        framedata1['frame'] = count/dfps
        framedata1['time'] = video_length*count/(dfps)**2
        framedata1 = framedata1.set_index('time')
        framedata2['frame'] = count/dfps
        framedata2['time'] = video_length*count/(dfps)**2
        framedata2 = framedata2.set_index('time')
        player1 = player1.append(framedata1)
        player2 = player2.append(framedata2)
        success, image = cap.read()
        count += 1
    player1.to_csv("player1.csv")
    player2.to_csv("player2.csv")

    return

def ball_to_frames(video_filename):
    ball = pd.DataFrame()

    cap = cv2.VideoCapture(video_filename)
    cfps = int(cap.get(cv2.CAP_PROP_FPS))
    dfps = 10
    totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_length = totalframes/cfps
    print("this will take " + str(video_length) + " seconds...")
    count = 0
    success, image = cap.read()
    while count % dfps == 0: #for each frame get json from darkflow, add to table
        frame = np.array(image)
        json = tfnet.return_predict(frame)
        framedata = json_normalize(json)
        framedata['x'] = (framedata['bottomright.x'] + framedata['topleft.x'])/2
        framedata['y'] = (framedata['bottomright.y'] + framedata['topleft.y'])/2
        print('logged balls count: '+ str(count))
        framedata = framedata.drop(['bottomright.x', 'bottomright.y', 'topleft.x', 'topleft.y'], 1)
        framedata = framedata[framedata.label == 'sports ball']
        framedata = framedata.loc[framedata.groupby('label')['confidence'].idxmax()]
        framedata['frame'] = count/dfps
        framedata['time'] = video_length*count/(dfps)**2
        framedata = framedata.set_index('time')
        ball = ball.append(framedata)
        success, image = cap.read()
        count += 1
    ball.to_csv("ball.csv")

    return

# get player coordinates
playeroptions = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.6}
tfnet = TFNet(playeroptions)
players_to_frames('test3.mp4')

# get get ball coordinates
balloptions = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.001}
tfnet = TFNet(balloptions)
ball_to_frames('test3.mp4')
