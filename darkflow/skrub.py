from darkflow.net.build import TFNet
import cv2
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def clip_to_data(video_filename, subject, options):
    clipdata = pd.DataFrame()
    tfnet = TFNet(options) #create darkflow

    cap = cv2.VideoCapture(video_filename)
    cfps = int(cap.get(cv2.CAP_PROP_FPS))
    dfps = 10
    totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_length = totalframes/cfps
    count = 0
    success, image = cap.read()

    print("this will take " + str(totalframes/dfps) + " seconds...")

    while success: #for each frame get json from darkflow, add to table
        if (count % dfps == 0):
            frame = np.array(image)
            json = tfnet.return_predict(frame) #darkflow
            framedata = json_normalize(json)
            framedata['x'] = (framedata['bottomright.x'] + framedata['topleft.x'])/2
            framedata['y'] = (framedata['bottomright.y'] + framedata['topleft.y'])/2
            framedata['frame'] = count/dfps
            framedata['time'] = video_length*count/(dfps)**2
            framedata = framedata.set_index(['frame','time'])

            print('frame count: '+ str(int(count/dfps)))
            framedata = framedata.drop(['bottomright.x', 'bottomright.y', 'topleft.x', 'topleft.y'], 1)

            #take element witht the heighest probability to be our subject
            if (subject == "person"):
                framedata = framedata[framedata.label == 'person']
                framedata = framedata[framedata.y >= 250]
            elif(subject == "ball"):
                framedata = framedata[framedata.label == 'sports ball']

            framedata = framedata.loc[framedata1.groupby('label')['confidence'].idxmax()]
            clipdata = clipdata.append(framedata)
        success, image = cap.read()
        count += 1

    clipdata.to_csv(subject + ".csv")
    return clipdata

# set darkflow options
playeroptions = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.6}
balloptions = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.001}

#get coords
player = clip_to_data('test3.mp4','person', playeroptions)
ball = clip_to_data('test3.mp4','ball', balloptions)
