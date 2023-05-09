import cv2
import time
import numpy as np
import argparse
from imutils.video import VideoStream
from imutils.video import FPS
import imutils

# load the COCO class names
with open('input/object_detection_classes_coco.txt', 'r') as f:   #object_detection_classes_coco.txt
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(100, 255, size=(len(class_names), 3)) #0,255

# load the DNN model
model = cv2.dnn.readNet(model='input/frozen_inference_graph.pb',
                        config='input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

# capture the video
#cap = cv2.VideoCapture('../../input/video_2.mov')
cap = cv2.VideoCapture(-1)
#cap = VideoStream(usePiCamera=False).start()
time.sleep(2.0)
# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# create the `VideoWriter()` object
out = cv2.VideoWriter('outputs/video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

# set up argparse for type of object to detect
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--object", required=True,
	help="name of object to detect")
args = vars(ap.parse_args())

# detect objects in each frame of the video
while cap.isOpened():
    ret,frame = cap.read()
    #frame = imutils.resize(frame, width=1000)
    if ret:
        image = frame
        image_height, image_width, _ = image.shape
        # create blob from image
        blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                     swapRB=True)
        # start time to calculate FPS
        start = time.time()
        model.setInput(blob)
        output = model.forward()        
        # end time after detection
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # extract the confidence of the detection
            confidence = detection[2]
            # draw bounding boxes only if the detection confidence is above...
            # ... a certain threshold, else skip 
            #print(confidence)
            if confidence > .4:
                # get the class id
                class_id = detection[1]
                # map the class id to the class 
                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                # put the FPS text on top of the frame
                cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #draw bounding boxes only around detected objects of the class name "person"
                if class_name == args["object"]:
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=4)
                    # put the class name text on the detected object
                    cv2.rectangle(image, (int(box_x),int(box_y)), (int(box_width),int(box_y+35)), color, -1) #int((box_y+((box_height-box_y)/7)))
                    cv2.putText(image, f"{class_name} {round(confidence*100,1)}", (int(box_x), int(box_y + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
                    cv2.putText(image, "Human detected", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # draw a circle at the center point of the bounding boxes
                    cv2.circle(image, (int(box_x+(box_width-box_x)/2), 
                        int(box_y+(box_height-box_y)/2)), 7, (0,0,255), -1)
                    
#                elif class_name != "person":
#                    cv2.putText(image, f"{fps:.2f} FPS     No humans detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
