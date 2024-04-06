import cv2
from nsfw_detector import predict
import numpy as np

model = predict.load_model('./nsfw_mobilenet2.224x224.h5')
threshP = 0.3
threshS = 0.4

def is_video_nsfw(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    is_nsfw = False
    while(cap.isOpened()):
        # Capture frame-by-frame
        frame_num += 1
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, ((224, 224)[1], (224, 224)[0]))
            cv2.imwrite('./image.png', frame)
            result = predict.classify(model, './test.mp4')
            print(result)
            if result['./image.png']['porn'] > threshP:
                # video is NSFW
                is_nsfw = True
                break
            if result['./image.png']['sexy'] > threshS:
                # video is NSFW
                is_nsfw = True
                break

        # Break the loop
        else:
            break
    cap.release()
    return is_nsfw

let  = is_video_nsfw('./ptest.mp4')
print(let)



# from nsfw_detector import predict
# model = predict.load_model('./nsfw_mobilenet2.224x224.h5')

# # Predict single image
# let = predict.classify(model, './image.png')
# print(let)
