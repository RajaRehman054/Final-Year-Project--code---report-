from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
from nsfw_detector import predict
import numpy as np

app = Flask(__name__)
CORS(app)

model = predict.load_model('./nsfw_mobilenet2.224x224.h5')
threshP = 0.95
threshS = 0.95

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
            result = predict.classify(model, './image.png')
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

# Define a route for the root endpoint
@app.route('/' , methods = ['POST'])
def hello_world():
    print("here")
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    file.save('video.mp4')
    notSave  = is_video_nsfw('./video.mp4')
    return jsonify({'notSafe': notSave})


# Run the application
if __name__ == '__main__':
    app.run(debug=True)