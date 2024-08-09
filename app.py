from flask import Flask, render_template, request, jsonify, url_for, Response, session, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import os, io, cv2, base64, csv
from ultralytics import YOLO
import numpy as np
import pandas as pd
from flask_socketio import SocketIO
from flask_cors import CORS
import yt_dlp as youtube_dl
import threading
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for the app
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory for uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # Allowed file extensions
app.config['SECRET_KEY'] = os.urandom(24)  # Secret key for session management


video_stream_active = True

# Load YOLO model for detection
model = YOLO('models/yolov8m-detection.pt')

# Function to check if a file is allowed based on its extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for homepage
@app.route('/')
def homepage():
    return render_template('index.html')

# Function to calculate size of bounding box in inches
def calculate_size(box):
    conversion_factor = 0.0393701

    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    width_px = x2 - x1
    height_px = y2 - y1

    width_inch = width_px * conversion_factor
    height_inch = height_px * conversion_factor

    return round(width_inch, 2), round(height_inch, 2)


# Function to calculate distance in meters between two points
def calculate_distance_in_meters(pt1, pt2, pixel_per_meter):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    distance_px = np.sqrt(dx**2 + dy**2)
    return distance_px * pixel_per_meter


# Function to draw bounding boxes on the image based on detections
def draw_bounding_boxes(img, detections):
    # Color map for different detected classes
    color_map = {
        'Japanese Knotweed': "#FF9D97",
        'Himalayan Balsam': "#00C2FF",
        'Bindweed': "#FF95C8",
        'Ground Elder': "#FF3838",
        'Giant Hogweed': "#3DDB86",
        'Black Grass': "#2C99A8"
    }
    
    for det in detections:
        # Get the color for the current class
        box_color = color_map.get(det['class'], "#00D4BB")
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        obj_type = det['class']
        prob = det['confidence']

        # Expand the bounding box slightly
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(img.shape[1], x2 + 5)
        y2 = min(img.shape[0], y2 + 5)

        # Convert hex color to BGR for OpenCV
        box_color_bgr = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # Draw rectangle for the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), box_color_bgr, thickness=4)

        # Set font scale and thickness for text
        font_scale = 0.8 
        thickness = 2    

        # Get size of the text to be displayed
        text_size, _ = cv2.getTextSize(f"{obj_type} {prob:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size
        text_x = x1
        text_y = y1 - 5

        # Draw rectangle for the text background
        img = cv2.rectangle(img, (text_x, text_y - text_height), (text_x + text_width, text_y), box_color_bgr, -1)
        # Put the text on the image
        img = cv2.putText(img, f"{obj_type} {prob:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    return img


# Function to get bounding box detection results
def get_bbox_detection(source, all_detections, is_image=True):
    # If the source is an image file path
    if is_image:
        src = cv2.imread(source)  # Read the image using OpenCV
        if src is None:
            raise ValueError("Failed to read uploaded image")  # Raise an error if image reading fails
    else:
        src = source  # If the source is not an image file path, use the source directly (e.g., video frame)

    # Draw bounding boxes on the source image/frame
    src = draw_bounding_boxes(src, all_detections)

    # If the source is an image
    if is_image:
        retval, buffer = cv2.imencode('.jpg', src)  # Encode the image to JPEG format
        encoded_image = base64.b64encode(buffer).decode('utf-8')  # Convert the encoded image to base64
        return encoded_image  # Return the base64 encoded image
    else:
        return src  # Return the processed source (e.g., video frame)


# Route for uploading a file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = model.predict(filepath)
        if not results or len(results) == 0:
            return jsonify({'error': 'No results from model'})
        
        result = results[0]
        
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            width_inch, height_inch = calculate_size({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

            detections.append({
                'class': result.names[class_id],
                'confidence': prob,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': round(width_inch, 2),
                'height': round(height_inch, 2)
            })

        encoded_image = get_bbox_detection(filepath, detections)

        if encoded_image is None:
            return jsonify({'error': 'Error processing image'})

        object_count = {}
        for det in detections:
            obj_class = det['class']
            if obj_class in object_count:
                object_count[obj_class] += 1
            else:
                object_count[obj_class] = 1

        return jsonify({
            'image_base64': encoded_image,
            'detections': object_count,
            'detection_details': detections
        })

    return jsonify({'error': 'Invalid file format'})


# Route to render the detection page
@app.route('/detection', methods=['GET', 'POST'])
def detection_page():
    return render_template('detection.html')  # Render the detection.html template


# Route for video detection and handling URL input
@app.route('/detection-video', methods=['GET', 'POST'])
def live_video():
    if request.method == 'POST':
        url = request.form['url']  # Get the URL from the form input
        timestamp = datetime.now().timestamp()
        session['url'] = f"{url}?timestamp={timestamp}"  # Append timestamp to URL
        return redirect(url_for('live_video'))  # Redirect to the same route
    return render_template('video_stream.html')  # Render the video_stream.html template


class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self._data = []
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
        self._cap = None

    def show(self, url):
        print(url)
        self._data = []  # Clear detection data

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            if self._thread is not None:
                self._thread.join()
                self._thread = None

        # YouTube download options
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": 'bestvideo[height<=480]',
            "noplaylist": True,
            "forceurl": True,
        }

        # Extract video URL using youtube_dl
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        info = ydl.extract_info(url, download=False)
        if 'formats' in info:
            best_video = None
            for format in info['formats']:
                if format.get('url') and format.get('vcodec') != 'none':
                    if best_video is None or format.get('height', 0) > best_video.get('height', 0):
                        best_video = format

            if best_video:
                video_url = best_video['url']
                print(f"Video URL: {video_url}")
            else:
                raise Exception("No suitable video URL found in the extracted info.")
        else:
            raise Exception("No formats found in the extracted info.")

        # Start the video capturing in a separate thread
        self._thread = threading.Thread(target=self._capture_video, args=(video_url,))
        self._thread.start()

        while True:
            if self._frame is None:
                continue

            # Encode frame as JPEG
            with self._lock:
                frame = cv2.imencode(".jpg", self._frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def _capture_video(self, video_url):
        self._cap = cv2.VideoCapture(video_url)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            grabbed, frame = self._cap.read()
            if not grabbed:
                break
            with self._lock:
                self._frame = frame
        self._cap.release()

VIDEO = VideoStreaming()

@app.route('/video_feed')
def video_feed():
    url = session.get('url', None)
    if url is None:
        return redirect(url_for('detection_page'))
    
    response = Response(VIDEO.show(url), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/realtime_webcam')
def realtime_webcam():
    return render_template('webcam.html')


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
