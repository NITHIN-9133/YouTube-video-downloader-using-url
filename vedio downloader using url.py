import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, url_for, send_file
from pytube import YouTube
import uuid
import logging
from io import BytesIO
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)

# Create necessary directories
os.makedirs('static/downloads', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Simple recommendation model using Keras
def create_recommendation_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')  # 3 categories: music, tutorial, entertainment
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train a simple model with dummy data
def train_dummy_model():
    # Dummy data: features might include video length, view count, likes, etc.
    X_train = np.random.random((100, 5))
    y_train = np.random.randint(0, 3, 100)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = create_recommendation_model()
    model.fit(X_train, y_train, epochs=5, verbose=0)
    model.save('models/recommendation_model.h5')
    
    return model, scaler

# Video feature extraction (simplified)
def extract_video_features(video_info):
    # In a real application, we would extract meaningful features
    # Here we're just creating dummy features for demonstration
    features = np.array([
        [
            float(video_info.length),
            float(video_info.views),
            float(len(video_info.title)),
            float(video_info.rating if video_info.rating else 0),
            float(2023)  # Placeholder for publish year
        ]
    ])
    return features

# Class to manage downloads
class DownloadManager:
    def __init__(self):
        self.downloads = {}
        
    def add_download(self, url, path):
        download_id = str(uuid.uuid4())
        self.downloads[download_id] = {
            'url': url,
            'path': path,
            'status': 'pending',
            'progress': 0
        }
        return download_id
        
    def update_status(self, download_id, status, progress=None):
        if download_id in self.downloads:
            self.downloads[download_id]['status'] = status
            if progress is not None:
                self.downloads[download_id]['progress'] = progress
                
    def get_download(self, download_id):
        return self.downloads.get(download_id)

# Initialize the download manager
download_manager = DownloadManager()

# Load or train model
try:
    model = keras.models.load_model('models/recommendation_model.h5')
    scaler = StandardScaler()  # In a real app, you would save and load the scaler too
    logger.info("Model loaded successfully")
except:
    logger.info("Training new model")
    model, scaler = train_dummy_model()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_url():
    url = request.form.get('url')
    if not url:
        return render_template('index.html', error="Please enter a YouTube URL")
    
    try:
        yt = YouTube(url)
        
        # Extract video features and perform prediction
        features = extract_video_features(yt)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        category_idx = np.argmax(prediction[0])
        categories = ['Music', 'Tutorial', 'Entertainment']
        predicted_category = categories[category_idx]
        
        video_info = {
            'id': yt.video_id,
            'title': yt.title,
            'url': url,
            'thumbnail': yt.thumbnail_url,
            'category': predicted_category,
            'confidence': float(prediction[0][category_idx])
        }
        
        return render_template('preview.html', video=video_info)
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return render_template('index.html', error=f"Error: {str(e)}")

@app.route('/download', methods=['POST'])
def download_video():
    url = request.form.get('url')
    if not url:
        return redirect(url_for('index'))
    
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        download_path = os.path.join('static/downloads', f"{uuid.uuid4()}.mp4")
        
        # Add download to manager
        download_id = download_manager.add_download(url, download_path)
        
        # Start download in background thread
        def do_download():
            try:
                stream.download(filename=download_path)
                download_manager.update_status(download_id, 'completed', 100)
            except Exception as e:
                logger.error(f"Download error: {e}")
                download_manager.update_status(download_id, 'failed')
        
        thread = threading.Thread(target=do_download)
        thread.start()
        
        return redirect(url_for('download_status', download_id=download_id))
    except Exception as e:
        logger.error(f"Error starting download: {e}")
        return render_template('index.html', error=f"Download error: {str(e)}")

@app.route('/status/<download_id>')
def download_status(download_id):
    download = download_manager.get_download(download_id)
    if not download:
        return redirect(url_for('index'))
    
    if download['status'] == 'completed':
        filename = os.path.basename(download['path'])
        download_url = url_for('static', filename=f'downloads/{filename}')
        return render_template('download_complete.html', download_url=download_url, filename=filename)
    
    return render_template('downloading.html', download_id=download_id)

@app.route('/check_progress/<download_id>')
def check_progress(download_id):
    download = download_manager.get_download(download_id)
    if not download:
        return {'status': 'not_found'}
    
    return {
        'status': download['status'],
        'progress': download['progress']
    }

# HTML Templates (normally these would be in separate files)
@app.template_filter('html')
def html_template(template_name):
    templates = {
        'index.html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>YouTube Downloader</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .container { background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    h1 { color: #ff0000; }
                    input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
                    button { background-color: #ff0000; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; }
                    button:hover { background-color: #cc0000; }
                    .error { color: red; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>YouTube Video Downloader</h1>
                    <p>Enter a YouTube URL to download:</p>
                    <form action="/process" method="post">
                        <input type="text" name="url" placeholder="https://www.youtube.com/watch?v=..." required>
                        <button type="submit">Process URL</button>
                    </form>
                    {% if error %}
                    <p class="error">{{ error }}</p>
                    {% endif %}
                </div>
            </body>
            </html>
        ''',
        'preview.html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Preview</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .container { background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    h1 { color: #ff0000; }
                    .video-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin: 20px 0; }
                    .video-container iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
                    .download-btn { background-color: #ff0000; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; display: block; margin: 20px auto; font-size: 16px; }
                    .download-btn:hover { background-color: #cc0000; }
                    .video-info { margin-top: 20px; }
                    .prediction { background-color: #e0f7fa; padding: 10px; border-radius: 4px; margin-top: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Video Preview</h1>
                    <div class="video-container">
                        <iframe width="560" height="315" src="https://www.youtube.com/embed/{{ video.id }}" frameborder="0" allowfullscreen></iframe>
                    </div>
                    <div class="video-info">
                        <h2>{{ video.title }}</h2>
                        <div class="prediction">
                            <p>AI prediction: This video is likely a <strong>{{ video.category }}</strong> ({{ video.confidence|round(2) * 100 }}% confidence)</p>
                        </div>
                    </div>
                    <form action="/download" method="post">
                        <input type="hidden" name="url" value="{{ video.url }}">
                        <button type="submit" class="download-btn">Download Video</button>
                    </form>
                </div>
            </body>
            </html>
        ''',
        'downloading.html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Downloading...</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .container { background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); text-align: center; }
                    h1 { color: #ff0000; }
                    .progress-container { margin: 20px 0; background-color: #ddd; border-radius: 4px; }
                    .progress-bar { height: 20px; background-color: #ff0000; width: 0%; border-radius: 4px; transition: width 0.3s; }
                </style>
                <script>
                    function checkProgress() {
                        fetch('/check_progress/{{ download_id }}')
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'completed') {
                                    window.location.href = '/status/{{ download_id }}';
                                } else if (data.status === 'failed') {
                                    document.getElementById('status').innerHTML = 'Download failed. <a href="/">Try again</a>';
                                } else {
                                    document.getElementById('progress-bar').style.width = data.progress + '%';
                                    setTimeout(checkProgress, 1000);
                                }
                            });
                    }
                    
                    window.onload = function() {
                        setTimeout(checkProgress, 1000);
                    };
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Downloading Video</h1>
                    <p id="status">Please wait while we download your video...</p>
                    <div class="progress-container">
                        <div id="progress-bar" class="progress-bar"></div>
                    </div>
                </div>
            </body>
            </html>
        ''',
        'download_complete.html': '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Download Complete</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .container { background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); text-align: center; }
                    h1 { color: #ff0000; }
                    .download-link { background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block; margin: 20px 0; }
                    .download-link:hover { background-color: #45a049; }
                    .home-link { display: block; margin-top: 20px; color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Download Complete!</h1>
                    <p>Your video has been successfully downloaded.</p>
                    <a class="download-link" href="{{ download_url }}" download="{{ filename }}">Save to Gallery</a>
                    <a class="home-link" href="/">Download Another Video</a>
                </div>
            </body>
            </html>
        '''
    }
    return templates.get(template_name, '')

def render_template(template_name, **context):
    template = html_template(template_name)
    for key, value in context.items():
        template = template.replace('{{ ' + key + ' }}', str(value))
    
    # Handle conditionals (very basic implementation)
    import re
    cond_pattern = r'{%\s*if\s+([^%]+)\s*%}(.*?){%\s*endif\s*%}'
    for match in re.finditer(cond_pattern, template, re.DOTALL):
        condition, content = match.groups()
        condition_result = eval(condition, {'context': context}, context)
        if not condition_result:
            content = ''
        template = template.replace(match.group(0), content)
    
    return template

if __name__ == '__main__':
    app.run(debug=True)