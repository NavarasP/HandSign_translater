import os
import time
import logging
import tempfile
from gtts import gTTS
from django.shortcuts import render
from django.http import HttpResponse
from main.functions import generate_frames, process_image
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageUploadForm
import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage



def home(request):
    return render(request, 'main/index.html')

def login(request):
    # Implement this view
    return render(request, 'main/login.html')


def upload_image(request):
    label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Define a default filename for the uploaded image
            default_filename = 'dump/uploaded_image.jpg'

            # Save the uploaded image with the default filename
            image = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save(default_filename, image)

            # Open the saved image for processing
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            label = process_image(image_path)

            # Return the response with the label and the image URL
            uploaded_file_url = fs.url(filename)
            return render(request, 'main/image.html', {'form': form, 'label': label, 'image_url': uploaded_file_url})

    else:
        form = ImageUploadForm()

    return render(request, 'main/image.html', {'form': form, 'label': label})




import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from .forms import VideoUploadForm
from .functions import process_video

def upload_video(request):
    label = None
    video_filename = 'dump/uploaded_video.mp4'  # Default filename for the video

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded video with a default filename
            video = form.cleaned_data['video']
            fs = FileSystemStorage()
            filename = fs.save(video_filename, video)
            uploaded_file_url = fs.url(filename)

            # Open the saved video for processing
            video_path = os.path.join(settings.MEDIA_ROOT, video_filename)
            label = process_video(video_path)

            # Clean up the saved video file after processing if necessary
            # os.remove(video_path)

            return render(request, 'main/video.html', {'form': form, 'label': label, 'video_url': uploaded_file_url})

    else:
        form = VideoUploadForm()

    return render(request, 'main/video.html', {'form': form, 'label': label})




def live_video(request):
    # Implement this view
    return render(request, 'main/live.html')


def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



logger = logging.getLogger(__name__)





def stream_predictions(request):
    def event_stream():
        last_sent_data = ""
        while True:
            try:
                with open('predicted_labels.txt', 'r') as file:
                    data = file.read().replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
                    if data and data != last_sent_data:
                        last_sent_data = data
                        yield f"data: {data}\n\n"
            except Exception as e:
                logger.error(f"Error reading file: {e}")
            time.sleep(1)  # Adjust the sleep duration as needed

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')





def download_txt(request):
    # Logic to generate or retrieve the text content
    content = "This is the transcript content."
    response = HttpResponse(content, content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename="transcript.txt"'
    return response

def download_audio(request):
    # Get the text content (you might want to pass this from the front-end or retrieve it from a database)
    text_content = "This is the transcript content that will be converted to speech."

    # Create a gTTS object
    tts = gTTS(text=text_content, lang='en')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        # Save the speech to the temporary file
        tts.save(fp.name)
        
        # Reopen the file and create the response
        with open(fp.name, 'rb') as audio_file:
            response = HttpResponse(audio_file.read(), content_type="audio/mpeg")
            response['Content-Disposition'] = 'attachment; filename="transcript_audio.mp3"'

    # Delete the temporary file
    os.unlink(fp.name)

    return response