import os
import time
import logging
import tempfile
from gtts import gTTS
from django.shortcuts import render
from django.http import HttpResponse
from main.functions import generate_frames
from django.http import StreamingHttpResponse




def home(request):
    return render(request, 'main/index.html')

def login(request):
    # Implement this view
    return render(request, 'main/login.html')

def upload_image(request):
    # Implement this view
    return render(request, 'main/image.html')

def process_image(request):
    # Implement this view
    pass

def upload_video(request):
    # Implement this view
    return render(request, 'main/video.html')

def process_video(request):
    # Implement this view
    pass

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