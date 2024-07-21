from django.shortcuts import render
from main.functions import generate_frames
from django.http import StreamingHttpResponse

from django.http import HttpResponse
import os

def home(request):
    return render(request, 'main/index.html')

def login(request):
    # Implement this view
    return render(request, 'main/login.html')

def upload_image(request):
    # Implement this view
    pass

def upload_video(request):
    # Implement this view
    pass

def live_video(request):
    # Implement this view
    return render(request, 'main/live.html')


def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



import os
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def stream_transcript(request):
    def event_stream():
        transcript_path = 'predicted_labels.txt'
        last_position = 0
        while True:
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as file:
                    file.seek(last_position)
                    new_content = file.read()
                    if new_content:
                        yield f"data: {new_content}\n\n"
                        last_position = file.tell()
            yield ":\n\n"  # Keep-alive

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response