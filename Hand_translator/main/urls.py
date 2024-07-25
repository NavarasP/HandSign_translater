from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('live_video/', views.live_video, name='live_video'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stream_predictions/', views.stream_predictions, name='stream_predictions'),
    path('download-txt/', views.download_txt, name='download_txt'),
    path('download-audio/', views.download_audio, name='download_audio'),
    path('process_image/', views.stream_predictions, name='process_image'),
    path('process_video/', views.stream_predictions, name='process_video'),
    
]




