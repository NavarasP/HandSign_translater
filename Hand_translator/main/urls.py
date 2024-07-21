from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('live_video/', views.live_video, name='live_video'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stream_transcript/', views.stream_transcript, name='stream_transcript'),


]
