from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(required=True)


class VideoUploadForm(forms.Form):
    video = forms.FileField(label='Upload a video', required=True)
