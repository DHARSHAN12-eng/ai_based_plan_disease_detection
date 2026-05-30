from django.db import models


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=255, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True, default=0.0)
    details = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Image uploaded at {self.uploaded_at}"
