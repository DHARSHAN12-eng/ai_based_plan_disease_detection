import json
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import UploadedImage
from .utils import analyze_leaf_image, DISEASE_DATABASE
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'detection/signup.html', {'form': form})


@login_required
def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            # Perform real image analysis
            prediction, confidence, details = analyze_leaf_image(uploaded_image.image.path)
            uploaded_image.prediction = prediction
            uploaded_image.confidence = confidence
            uploaded_image.details = json.dumps(details)
            uploaded_image.save()
            return redirect('result', pk=uploaded_image.pk)
    else:
        form = ImageUploadForm()
    return render(request, 'detection/home.html', {'form': form})


@login_required
def result(request, pk):
    uploaded_image = UploadedImage.objects.get(pk=pk)

    # Parse stored details
    details = {}
    if uploaded_image.details:
        try:
            details = json.loads(uploaded_image.details)
        except json.JSONDecodeError:
            details = {}

    # Get disease info
    disease_info = DISEASE_DATABASE.get(uploaded_image.prediction, DISEASE_DATABASE['Healthy'])

    context = {
        'uploaded_image': uploaded_image,
        'details': details,
        'disease_info': disease_info,
        'confidence': uploaded_image.confidence or 0,
        'recommendations': details.get('recommendations', disease_info.get('recommendations', [])),
        'severity': details.get('severity_label', disease_info.get('severity_label', 'Unknown')),
        'severity_color': details.get('color', disease_info.get('color', '#888')),
        'disease_icon': details.get('icon', disease_info.get('icon', '🔍')),
        'disease_description': details.get('description', disease_info.get('description', '')),
        'analysis': details.get('analysis', {}),
    }
    return render(request, 'detection/result.html', context)


@login_required
def history(request):
    """View past analysis results."""
    images = UploadedImage.objects.filter().order_by('-uploaded_at')[:20]
    results = []
    for img in images:
        info = DISEASE_DATABASE.get(img.prediction, DISEASE_DATABASE['Healthy'])
        results.append({
            'image': img,
            'color': info.get('color', '#888'),
            'icon': info.get('icon', '🔍'),
            'severity_label': info.get('severity_label', 'Unknown'),
        })
    return render(request, 'detection/history.html', {'results': results})
