"""
PlantGuard AI — Leaf Disease Analysis Engine
Supports High-Accuracy TensorFlow/Keras models with a Pillow color-analysis fallback.
"""

from PIL import Image
import os
import math
import json
import numpy as np

# Try importing TensorFlow, but don't crash if it's not installed yet
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
    from tensorflow.keras.models import load_model # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ==========================================
# 🌿 TF Model Setup
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant_disease_model.h5')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'class_indices.json')

AI_MODEL = None
CLASS_INDICES = None

if TF_AVAILABLE and os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
    try:
        AI_MODEL = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            CLASS_INDICES = json.load(f)
        print("✅ PlantGuard AI: High-Accuracy TensorFlow Model Loaded Successfully!")
    except Exception as e:
        print(f"⚠️ PlantGuard AI: Failed to load TF model: {e}")

# ==========================================
# 📚 Disease Database
# ==========================================
DISEASE_DATABASE = {
    'Healthy': {
        'severity': 'none',
        'severity_label': 'No Issues',
        'color': '#10b981',
        'icon': '✅',
        'description': 'Your plant appears to be in excellent health. The leaf shows strong green coloration with no visible signs of disease or nutrient deficiency.',
        'recommendations': [
            'Continue your current watering schedule',
            'Maintain proper sunlight exposure (6-8 hours daily)',
            'Apply balanced fertilizer monthly during growing season',
            'Monitor regularly for early signs of pest activity',
        ],
    },
    'Leaf Blight': {
        'severity': 'high',
        'severity_label': 'Severe',
        'color': '#ef4444',
        'icon': '🔴',
        'description': 'Leaf blight is detected, characterized by large brown/dark patches on the leaf surface. This is typically caused by fungal pathogens.',
        'recommendations': [
            'Remove and destroy all infected leaves immediately',
            'Apply copper-based fungicide',
            'Improve air circulation around the plant',
            'Avoid overhead watering — water at the base',
        ],
    },
    'Rust': {
        'severity': 'medium',
        'severity_label': 'Moderate',
        'color': '#f59e0b',
        'icon': '🟠',
        'description': 'Rust disease detected, identified by orange-brown pustules and discoloration. Caused by Puccinia fungi.',
        'recommendations': [
            'Remove infected leaves to prevent spread',
            'Apply sulfur-based fungicide early in infection',
            'Space plants adequately for air circulation',
            'Avoid wetting foliage during watering',
        ],
    },
    'Powdery Mildew': {
        'severity': 'medium',
        'severity_label': 'Moderate',
        'color': '#8b5cf6',
        'icon': '🟣',
        'description': 'Powdery mildew detected, appearing as a white/grayish powdery coating on leaf surfaces.',
        'recommendations': [
            'Apply potassium bicarbonate or neem oil spray',
            'Ensure proper spacing between plants',
            'Prune overcrowded areas to improve airflow',
            'Water plants in the morning to allow drying',
        ],
    },
    'Leaf Spot': {
        'severity': 'low',
        'severity_label': 'Mild',
        'color': '#3b82f6',
        'icon': '🔵',
        'description': 'Leaf spot disease detected, showing small circular spots with defined borders.',
        'recommendations': [
            'Remove affected leaves to limit disease spread',
            'Apply appropriate fungicide or bactericide',
            'Avoid overhead irrigation',
        ],
    },
    'Nutrient Deficiency': {
        'severity': 'low',
        'severity_label': 'Mild',
        'color': '#06b6d4',
        'icon': '💧',
        'description': 'Signs of nutrient deficiency detected, indicated by yellowing (chlorosis) of leaf tissue.',
        'recommendations': [
            'Test soil pH and nutrient levels',
            'Apply balanced NPK fertilizer',
            'Ensure proper soil drainage',
        ],
    },
    # ---- PLUGINS FOR PLANTVILLAGE DATASET CLASSIFICATIONS ----
    'Tomato___Early_blight': {
        'severity': 'high', 'severity_label': 'Severe', 'color': '#ef4444', 'icon': '🍅',
        'description': 'Early blight detected on Tomato leaf (Alternaria solani).',
        'recommendations': ['Remove affected leaves', 'Apply fungicide (chlorothalonil or copper)', 'Mulch around base'],
    },
    'Tomato___Late_blight': {
        'severity': 'high', 'severity_label': 'Severe', 'color': '#dc2626', 'icon': '🍅',
        'description': 'Late blight detected on Tomato leaf (Phytophthora infestans). Very destructive.',
        'recommendations': ['Remove infected plants immediately', 'Apply copper fungicide', 'Ensure excellent airflow'],
    },
    'Potato___Early_blight': {
        'severity': 'high', 'severity_label': 'Severe', 'color': '#d97706', 'icon': '🥔',
        'description': 'Early blight detected on Potato leaf.',
        'recommendations': ['Apply chlorothalonil fungicide', 'Crop rotation', 'Ensure proper nutrition'],
    },
    'Potato___Late_blight': {
        'severity': 'high', 'severity_label': 'Severe', 'color': '#b45309', 'icon': '🥔',
        'description': 'Late blight detected on Potato leaf.',
        'recommendations': ['Destroy infected foliage', 'Harvest tubers in dry weather', 'Apply protective fungicide'],
    },
    'Apple___Apple_scab': {
        'severity': 'medium', 'severity_label': 'Moderate', 'color': '#4d7c0f', 'icon': '🍎',
        'description': 'Apple scab detected. Causes olive-green to black spots on leaves and fruit.',
        'recommendations': ['Rake up and destroy fallen leaves', 'Apply fungicide at green tip stage', 'Prune to increase airflow'],
    }
}


def analyze_leaf_image(image_path):
    """
    Main entry point for analysis.
    Uses TensorFlow if a model is loaded, otherwise falls back to Pillow color analysis.
    """
    if AI_MODEL is not None and CLASS_INDICES is not None:
        return _predict_with_tf(image_path)
    else:
        return _predict_with_pillow(image_path)


# ==========================================
# 🧠 TENSORFLOW PREDICTION ENGINE
# ==========================================
def _predict_with_tf(image_path):
    try:
        # Load and preprocess image for MobileNetV2
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale like in training

        # Predict
        predictions = AI_MODEL.predict(img_array)
        predicted_class_index = str(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        # Get class name
        predicted_disease = CLASS_INDICES.get(predicted_class_index, "Unknown")

        # Cleanup class name (e.g., 'Tomato___Healthy' -> 'Healthy' or format nicely)
        display_name = predicted_disease
        if 'Healthy' in predicted_disease:
            display_name = 'Healthy'
        elif predicted_disease not in DISEASE_DATABASE:
            # Fallback formatting for unknown PlantVillage classes
            display_name = predicted_disease.replace('___', ' - ').replace('_', ' ')

        details = _get_disease_info(predicted_disease if predicted_disease in DISEASE_DATABASE else 'Unknown')
        
        # Override name and description for dynamic classes
        details['original_class'] = predicted_disease
        if predicted_disease not in DISEASE_DATABASE:
            details['description'] = f"AI detected {display_name} with {confidence:.1f}% confidence."

        details['analysis'] = {
            'engine': 'TensorFlow Deep Learning',
            'raw_confidence': round(confidence, 2)
        }

        return display_name, round(confidence, 1), details

    except Exception as e:
        details = _get_disease_info('Healthy')
        details['error'] = f"TF Error: {str(e)}"
        return 'Healthy', 0.0, details


# ==========================================
# 🎨 PILLOW FALLBACK ENGINE (Color Analysis)
# ==========================================
def _predict_with_pillow(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)

        pixels = list(img.getdata())
        total_pixels = len(pixels)

        if total_pixels == 0:
            return 'Healthy', 50.0, _get_disease_info('Healthy')

        r_values = [p[0] for p in pixels]
        g_values = [p[1] for p in pixels]
        b_values = [p[2] for p in pixels]

        avg_r = sum(r_values) / total_pixels
        avg_g = sum(g_values) / total_pixels
        avg_b = sum(b_values) / total_pixels

        std_r = _std_dev(r_values, avg_r)
        std_g = _std_dev(g_values, avg_g)
        std_b = _std_dev(b_values, avg_b)

        total_color = avg_r + avg_g + avg_b
        if total_color == 0: total_color = 1
        red_ratio = avg_r / total_color

        green_pixels = brown_pixels = yellow_pixels = white_pixels = dark_pixels = orange_pixels = 0

        for r, g, b in pixels:
            brightness = (r + g + b) / 3
            if g > r * 1.1 and g > b * 1.1 and g > 60: green_pixels += 1
            elif r > g and r > b and 40 < brightness < 140: brown_pixels += 1
            elif r > 150 and g > 150 and b < 100: yellow_pixels += 1
            elif r > 160 and 80 < g < 160 and b < 80: orange_pixels += 1
            elif brightness > 200 and std_pixel(r, g, b) < 20: white_pixels += 1
            elif brightness < 50: dark_pixels += 1

        leaf_pixels = green_pixels + brown_pixels + yellow_pixels + orange_pixels
        if leaf_pixels == 0:
            leaf_pixels = 1

        green_pct = (green_pixels / leaf_pixels) * 100
        brown_pct = (brown_pixels / leaf_pixels) * 100
        yellow_pct = (yellow_pixels / leaf_pixels) * 100
        orange_pct = (orange_pixels / leaf_pixels) * 100

        white_pct = (white_pixels / total_pixels) * 100
        dark_pct = (dark_pixels / total_pixels) * 100
        color_variance = std_r + std_g + std_b

        scores = {}
        # Healthy requires mostly green, penalized heavily by any disease colors
        scores['Healthy'] = max(0, min(100, green_pct - (brown_pct * 3) - (yellow_pct * 3) - (orange_pct * 3)))
        
        # Diseases shouldn't be penalized by green, because diseased leaves are still partially green!
        scores['Leaf Blight'] = max(0, min(100, (brown_pct * 4) + (dark_pct * 0.5)))
        scores['Rust'] = max(0, min(100, (orange_pct * 5) + (brown_pct * 2)))
        scores['Powdery Mildew'] = max(0, min(100, (white_pct * 4) + (color_variance * 0.1)))
        scores['Leaf Spot'] = max(0, min(100, (brown_pct * 2.5) + (dark_pct * 1.0) + (color_variance * 0.2)))
        scores['Nutrient Deficiency'] = max(0, min(100, (yellow_pct * 4)))

        best_disease = max(scores, key=scores.get)
        best_score = scores[best_disease]

        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = min(95, max(35, 40 + (sorted_scores[0] - sorted_scores[1]) * 1.5)) if len(sorted_scores) > 1 and sorted_scores[0] > 0 else 50.0

        if best_score < 1:
            best_disease = 'Healthy'
            confidence = 50.0

        details = _get_disease_info(best_disease)
        details['analysis'] = {
            'engine': 'Pillow Color Heuristics (TF Model Not Found)',
            'green_percentage': round(green_pct, 1),
            'brown_percentage': round(brown_pct, 1),
            'yellow_percentage': round(yellow_pct, 1),
            'white_percentage': round(white_pct, 1),
            'color_variance': round(color_variance, 1),
        }
        
        return best_disease, round(confidence, 1), details

    except Exception as e:
        details = _get_disease_info('Healthy')
        details['error'] = str(e)
        return 'Healthy', 40.0, details


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def _get_disease_info(disease_name):
    # Fallback for dynamic AI classes that aren't strictly in DB
    if disease_name not in DISEASE_DATABASE:
        return {
            'severity': 'unknown',
            'severity_label': 'AI Detected',
            'color': '#8b5cf6',
            'icon': '🔍',
            'description': 'Detected by the Deep Learning model.',
            'recommendations': ['Consult local agricultural expert', 'Monitor plant closely'],
        }
    return DISEASE_DATABASE.get(disease_name).copy()

def _std_dev(values, mean):
    if len(values) == 0: return 0
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

def std_pixel(r, g, b):
    mean = (r + g + b) / 3
    return math.sqrt(((r - mean)**2 + (g - mean)**2 + (b - mean)**2) / 3)
