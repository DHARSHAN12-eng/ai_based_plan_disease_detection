import os
from PIL import Image
import random

def generate_dummy_dataset():
    dataset_dir = "dummy_dataset"
    
    crops = ["Apple", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Potato", "Strawberry", "Tomato", "Wheat", "Rice", "Soybean", "Citrus", "Banana", "Cotton", "Barley", "Oat", "Sugarcane", "Coffee", "Tea", "Cassava", "Mango", "Papaya", "Almond", "Walnut"]
    diseases = ["Blight", "Rust", "Scab", "Rot", "Spot", "Mildew", "Mosaic_Virus", "Wilt", "Smut", "Canker", "Anthracnose", "Mold", "Chlorosis", "Necrosis", "Gall", "Curl", "Stunt", "Yellows", "Pox", "Blast", "Blackleg", "Fire_Blight", "Dieback"]
    
    classes = ["Healthy"]
    for crop in crops:
        classes.append(f"{crop}___Healthy")
        for disease in diseases:
            if len(classes) >= 505: # Ensure we get over 500 classes
                break
            classes.append(f"{crop}___{disease}")
        if len(classes) >= 505:
            break
            
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    print(f"Creating dummy dataset with {len(classes)} classes...")
        
    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        # Create 5 dummy images per class so train/validation split works
        for i in range(5):
            # Random color as base, modified by disease type
            r, g, b = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
            if "Healthy" in cls:
                r, g, b = random.randint(20, 100), random.randint(150, 255), random.randint(20, 100) # Greenish
            elif "Blight" in cls or "Rot" in cls or "Necrosis" in cls:
                r, g, b = random.randint(100, 200), random.randint(50, 100), random.randint(0, 50) # Brownish
            elif "Rust" in cls or "Spot" in cls:
                r, g, b = random.randint(200, 255), random.randint(100, 180), random.randint(0, 50) # Orange/Yellowish
                
            img = Image.new('RGB', (224, 224), color=(r, g, b))
            img.save(os.path.join(class_dir, f"img_{i}.jpg"))
            
    print(f"✅ Created dummy dataset in '{dataset_dir}' with {len(classes)} classes.")

if __name__ == "__main__":
    generate_dummy_dataset()
