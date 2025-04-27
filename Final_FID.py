import os
from torchvision import transforms, datasets
from PIL import Image

# Set the target image size to match model 
image_size = 64 #change image size/resoloution. needs to be same as training dataset
max_images = 3000 # change a,ount of FID images created

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
])

# Path to your original dataset 
dataset_path = r"C:\Users\ewanp\OneDrive\Desktop\project 10\dataset\img_align_celeba"

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Create the output folder where the preprocessed "real" images will be saved
output_folder = r"fid_data/real"
os.makedirs(output_folder, exist_ok=True)

# Iterate over the dataset and save up to max_images preprocessed images
for idx, (img, _) in enumerate(dataset):
    if idx >= max_images:
        break
    output_file = os.path.join(output_folder, f"real_{idx}.png")
    img.save(output_file)

print(f"Preprocessed {min(len(dataset), max_images)} real images have been saved to {output_folder}.")
