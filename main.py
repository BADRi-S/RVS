import os
import cv2
import numpy as np
import torch
from model import build_unet  

#Path to the checkpoint file
checkpoint_path = "files/savedModel/checkpoint.pth"

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = build_unet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def generate_segmentation(input_image_path, output_image_path):
    # Read the input image
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512))  # Resize to match model input size
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    
    # Move input data to the same device as the model
    x = x.to(device)

    # Perform inference
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    # Save the output mask image
    pred_mask = mask_parse(pred_y)

   
    if not output_image_path.endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Invalid output image file extension. Please use a supported format (e.g., .png, .jpg)")

    cv2.imwrite(output_image_path, pred_mask)


input_image_path = "files/image/5.png"
output_image_path = "files/output/op5.png"
generate_segmentation(input_image_path, output_image_path)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

#Result Image Post Processing
def main():
    # Load the image
    img = cv2.imread(output_image_path)

    # Display the original image
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)  # Wait for any key to be pressed

    # Apply gamma correction
    gamma = 50.0  # Set the gamma value (higher gamma for maximum effect)
    img_gamma_corrected = adjust_gamma(img, gamma)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2GRAY)

    # Apply contrast reduction by multiplying by a fraction (e.g., 0.5)
    img_low_contrast = (img_gray * 0.9).astype(np.uint8)

    # Display the processed image
    cv2.imshow('Processed Image', img_low_contrast)
    cv2.waitKey(0)  # Wait for any key to be pressed

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

