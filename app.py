import os
from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import torch
from model import build_unet
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management

# Path to the checkpoint file
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

def generate_segmentation(input_image):
    image = cv2.resize(input_image, (512, 512))  # Resize to match model input size
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
    return pred_mask

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':  # Simplified check, replace with proper validation
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        
        # Save the uploaded image
        input_image_path = os.path.join('files', 'image', file.filename)
        file.save(input_image_path)
        
        # Load the input image
        input_image = cv2.imread(input_image_path)
        
        # Generate the segmentation mask
        masked_image = generate_segmentation(input_image)
        
        # Apply post-processing
        gamma_corrected = adjust_gamma(masked_image, gamma=50.0)
        gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
        low_contrast_image = (gray_image * 0.9).astype(np.uint8)
        
        # Save the processed images
        cv2.imwrite(os.path.join('files', 'output', 'masked.png'), masked_image)
        cv2.imwrite(os.path.join('files', 'output', 'post_processed.png'), low_contrast_image)
        
        # Encode the images to base64
        original_image_b64 = base64.b64encode(cv2.imencode('.png', input_image)[1]).decode()
        masked_image_b64 = base64.b64encode(cv2.imencode('.png', masked_image)[1]).decode()
        post_processed_image_b64 = base64.b64encode(cv2.imencode('.png', low_contrast_image)[1]).decode()
        
        # Render the result page with the base64 encoded images
        return render_template('result.html', original_image=original_image_b64, masked_image=masked_image_b64, post_processed_image=post_processed_image_b64)
    
    return render_template('index.html')

@app.route('/about')
def about():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/result')
def result():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Load the processed images
    original_image = cv2.imread(os.path.join('files', 'image', os.listdir('files/image')[0]))
    masked_image = cv2.imread(os.path.join('files', 'output', 'masked.png'))
    post_processed_image = cv2.imread(os.path.join('files', 'output', 'post_processed.png'))
    
    # Encode the images to base64
    original_image_b64 = base64.b64encode(cv2.imencode('.png', original_image)[1]).decode()
    masked_image_b64 = base64.b64encode(cv2.imencode('.png', masked_image)[1]).decode()
    post_processed_image_b64 = base64.b64encode(cv2.imencode('.png', post_processed_image)[1]).decode()
    
    # Render the result page with the base64 encoded images
    return render_template('result.html', original_image=original_image_b64, masked_image=masked_image_b64, post_processed_image=post_processed_image_b64)

if __name__ == '__main__':
    app.run(debug=True)
