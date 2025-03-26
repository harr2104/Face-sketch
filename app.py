import base64
import io
import os
import PIL
from flask import (
    Flask, flash, render_template, request, send_file
)
import numpy as np
import logging
import cv2
import tensorflow as tf
from PIL import Image

outputname = "pred_letter.jpeg"
size = []
garrey = []
ogname = []
ogsize = []

# Logging configuration
logging.basicConfig(level=logging.DEBUG, filename="log.log",
                    filemode="a", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    app.logger.info('Index.html page working')
    return render_template("index.html")



@app.route("/project.html")
def project():
    app.logger.info('project.html page working')
    return render_template("project.html")

@app.route('/project.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        filename = "input.jpeg"
        f = request.files['file']
        app.logger.info('Input image uploaded')

        garrey.append(filename)
        f.save(filename)
        resizeinbox(filename)
        app.logger.info('resizeinbox() is done on input image')

        img = Image.open(filename)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        data = io.BytesIO()
        img.save(data, "JPEG")
        encode_img_data = base64.b64encode(data.getvalue())

        return render_template('project.html', filename=encode_img_data.decode("UTF-8"))

@app.route('/transform', methods=['GET', 'POST'])
def transform():
    try:
        if not garrey:
            flash('No image uploaded. Please upload an image before proceeding.', 'error')
            return "No image uploaded", 400

        filename = garrey.pop()
        app.logger.info('Input image shown in input box')

        img = Image.open(filename)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        data = io.BytesIO()
        img.save(data, "JPEG")
        encode_img_data = base64.b64encode(data.getvalue())

        # Read image using OpenCV and preprocess
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        image = cv2.resize(image, (128, 128))
        img2 = (image - 127.5) / 127.5
        img = np.reshape(img2, (1, 128, 128, 3))  

        app.logger.info('Input image normalized for model')

        # Load GAN model
        model_path = "D:\\Mini project - 1\\Face sketch\\saved_model\\pix2pix_generator_fixed.h5" #give your saved model path
        loaded_styled_generator = tf.keras.models.load_model(model_path)
        app.logger.info('Model loaded successfully')

        
        pred_letter = loaded_styled_generator(img, training=False)[0].numpy()
        pred_letter = (pred_letter * 127.5 + 127.5).astype(np.uint8)

        
        if size:
            width = size.pop()
            height = size.pop()
            pred_letter = cv2.resize(pred_letter, (width, height))
            app.logger.info('Resized predicted image to match input dimensions')

        # Save the output
        outputname = "pred_letter.jpeg"
        cv2.imwrite(outputname, pred_letter)

        img2 = Image.open(outputname)
        if img2.mode == "RGBA":
            img2 = img2.convert("RGB")
        
        data = io.BytesIO()
        img2.save(data, "JPEG")
        app.logger.info('Predicted image saved successfully')

        encode_img_data2 = base64.b64encode(data.getvalue())

        return render_template('project.html',
                               filename=encode_img_data.decode("UTF-8"),
                               outputname=encode_img_data2.decode("UTF-8"))

    except Exception as e:
        app.logger.critical(f'Error in transform function: {str(e)}')
        flash('An error occurred during transformation. Please try again.', 'error')
        return f"Error: {str(e)}", 500

@app.route('/download')
def download_file():
    ogheight = ogsize.pop()
    ogwidth = ogsize.pop()
    path = "pred_letter.jpeg"
    return send_file(path, as_attachment=True)

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        filename = "input.jpeg"
        img_data = request.files['img'].read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(filename, img)
        garrey.append(filename)

        app.logger.info('Input image captured')
        resizeinbox(filename)
        app.logger.info('resizeinbox() is done on input image')

        img = Image.open(filename)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        data = io.BytesIO()
        img.save(data, "JPEG")
        app.logger.info('save2')

        encode_img_data = base64.b64encode(data.getvalue())
        app.logger.info('encoded')

        return render_template('project.html', filename=encode_img_data.decode("UTF-8"))

def resizeinbox(filename):
    fixed_size = 400
    image = Image.open(filename)

    if float(image.size[1]) < 400 and float(image.size[0]) < 400:
        size.append(image.size[0])
        size.append(image.size[1])
        app.logger.info('given image dim < 400')
    elif float(image.size[1]) > float(image.size[0]):
        ogsize.append(image.size[0])
        ogsize.append(image.size[1])
        height_percent = (fixed_size / float(image.size[1]))
        width_size = int((float(image.size[0]) * height_percent))
        image = image.resize((width_size, fixed_size), PIL.Image.NEAREST)
        image.save(filename)
        size.append(fixed_size)
        size.append(width_size)
        app.logger.info('given image is portrait')
    else:
        ogsize.append(image.size[0])
        ogsize.append(image.size[1])
        height_percent = (fixed_size / float(image.size[0]))
        width_size = int((float(image.size[1]) * height_percent))
        image = image.resize((fixed_size, width_size), PIL.Image.NEAREST)
        image.save(filename)
        size.append(width_size)
        size.append(fixed_size)
        app.logger.info('given image is landscape')

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)