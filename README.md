
This project utilizes a Generative Adversarial Network (GAN) to convert real images into high-quality pencil sketches. By leveraging deep learning techniques, the system can produce sketches that closely resemble hand-drawn artwork, capturing intricate details and artistic strokes.

Features:

- Converts real images into detailed pencil sketches.

- Utilizes a GAN-based deep learning model for enhanced realism.

- Simple Flask web interface for easy user interaction.

- Supports multiple sketch styles and shading techniques.

- Allows users to upload images and download the generated sketches.

Steps to Run the Project:

1.Download or Train the GAN Model
   If a pre-trained model is available, place it in the models/ directory.
   Otherwise, train the model using:
        python train.py --epochs 100 --batch_size 16

2.Run the Flask Web Application
    Start the web application by running:
        python app.py

3.The application will be available at:
    http://127.0.0.1:5000/
    Use the Application
    
4.Open your browser and navigate to http://127.0.0.1:5000/.
   Upload an image and click "Generate Sketch".
   Download the generated pencil sketch.