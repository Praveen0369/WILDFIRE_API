from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
from cnn import CNNET  # Assuming your OPTI class is defined in optinet module
from flask_cors import CORS
from optinet import OPTI 
from veginet import VGG
from soilu import SOIL
from adanet import ADA
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

opti_instance = OPTI()
 # Create an instance of the OPTI class

@app.route("/img_cnn", methods=["POST"])
def process_image():
    try:
        file = request.files['image']

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the image via file.stream
            img = Image.open(BytesIO(file.stream.read()))
            
            # Call the opti_con method using the instance
            processed_image_path = CNNET.cnet_con(img)

            # Return the processed image file
            return send_file(processed_image_path, mimetype='image/png', as_attachment=True)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/img_ada", methods=["POST"])
def process_image5():
    try:
        file = request.files['image']

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the image via file.stream
            img = Image.open(BytesIO(file.stream.read()))
            processed_image_path = ADA.adanet_con(img)

            # Return the processed image file
            return send_file(processed_image_path, mimetype='image/png', as_attachment=True)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route("/img_opti", methods=["POST"])
def process_image2():
    try:
        file = request.files['image']

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the image via file.stream
            img = Image.open(BytesIO(file.stream.read()))
            
            # Call the opti_con method using the instance
            processed_image_path = opti_instance.opti_con(img)

            # Return the processed image file
            return send_file(processed_image_path, mimetype='image/png', as_attachment=True)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/img_vgg", methods=["POST"])
def process_image3():
    try:
        file = request.files['image']

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the image via file.stream
            img = Image.open(BytesIO(file.stream.read()))
           # img_array = np.array(img)
            
            # Call the con_veg method using the class
            processed_image_path = VGG.con_veg(img)

            # Call the opti_con method using the instance
            #processed_image_path = VGG.con_veg(img)

            # Return the processed image file
            return send_file(processed_image_path, mimetype='image/png', as_attachment=True)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route("/img_soil", methods=["POST"])
def process_image4():
    try:
        file = request.files['image']

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the image via file.stream
            img = Image.open(BytesIO(file.stream.read()))
            
            # Call the opti_con method using the instance
            processed_image_path = SOIL.con_soil(r'SOIL_IMAGES')

            # Return the processed image file
            return send_file(processed_image_path, mimetype='image/png', as_attachment=True)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

def allowed_file(filename):
    # Add any additional file format checks if needed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

if __name__ == "__main__":
    app.run(debug=True)
