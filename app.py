import os
from flask import Flask, jsonify, request, send_from_directory
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bson.binary
from PIL import Image
from img2vec_pytorch import Img2Vec
import io
import logging
import base64

app = Flask(__name__)

# Setting up MongoDB connection using environment variable
mongo_uri = os.environ.get('MONGO_URI', "mongodb+srv://thaphat:NC2pGmyhxqNpv4wM@templedb.tmtu6rf.mongodb.net/mydatabase?retryWrites=true&w=majority&appName=TempleDB")
app.config['MONGO_URI'] = mongo_uri
mongo = PyMongo(app)

# Select or create collections
collection = mongo.db.mycollection
image_features_collection = mongo.db.image_features

# Load the efficientnet_b5 model from img2vec
img2vec = Img2Vec(model='efficientnet_b5')

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_features(img):
    try:
        features = img2vec.get_vec(img, tensor=False)
        return features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify(message="No image file sent"), 400
        if 'description' not in request.form:
            return jsonify(message="No description sent"), 400
        if 'topic' not in request.form:
            return jsonify(message="No topic sent"), 400

        image_file = request.files['image']
        description = request.form['description']
        topic = request.form['topic']

        img = Image.open(image_file)
        features = extract_features(img)
        if features is None:
            return jsonify(message="Error extracting features"), 500

        # Save image in MongoDB with feature vector and description
        image_binary = io.BytesIO()
        img.save(image_binary, format=img.format)
        image_binary = image_binary.getvalue()

        document = {
            'topic': topic,
            'description': description,
            'features': features.tolist(),  # Store feature vector as array
            'filename': image_file.filename,
            'image': bson.binary.Binary(image_binary)
        }

        logging.info(f"Document to be inserted: {document}")
        image_features_collection.insert_one(document)
        return jsonify(message="Image, description, and features added successfully"), 201
    except Exception as e:
        logging.error(f"Error uploading image: {e}")
        return jsonify(message="Error uploading image"), 500

@app.route('/search_image', methods=['POST'])
def search_image():
    try:
        if 'image' not in request.files:
            return jsonify(message="No image file sent"), 400

        image_file = request.files['image']
        img = Image.open(image_file)
        query_features = extract_features(img)
        if query_features is None:
            return jsonify(message="Error extracting features"), 500

        search_query = {
            "$vectorSearch": {
                "index": "vector_index",  # Index name used for search
                "path": "features",  # Path to the features in the document
                "queryVector": query_features.tolist(),  # Query vector
                "numCandidates": 10,  # Number of candidates to consider
                "limit": 1  # Number of results to return
            }
        }

        pipeline = [
            search_query,
            {"$limit": 5}  # Limit number of results
        ]

        results = list(image_features_collection.aggregate(pipeline))
        if not results:
            return jsonify(message="No similar images found"), 404

        result_images = []
        for result in results:
            if 'image' in result:
                image_data = result['image']
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                result_images.append({
                    'filename': result['filename'],
                    'topic': result.get('topic', 'N/A'),
                    'description': result['description'],
                    'image': image_base64
                })
            else:
                logging.error(f"No 'image' field in result with ID {result['_id']}")

        return jsonify(results=result_images)
    except Exception as e:
        logging.error(f"Error searching image: {e}")
        return jsonify(message=f"Error searching image: {e}"), 500

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
