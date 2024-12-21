from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Validate file type (if necessary)
    if file.filename.split('.')[-1].lower() != 'wav':
        return jsonify({'error': 'Invalid file type'}), 400

    # # Save the file (if needed) or process it in memory
    file.save(f'uploads/{file.filename}')  # Save locally or process further
    return jsonify({'message': f"File {file.filename} uploaded successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True,port=8081)