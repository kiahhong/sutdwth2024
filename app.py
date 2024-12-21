from flask import Flask, request, jsonify
import librosa
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv1D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Input, add, Flatten, Dense, BatchNormalization, Dropout, LSTM, GRU
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, Activation, LeakyReLU, ReLU

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
    
    val=[]
    classes = ["COPD" ,"Bronchiolitis ", "Pneumoina", "URTI", "Healthy"]

    # ADD THE SOUND FILE OVER HERE
    sound_file = f'uploads/{file.filename}'
    features = 52
    Input_Sample = Input(shape=(1,52))

    model_conv = Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(Input_Sample)
    model_conv = MaxPooling1D(pool_size=2, strides = 2, padding = 'same')(model_conv)
    model_conv = BatchNormalization()(model_conv)

    model_conv = Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu')(model_conv)
    model_conv = MaxPooling1D(pool_size=2, strides = 2, padding = 'same')(model_conv)
    model_conv = BatchNormalization()(model_conv)

    model_2_1 = GRU(32,return_sequences=True,activation='tanh',go_backwards=True)(model_conv)
    model_2 = GRU(128,return_sequences=True, activation='tanh',go_backwards=True)(model_2_1)

    model_3 = GRU(64,return_sequences=True,activation='tanh',go_backwards=True)(model_conv)
    model_3 = GRU(128,return_sequences=True, activation='tanh',go_backwards=True)(model_3)

    model_x = GRU(64,return_sequences=True,activation='tanh',go_backwards=True)(model_conv)
    model_x = GRU(128,return_sequences=True, activation='tanh',go_backwards=True)(model_x)

    model_add_1 = add([model_3,model_2,model_x])

    model_5 = GRU(128,return_sequences=True,activation='tanh',go_backwards=True)(model_add_1)
    model_5 = GRU(32,return_sequences=True, activation='tanh',go_backwards=True)(model_5)

    model_6 = GRU(64,return_sequences=True,activation='tanh',go_backwards=True)(model_add_1)
    model_6 = GRU(32,return_sequences=True, activation='tanh',go_backwards=True)(model_6)

    model_add_2 = add([model_5,model_6,model_2_1])


    model_7 = Dense(32, activation=None)(model_add_2)
    model_7 = LeakyReLU()(model_7)
    model_7 = Dense(128, activation=None)(model_7)
    model_7 = LeakyReLU()(model_7)

    model_9 = Dense(64, activation=None)(model_add_2)
    model_9 = LeakyReLU()(model_9)
    model_9 = Dense(128, activation=None)(model_9)
    model_9 = LeakyReLU()(model_9)

    model_add_3 = add([model_7,model_9])

    model_10 = Dense(64, activation=None)(model_add_3)
    model_10 = LeakyReLU()(model_10)

    model_10 = Dense(32, activation=None)(model_10)
    model_10 = LeakyReLU()(model_10)

    model_10 = Dense(5, activation="softmax")(model_10)

    gru_model = Model(inputs=Input_Sample, outputs = model_10)

    optimiser = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    gru_model.compile(optimizer=optimiser, loss='categorical_crossentropy',metrics=['accuracy'])

    gru_model.load_weights('GRU_Model.h5')
    data_x, sampling_rate = librosa.load(sound_file,res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T,axis=0)
    val.append(mfccs)
    val = np.expand_dims(val,axis=1)
    # print(val)
    p=classes[np.argmax(gru_model.predict(val))]
    return jsonify({'message': f"File {classes[np.argmax(gru_model.predict(val))]} uploaded successfully!"}), 200
        # return jsonify({'message': f"File {file.filename} uploaded successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True,port=8081)