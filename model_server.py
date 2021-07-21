from flask import Flask, current_app, request, jsonify, send_file
import json
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from Model.generator import *
from Model.ops import *


generator = make_style_gan_generator(512, 512)

generator.load_weights("SavedModels/generator_latest.h5")

print("Loaded model from disk")

idx = 0

sprite_dictionary = {}

app = Flask(__name__)

@app.route('/getSpriteJson', methods=['GET'])
def getSpriteJson():

    noiseVector = noise(1,512)
    noiseImage = noise_image(1,512)

    global idx

    idx+=1

    sprite_dictionary[str(idx)] = [noiseVector,noiseImage]

    noiseVectorList = [noiseVector] * int(log2(512) - 1)

    image = generator.predict(noiseVectorList + [noiseImage], batch_size = 1)

    resizedImage = cv2.resize(image[0]*255., dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("generated.jpg", cv2.cvtColor(resizedImage, cv2.COLOR_RGB2BGR))

    with open("generated.jpg", "rb") as image_file:
        b64Image = base64.b64encode(image_file.read())

    return jsonify(id=idx,image=b64Image.decode('utf-8'))


@app.route('/getLerpSpriteJson', methods=['GET'])
def getLerpSpriteJson():

    startIDX = request.args.get('startID')
    startIDX = str(startIDX)

    endIDX = request.args.get('endID')
    endIDX = str(endIDX)

    lerpProgress = request.args.get('lerpProgress')
    lerpProgress = int(lerpProgress)

    numFrames = request.args.get('numFrames')
    numFrames = int(numFrames)

    linX = list(np.linspace(0, 1, numFrames))

    v1 = sprite_dictionary[startIDX][0]
    v2 = sprite_dictionary[endIDX][0]

    noise_image_1 = sprite_dictionary[startIDX][1]
    noise_image_2 = sprite_dictionary[endIDX][1]

    x = linX[lerpProgress]

    lerpedVector = v1 * (1-x) + v2 * (x)

    lerpedNoiseImage = noise_image_1 * (1-x) + noise_image_2 * (x)

    global idx

    idx+=1

    sprite_dictionary[str(idx)] = [lerpedVector,lerpedNoiseImage]


    noiseVectorList = [lerpedVector] * int(log2(512) - 1)

    image = generator.predict(noiseVectorList + [lerpedNoiseImage], batch_size = 1)

    resizedImage = cv2.resize(image[0]*255., dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("generated.jpg", cv2.cvtColor(resizedImage, cv2.COLOR_RGB2BGR))

    with open("generated.jpg", "rb") as image_file:
        b64Image = base64.b64encode(image_file.read())

    return jsonify(id=idx,image=b64Image.decode('utf-8'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)