/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as tf from '@tensorflow/tfjs';
// import * as tfd from '@tensorflow/tfjs-data';

// import {ControllerDataset} from './controller_dataset';
// import * as ui from './ui';

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 3;

// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;
let model;
let isPredicting = false;

async function predict() {
  // ui.isPredicting();

  var canvas = document.querySelector("#videoCanvas");
  var ctx = canvas.getContext('2d');

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  while (isPredicting) {
    // Capture the frame from the webcam.
    // const img = await getImage();
    console.log("predicting...")

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
    // videoCam = document.getElementById('videoElement');
    const predictions = await model.estimateFaces(video, returnTensors);

    if (predictions.length > 0) {
      // console.log(predictions.length + " faces")

      // `predictions` is an array of objects describing each detected face, for example:

      // [
      //   {
      //     topLeft: [232.28, 145.26],
      //     bottomRight: [449.75, 308.36],
      //     probability: [0.998],
      //     landmarks: [
      //       [295.13, 177.64], // right eye
      //       [382.32, 175.56], // left eye
      //       [341.18, 205.03], // nose
      //       [345.12, 250.61], // mouth
      //       [252.76, 211.37], // right ear
      //       [431.20, 204.93] // left ear
      //     ]
      //   }
      // ]


      for (let i = 0; i < predictions.length; i++) {
        const start = predictions[i].topLeft;
        const end = predictions[i].bottomRight;
        const size = [end[0] - start[0], end[1] - start[1]];

        console.log("topleft: " + predictions[i].topLeft)
        console.log("bottomright: " + predictions[i].bottomRight)

        // Render a rectangle over each detected face.
        // ctx.fillRect(start[0], start[1], size[0], size[1]);
        // ctx.stroke();



        x = start[0]
        y = start[1]
        width = end[0] - start[0]
        height = end[1] - start[1]

        console.log("x: " + x)
        console.log("y: " + y)
        console.log("width: " + width)
        console.log("height: " + height)

        // console.log(ctx)
        // ctx.fillRect(x, y, width, height);
        // const faceArea = 300;
        // const pX=canvas.width/2 - faceArea/2;
        // const pY=canvas.height/2 - faceArea/2;
        ctx.rect(Math.floor(x),Math.floor(y),Math.floor(width),Math.floor(height));
        ctx.lineWidth = "6";
        ctx.strokeStyle = "red";
        ctx.stroke();
      }
    }

    // ctx.fillRect(10, 10, 100, 100);
    // ctx.lineWidth = "6";
    // ctx.strokeStyle = "red";
    // ctx.stroke();


    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model, i.e., "embeddings" of the input images.
    // const embeddings = truncatedMobileNet.predict(img);

    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    // const predictions = model.predict(embeddings);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    // const predictedClass = predictions.as1D().argMax();
    // const classId = (await predictedClass.data())[0];
    // img.dispose();

    // ui.predictClass(classId);
    // isPredicting = false
    await tf.nextFrame();
  }
  // ui.donePredicting();
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

async function setupCamera() {
  console.log("Init cam..")
  video = document.getElementById('videoElement');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;
  console.log("Cam loaded")

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}


document.getElementById('predict').addEventListener('click', () => {
  // ui.startPacman();
  isPredicting = true;
  predict();
});

async function init() {

  console.log("Loading model...")
  model = await blazeface.load();
  console.log("Model loaded")

  // const screenShot = await webcam.capture();
  // // truncatedMobileNet.predict(screenShot.expandDims(0));
  // screenShot.dispose();
}

// Initialize the application.
setupCamera()
init();
