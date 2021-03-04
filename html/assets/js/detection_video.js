let detection_model_video, classification_model_video, ctx, videoWidth, videoHeight, video, canvas;

async function setupCamera() {
  video = document.getElementById('videoElement');

  const constraints = {
    audio: false,
    video: true
  };

  function handleSuccess(stream) {
    window.stream = stream; // make stream available to browser console
    video.srcObject = stream;
  }

  function handleError(error) {
    console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
  }

  const stream = await navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);

  // video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const preprocessPrediction = (prediction) => {
  const start = prediction.topLeft;
  const end = prediction.bottomRight;
  x = start[0]
  y = start[1]
  width = start[0] - end[0]
  height = start[1] - end[1]
  diff = 0

  if (height < width) {

    // console.log("height < width")

    const delta = parseInt(Math.round((height - width) / 2))
    const y_min = y - diff - delta
    const y_max = y + height + diff
    const x_min = x - delta - diff
    const x_max = x + width + delta + diff

    let width_ = x_min - x_max
    let height_ = y_min - y_max
    const width_delta = width_ / 3
    const height_delta = height_ / 5

    const x_ = start[0] + delta + width_delta / 2
    const y_ = start[1]
    width_ = width_ - width_delta
    height_ = height_ - height_delta
    return [x_, y_, width_, height_]
  } else if (width < height) {

    // console.log("width < height")

    const delta = parseInt(Math.round((width - height) / 2))
    const y_min = y - delta - diff
    const y_max = y + height + delta + diff
    const x_min = x - diff
    const x_max = x + width + diff

    let width_ = x_min - x_max
    const width_delta = width_ / 3

    const x_ = start[0] + width_delta / 2
    const y_ = start[1] + delta
    width_ = width_ - width_delta
    const height_ = y_min - y_max
    return [x_, y_, width_, height_]
  }
}

const renderPrediction = async () => {
  tf.engine().startScope()
  const returnTensors = false;
  const flipHorizontal = false;
  const annotateBoxes = false;
  const predictions = await detection_model_video.estimateFaces(video, returnTensors, flipHorizontal, annotateBoxes);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
      }

      // const start = predictions[i].topLeft;
      // const end = predictions[i].bottomRight;
      // const size = [end[0] - start[0], end[1] - start[1]];
      // ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
      // ctx.fillRect(start[0], start[1], size[0], size[1]);

      // ctx.beginPath();
      // ctx.rect(start[0], start[1], size[0], size[1]);
      // ctx.lineWidth = "3";
      // ctx.strokeStyle = "#48dbfb";
      // ctx.stroke();

      // ctx.fillStyle = "#48dbfb";
      // ctx.fillRect(start[0]-1, start[1]-15, 100, 15);

      // ctx.fillStyle = "#FFF";
      // ctx.fillText("Mask", start[0]+3, start[1]-4,);



      // console.log(width_)
      // console.log(height_)

      let x_, y_, width_, height_
      [x_, y_, width_, height_] = preprocessPrediction(predictions[i]);

      const img_width = video.width
      const img_height = video.height

      const x_normed = x_ / img_width
      const y_normed = y_ / img_height
      const width_normed = width_ / img_width
      const height_normed = height_ / img_height

      const img_tensor = tf.browser.fromPixels(video)

      const reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
      const resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [128, 128])
      const normed = resized.div(255.0)
      const prediction = classification_model_video.predict(normed).dataSync()
      tf.dispose()
      prediction_class = prediction.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
      predictions[i].predictionClass = prediction_class
      // console.log(prediction)
      var color = ""
      var title = ""
      if (prediction_class == 0) {
        color = "rgba(255, 0, 0, 1)"
        title = "No mask"
      } else if (prediction_class == 1) {
        color = "#2ecc71"
        title = "OK mask"
      } else {
        color = "rgba(255, 165, 0, 1)"
        title = "Bad mask"
      }
      title = title + " - " + prediction[prediction_class].toFixed(2);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = "2";
      ctx.strokeRect(x_, y_, width_, height_);

      var width_title = ctx.measureText(title).width;

      ctx.fillStyle = color;
      ctx.fillRect(x_ - 1, y_ - 17, width_title + 10, 17);
      ctx.fillStyle = "#FFF";
      ctx.font = "13px Helvetica";
      ctx.fillText(title, x_ + 5, y_ - 5);
    }
    canvas.data = predictions.map((prediction) => preprocessPrediction(prediction).concat([prediction.predictionClass]))
  }
  tf.engine().endScope()
  requestAnimationFrame(renderPrediction);
};

const state = {
  backend: 'webgl'
};

const setupPage = async () => {
  // await tf.setBackend(state.backend);
  await setupCamera();
  video.play();

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;


  // containers = document.getElementsByClassName("container");

  // for (i = 0; i < containers.length; i++) {
  //   containers[i].setAttribute("width", videoWidth);
  // }


  canvas = document.getElementById('videoCanvas');

  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  // detection_model_video = await blazeface.load();
  // classification_model_video = await tf.loadLayersModel('/html/assets/classification_model/model.json');
  // console.log(classification_model)

  renderPrediction();
};

async function load_detection_model() {
  detection_model_video = await blazeface.load();
}

async function load_classification_model() {
  classification_model_video = await tf.loadLayersModel("https://pierrebrgit.github.io/facemask/html//assets/classification_model/model.json");
}



Promise.all([load_detection_model(), load_classification_model()])

function capture() {
  var canvas = document.getElementById('canvas');
  var video = document.getElementById('videoElement');
  var videoCanvas = document.getElementById('videoCanvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const predictions = videoCanvas.data
  predictions.forEach((prediction,index) => {
    if (prediction[4] === 0) {
      var height_preview = prediction[3]
      var width_preview = prediction[2]
      canvas.getContext('2d').drawImage(video, prediction[0]-0.5*width_preview, prediction[1]-0.25*height_preview, 2*width_preview, 1.5*height_preview, index*100, 0, 100, 140);
    }
  })
}

 setupPage();
