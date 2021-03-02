async function setupCamera() {
  video = document.getElementById('videoElement');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction = async () => {
  const returnTensors = false;
  const flipHorizontal = false;
  const annotateBoxes = false;
  const predictions = await model.estimateFaces(
    video, returnTensors, flipHorizontal, annotateBoxes);

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

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      x = start[0]
      y = start[1]
      width = start[0] - end[0]
      height = start[1] - end[1]
      diff = 0
      if (height < width) {
        delta = parseInt(Math.round((height - width) / 2))
        y_min = y - diff - delta
        y_max = y + height + diff
        x_min = x - delta - diff
        x_max = x + width + delta + diff
        width_ = x_min - x_max
        width_delta = width_ / 3
        height_delta = width_ / 10
        x_ = start[0] + delta + width_delta / 2
        y_ = start[1]
        width_ = width_ - width_delta
        height_ = y_min - y_max
      } else if (width < height) {
        delta = parseInt(Math.round((width - height) / 2))
        y_min = y - delta - diff
        y_max = y + height + delta + diff
        x_min = x - diff
        x_max = x + width + diff
        width_ = x_min - x_max
        width_delta = width_ / 3
        height_delta = width_ / 10
        x_ = start[0] + width_delta / 2
        y_ = start[1] + delta
        width_ = width_ - width_delta
        height_ = y_min - y_max
      }
      // console.log(width_)
      // console.log(height_)
      img_width = video.width
      img_height = video.height

      x_normed = x_ / img_width
      y_normed = y_ / img_height
      width_normed = width_ / img_width
      height_normed = height_ / img_height

      const img_tensor = tf.browser.fromPixels(video)

      reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
      resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [128, 128])
      normed = resized.div(255.0)
      prediction = classification_model.predict(normed).dataSync()
      prediction_class = prediction.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
      // console.log(prediction)
      var color = ""
      var title = ""
      if (prediction_class == 0) {
        color = "rgba(255, 0, 0, 1)"
        title = "no mask"
      } else if (prediction_class == 1) {
        color = "rgba(0, 255, 0, 1)"
        title = "ok mask"
      } else {
        color = "rgba(255, 165, 0, 1)"
        title = "bad mask"
      }
      title = title + " - " + prediction[prediction_class].toFixed(2);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = "2";
      ctx.strokeRect(x_, y_, width_, height_);
      ctx.fillStyle = color;
      ctx.font = "100 15px Helvetica";
      ctx.fillText(title, x_ + 5, y_ - 5);
    }
  }

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('videoCanvas');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

  model = await blazeface.load();
  classification_model = await tf.loadLayersModel('https://pierrebrgit.github.io/facemask/html/assets/classification_model/model.json');
  console.log(classification_model)

  renderPrediction();
};

setupPage();
