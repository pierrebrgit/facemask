
async function detect_image() {
  console.log("detect_image()");
  // await tf.setBackend(state.backend);
  const detection_model_image = await blazeface.load();
  const classification_model_image = await tf.loadLayersModel("https://pierrebrgit.github.io/facemask/html//assets/classification_model/model.json");
  // console.log(classification_model_ing)

  const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
  console.log("Calling detection model");
  const predictions = await detection_model_image.estimateFaces(document.querySelector("#imageResult"), returnTensors);
  console.log("predictions done");

  const img = document.getElementById('imageResult');
  var img_width = img.clientWidth;
  var img_height = img.clientHeight;

  const img_tensor = tf.browser.fromPixels(img)


  canvas_img = document.getElementById('myCanvas');
  canvas_img.width = img_width;
  canvas_img.height = img_height;
  ctx_img = canvas_img.getContext('2d');


  if (predictions.length > 0) {

    /*
    `predictions` is an array of objects describing each detected face, for example:

    [
      {
        topLeft: [232.28, 145.26],
        bottomRight: [449.75, 308.36],
        probability: [0.998],
        landmarks: [
          [295.13, 177.64], // right eye
          [382.32, 175.56], // left eye
          [341.18, 205.03], // nose
          [345.12, 250.61], // mouth
          [252.76, 211.37], // right ear
          [431.20, 204.93] // left ear
        ]
      }
    ]
    */

    for (let i = 0; i < predictions.length; i++) {

      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
      }

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const x = start[0]
      const y = start[1]
      const width = start[0] - end[0]
      const height = start[1] - end[1]
      const diff = 0

      if (height < width) {

        // console.log("height < width")

        const delta = parseInt(Math.round((height - width) / 2))
        const y_min = y - diff - delta
        const y_max = y + height + diff
        const x_min = x - delta - diff
        const x_max = x + width + delta + diff

        width_ = x_min - x_max
        height_ = y_min - y_max
        const width_delta = width_ / 3
        const height_delta = height_ / 5

        x_ = start[0] + delta + width_delta / 2
        y_ = start[1]
        width_ = width_ - width_delta
        height_ = height_ - height_delta

      } else if (width < height) {

        // console.log("width < height")

        const delta = parseInt(Math.round((width - height) / 2))
        const y_min = y - delta - diff
        const y_max = y + height + delta + diff
        const x_min = x - diff
        const x_max = x + width + diff

        width_ = x_min - x_max
        const width_delta = width_ / 3

        x_ = start[0] + width_delta / 2
        y_ = start[1] + delta
        width_ = width_ - width_delta
        height_ = y_min - y_max

      }

      // console.log(width_)
      // console.log(height_)


      const x_normed = x_ / img_width
      const y_normed = y_ / img_height
      const width_normed = width_ / img_width
      const height_normed = height_ / img_height


      const reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
      const resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [128, 128])
      const normed = resized.div(255.0)

      const prediction = classification_model_image.predict(normed).dataSync()
      const prediction_class = prediction.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

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
      ctx_img.strokeStyle = color;
      ctx_img.fillStyle = color;
      ctx_img.lineWidth = "2";
      ctx_img.strokeRect(x_, y_, width_, height_);

      var width_title = ctx_img.measureText(title).width;

      ctx_img.fillStyle = color;
      ctx_img.fillRect(x_ - 1, y_ - 17, width_title + 10, 17);
      ctx_img.fillStyle = "#FFF";
      ctx_img.font = "13px Helvetica";
      ctx_img.fillText(title, x_ + 5, y_ - 5);

      // canvas2 = document.getElementById('canvas2');
      // canvas2.width = 128;
      // canvas2.height = 128;
      // final = tf.squeeze(normed)
      // pixel = tf.browser.toPixels(final, canvas2)


    }
  }
}
