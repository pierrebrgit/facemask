async function main() {
  // Load the model.
  const model = await blazeface.load();
  const classification = await tf.loadLayersModel('/html/assets/classification_model/model.json');
  // console.log(classification)


  // Pass in an image or video to the model. The model returns an array of
  // bounding boxes, probabilities, and landmarks, one for each detected face.

  const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
  const predictions = await model.estimateFaces(document.querySelector("#imageResult"), returnTensors);


  var img = document.getElementById('imageResult');
  var img_width = img.clientWidth;
  var img_height = img.clientHeight;

  const originalTensor = tf.browser.fromPixels(img)
  console.log(originalTensor)
  // console.log(typeof (originalTensor))


  canvas = document.getElementById('myCanvas');
  canvas.width = img_width;
  canvas.height = img_height;
  ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(255, 0, 0, 0.5)";


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

    // console.log(predictions)

    // save predictions to test_face_extract folder
    // predictions.forEach(async (prediction, index) => {

    //   const [y1, x1] = prediction.topLeft;
    //   const [y2, x2] = prediction.bottomRight;

    //   const x1s = Math.floor(x1);
    //   const y1s = Math.floor(y1);
    //   const x2s = Math.floor(x2);
    //   const y2s = Math.floor(y2);

    //   const faceTensor = originalTensor.slice([x1s, y1s], [x2s - x1s, y2s - y1s]);

    //   faces.push(faceTensor);

    // })

    faces = []

    for (let i = 0; i < predictions.length; i++) {

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

        x_ = start[0] + delta
        y_ = start[1]
        width_ = x_min - x_max
        height_ = y_min - y_max

      } else if (width < height) {

        delta = parseInt(Math.round((width - height) / 2))
        y_min = y - delta - diff
        y_max = y + height + delta + diff
        x_min = x - diff
        x_max = x + width + diff

        x_ = start[0]
        y_ = start[1] + delta
        width_ = x_min - x_max
        height_ = y_min - y_max

      }

      ctx.fillRect(x_, y_, width_, height_);


      x_ = x_ / img_width
      y_ = y_ / img_height
      width_ = width_ / img_width
      height_ = height_ / img_height


      reshaped = originalTensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
      resized = tf.image.cropAndResize(reshaped, [[y_, x_, y_ + height_, x_ + width_]], [0], [128, 128])
      normed = resized.div(255.0)


      canvas2 = document.getElementById('canvas2');
      canvas2.width = 128;
      canvas2.height = 128;

      final = tf.squeeze(normed)
      pixel = tf.browser.toPixels(final, canvas2)



      prediction = classification.predict(normed).dataSync()
      console.log(prediction)



    }
  }
}

main();
