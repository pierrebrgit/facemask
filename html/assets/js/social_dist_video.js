let detection_model_video, classification_model_video, ctx_img3, ctx_img_4, videoWidth, videoHeight, video, canvas;

var max_dist = document.getElementById('dist_mand');

function updateTextInput(val) {
  document.getElementById('textInput').value=val; 
}

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

const renderPrediction = async () => {
  tf.engine().startScope()
  const returnTensors = false;
  const flipHorizontal = false;
  const annotateBoxes = false;
  const predictions = await detection_model_video.estimateFaces(video, returnTensors, flipHorizontal);
  const img_width = video.width
  const img_height = video.height

  if (predictions.length > 0) {

    
    /* constants for socialdistanciation */ 
    const focal = 550 ;/* pixels */ 
    const mean_dim_face = 24.65;    // mean between women and men
    var people_coord = [];
    var social_dist_array = Array(predictions.length).fill().map(() => Array(predictions.length).fill(0.0)); 
    var counter = 0 ;
    const pixels_face = predictions[0].bottomRight[1]-predictions[0].topLeft[1] ;
    const ratio_pix_cm= (0.8*mean_dim_face)/pixels_face ;
    // var offset = 0.5 * parseInt(predictions[0].bottomRight[0]-predictions[0].topLeft[0]);
    var offset = 0.15 * img_height

    ctx_img_3.clearRect(0, 0, canvas.width, canvas.height);

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

      }
      else if (width < height) {

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

      const img_tensor = tf.browser.fromPixels(video)

      const reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
      const resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [128, 128])
      const normed = resized.div(255.0)
      const prediction = classification_model_video.predict(normed).dataSync()
      tf.dispose()
      const prediction_class = prediction.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
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
      ctx_img_3.strokeStyle = color;
      ctx_img_3.fillStyle = color;
      ctx_img_3.lineWidth = "2";
      ctx_img_3.strokeRect(x_, y_, width_, height_);

      var width_title = ctx_img_3.measureText(title).width;

      ctx_img_3.fillStyle = color;
      ctx_img_3.fillRect(x_ - 1, y_ - 17, width_title + 10, 17);
      ctx_img_3.fillStyle = "#FFF";
      ctx_img_3.font = "13px Helvetica";
      ctx_img_3.fillText(title, x_ + 5, y_ - 5);

      let pixels_face_temp = predictions[i].bottomRight[1]-predictions[i].topLeft[1]
      let ratio_pix_cm_temp= (0.8*mean_dim_face)/pixels_face_temp
      let Z_center_cm = focal*ratio_pix_cm_temp
      let X_center_cm = predictions[i].landmarks[2][0] * ratio_pix_cm
      let Y_center_cm = predictions[i].landmarks[2][1] * ratio_pix_cm
      people_coord.push([X_center_cm, Y_center_cm, Z_center_cm])
      console.log(people_coord)
      counter++
    }

    for (let person1 = 0; person1 < counter; person1++) {
      for (let person2 = 0; person2 < counter; person2++) {
        if (person2 !== person1) {
          social_dist_array[person1][person2] = Math.sqrt((people_coord[person1][0]-people_coord[person2][0])**2+(people_coord[person1][1]-people_coord[person2][1])**2+(people_coord[person1][2]-people_coord[person2][2])**2)
        }
      }
    }
    
    for (let person1 = 0; person1 < counter; person1++) {
      for (let person2 = person1+1; person2 < counter; person2++) {
          
          var x0_1=(predictions[person1].landmarks[2][0]+predictions[person2].landmarks[2][0])/2;
          const x1=predictions[person1].landmarks[2][0];
          const y1=predictions[person1].landmarks[2][1];
          const x2=predictions[person2].landmarks[2][0];
          const y2=predictions[person2].landmarks[2][1];
          if (social_dist_array[person1][person2] <= 100*max_dist.value ) {
            // ctx_img_4.fillStyle = "rgba(255, 0, 0, 1)";
            ctx_img_4 = canvas.getContext('2d');
            ctx_img_4.strokeStyle = "rgba(255, 0, 0, 1)";
            ctx_img_4.fillStyle = "rgba(255, 0, 0, 1)";
            ctx_img_4.beginPath();
            ctx_img_4.moveTo(x1,y1+offset);
            ctx_img_4.lineTo(x2,y2+offset);
            ctx_img_4.stroke();

            ctx_img_4.beginPath();
            ctx_img_4.moveTo(x1,parseInt(y1+offset-0.01*img_height));
            ctx_img_4.lineTo(x1,parseInt(y1+offset+0.01*img_height));
            ctx_img_4.stroke();
            ctx_img_4.beginPath();

            ctx_img_4.moveTo(x2,parseInt(y2+offset+0.01*img_height));
            ctx_img_4.lineTo(x2,parseInt(y2+offset-0.01*img_height));
            ctx_img_4.stroke();
            title = Math.round(social_dist_array[person1][person2]) / 100 + "m"
            var width_title = ctx_img_4.measureText(title).width;

            ctx_img_4.fillRect(parseInt((x1+x2)/2)-5,parseInt((y1+offset+y2+offset)/2)-13, width_title + 10, 17);
            ctx_img_4.fillStyle = "#FFF";
            ctx_img_4.fillText(title, parseInt((x1+x2)/2),parseInt((y1+offset+y2+offset)/2));
            // offset=parseInt(1.5*offset)
            social_dist_array[person2][person1] = 9999.0
          }
        
      }
    }

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



  containers = document.getElementsByClassName("container");

  for (i = 0; i < containers.length; i++) {
    containers[i].setAttribute("width", videoWidth);
  }




  canvas = document.getElementById('videoCanvas');

  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx_img_3 = canvas.getContext('2d');

  ctx_img_3.fillStyle = "rgba(255, 0, 0, 0.5)";


  // detection_model_video = await blazeface.load();
  // classification_model_video = await tf.loadLayersModel('/html/assets/classification_model/model.json');
  // console.log(classification_model)

  renderPrediction();
}

async function load_detection_model() {
  detection_model_video = await blazeface.load();
}

async function load_classification_model() {
  classification_model_video = await tf.loadLayersModel("https://pierrebrgit.github.io/facemask/html//assets/classification_model/model.json");
}

Promise.all([load_detection_model(), load_classification_model()]).then(setupPage);
// setupPage();
