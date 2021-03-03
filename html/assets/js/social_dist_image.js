async function distanciation_social_mask() {
    console.log("detect_image()");
    // await tf.setBackend(state.backend);
    const detection_model_image = await blazeface.load();
    const classification_model_image = await tf.loadLayersModel('/html/assets/classification_model/model.json');
    // console.log(classification_model_ing)
  
    const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
    console.log("Calling detection model");
    const predictions = await detection_model_image.estimateFaces(document.querySelector("#imageResult"), returnTensors);
    console.log("predictions done");
  
    const img = document.getElementById('imageResult');
    var img_width = img.clientWidth;
    var img_height = img.clientHeight;
  
    const img_tensor = tf.browser.fromPixels(img)
  
  
    canvas_img_2 = document.getElementById('myCanvas');
    canvas_img_2.width = img_width;
    canvas_img_2.height = img_height;
    ctx_img_2 = canvas_img_2.getContext('2d');
  
  
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
  
      /* constants for socialdistanciation */ 
      const focal = 1.18 ;/* pixels */ 
      const mean_dim_face = 24.65;    // mean between women and men
      var people_coord = Array(predictions.length);
      var social_dist_array = Array(predictions.length).fill().map(() => Array(predictions.length).fill(0));
      var counter = 1 ;
      const color_wrong = "rgba(255, 0, 0, 1)" ;
      const pixels_face = predictions[0].bottomRight[1]-predictions[0].topLeft[1] ;
      const ratio_pix_cm= (0.8*mean_dim_face)/pixels_face ;
      
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
  
  
  
        x_normed = x_ / img_width
        y_normed = y_ / img_height
        width_normed = width_ / img_width
        height_normed = height_ / img_height
  
  
        reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
        resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [128, 128])
        normed = resized.div(255.0)
  
        prediction = classification_model_image.predict(normed).dataSync()
        prediction_class = prediction.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
  
        console.log(prediction)
  
  
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
  
        ctx_img_2.strokeStyle = color;
        ctx_img_2.fillStyle = color;
        ctx_img_2.lineWidth = "2";
        ctx_img_2.strokeRect(x_, y_, width_, height_);
  
        ctx_img_2.fillStyle = color;
        ctx_img_2.font = "100 15px Helvetica";
        ctx_img_2.fillText(title, x_ + 5, y_ - 5);
  
        
  
  
        // canvas2 = document.getElementById('canvas2');
        // canvas2.width = 128;
        // canvas2.height = 128;
        // final = tf.squeeze(normed)
        // pixel = tf.browser.toPixels(final, canvas2)
        
        let pixels_face_temp = predictions[i].bottomRight[1]-predictions[i].topLeft[1]
        let ratio_pix_cm_temp= (0.8*mean_dim_face)/pixels_face_temp
        let Z_center_cm = focal*ratio_pix_cm_temp
        let X_center_cm = predictions[i].landmarks[2][0] * ratio_pix_cm
        let Y_center_cm = predictions[i].landmarks[2][1] * ratio_pix_cm
        people_coord.push([X_center_cm, Y_center_cm, Z_center_cm])
        console.log(people_coord)
        counter++
      }
    }
    console.log(people_coord)
  
    for (let person1 = 0; person1 < counter; person1++) {
      for (let person2 = 0; person2 < counter; person2++) {
        if (person2 != person1) {
          social_dist_array[person1][person2] = Math.sqrt((people_coord[person1][0]-people_coord[person2][0])**2+(people_coord[person1][1]-people_coord[person2][1])**2+(people_coord[person1][2]-people_coord[person2][2])**2)
        }
      }
    }
  }
  