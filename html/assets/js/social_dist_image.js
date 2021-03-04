var max_dist = document.getElementById('dist_mand');

function updateTextInput(val) {
  document.getElementById('textInput').value=val; 
}

async function distanciation_social_mask() {
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
  
  
    canvas_img_2 = document.getElementById('myCanvas');
    canvas_img_2.width = img_width;
    canvas_img_2.height = img_height;
    ctx_img_2 = canvas_img_2.getContext('2d');
  
  
    if (predictions.length > 0) {
    
      /* constants for socialdistanciation */ 
      const focal = 1.18 ;/* pixels */ 
      const mean_dim_face = 24.65;    // mean between women and men
      var people_coord = [];
      var social_dist_array = Array(predictions.length).fill().map(() => Array(predictions.length).fill(0.0)); 
      var counter = 0 ;
      const pixels_face = predictions[0].bottomRight[1]-predictions[0].topLeft[1] ;
      const ratio_pix_cm= (0.8*mean_dim_face)/pixels_face ;
      // var offset = 0.5 * parseInt(predictions[0].bottomRight[0]-predictions[0].topLeft[0]);
      var offset = 0.15 * img_height
      
      for (let i = 0; i < predictions.length; i++) {
  
        const start = predictions[i].topLeft;
        const end = predictions[i].bottomRight;
  
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
        const img_width = img.width
        const img_height = img.height
  
        const x_normed = x_ / img_width
        const y_normed = y_ / img_height
        const width_normed = width_ / img_width
        const height_normed = height_ / img_height
  
        const img_tensor = tf.browser.fromPixels(img)
  
        const reshaped = img_tensor.reshape([1, Math.floor(img_height), Math.floor(img_width), 3])
        const resized = tf.image.cropAndResize(reshaped, [[y_normed, x_normed, y_normed + height_normed, x_normed + width_normed]], [0], [128, 128])
        const normed = resized.div(255.0)
        const prediction = classification_model_image.predict(normed).dataSync()
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
        ctx_img_2.strokeStyle = color;
        ctx_img_2.fillStyle = color;
        ctx_img_2.lineWidth = "2";
        ctx_img_2.strokeRect(x_, y_, width_, height_);
  
        var width_title = ctx_img_2.measureText(title).width;
  
        ctx_img_2.fillStyle = color;
        ctx_img_2.fillRect(x_ - 1, y_ - 17, width_title + 10, 17);
        ctx_img_2.fillStyle = "#FFF";
        ctx_img_2.font = "13px Helvetica";
        ctx_img_2.fillText(title, x_ + 5, y_ - 5);
        
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
            
            if (social_dist_array[person1][person2] <= 100*max_dist.value) {
              ctx_img_3 = canvas_img_2.getContext('2d');
              ctx_img_3.strokeStyle = "rgba(255, 0, 0, 1)";
              ctx_img_3.fillStyle = "rgba(255, 0, 0, 1)";
              ctx_img_3.beginPath();
              ctx_img_3.moveTo(x1,y1+offset);
              ctx_img_3.lineTo(x2,y2+offset);
              ctx_img_3.stroke();
              ctx_img_3.beginPath();
              ctx_img_3.moveTo(x1,parseInt(y1+offset-0.01*img_height));
              ctx_img_3.lineTo(x1,parseInt(y1+offset+0.01*img_height));
              ctx_img_3.stroke();
              ctx_img_3.beginPath();
              ctx_img_3.moveTo(x2,parseInt(y2+offset+0.01*img_height));
              ctx_img_3.lineTo(x2,parseInt(y2+offset-0.01*img_height));
              ctx_img_3.stroke();
              title = Math.round(social_dist_array[person1][person2]) / 100 + "m"
              var width_title = ctx_img_3.measureText(title).width;
              ctx_img_2.fillStyle = color;
              ctx_img_3.fillRect(parseInt((x1+x2)/2)-5,parseInt((y1+offset+y2+offset)/2)-13, width_title + 10, 17);
              ctx_img_3.fillStyle = "#FFF";
              ctx_img_3.fillText(title, parseInt((x1+x2)/2),parseInt((y1+offset+y2+offset)/2));
              // offset=parseInt(1.5*offset)
              social_dist_array[person2][person1] = 9999.0
          }
        
      }
    }
  }
