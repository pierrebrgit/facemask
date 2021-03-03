let detection_model, classification_model

async function load_models() {
  console.log("Loading detection model...");
  detection_model = await blazeface.load();

  console.log("Loading classification model...");
  classification_model = await tf.loadLayersModel('/html/assets/classification_model/model.json');
}

load_models();
