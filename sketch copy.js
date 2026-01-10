let neuralModel;
let isTraining = false;
let isModelReady = false;
let dataLoaded = false;

async function setup() {
  noCanvas();
  
  console.log("Starting setup...");
  
  // Wait for TensorFlow to be ready
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log("✓ TensorFlow ready:", tf.getBackend());
  } catch (e) {
    try {
      await tf.setBackend('cpu');
      await tf.ready();
      console.log("✓ TensorFlow ready (CPU)");
    } catch (err) {
      console.error("TensorFlow failed:", err);
    }
  }
  
  // Extra delay for stability
  await new Promise(r => setTimeout(r, 1000));
  
  // Load CSV data
  await loadCSVData();
  
  // Setup buttons
  setupButtons();
  setDate();
}

function loadCSVData() {
  return new Promise((resolve, reject) => {
    console.log("Loading CSV...");
    
    Papa.parse("dataset_btc_fear_greed_copy.csv", {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: function(results) {
        console.log(`✓ Loaded ${results.data.length} rows from CSV`);
        
        // Create empty model
        neuralModel = ml5.neuralNetwork({
          task: 'classification',
          debug: false,
          inputs: 3,
          outputs: ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
        });
        
        console.log("✓ Created model with 3 inputs, 5 outputs");
        
        // Add data
        let count = 0;
        results.data.forEach(row => {
          if (row.date && row.volume && row.rate && row.prediction) {
            let dateNum = convertDate(row.date);
            neuralModel.addData(
              [dateNum, row.volume, row.rate],
              [row.prediction]
            );
            count++;
          }
        });
        
        console.log(`✓ Added ${count} training examples to model`);
        
        // Normalize after delay
        setTimeout(() => {
          neuralModel.normalizeData();
          dataLoaded = true;
          isModelReady = true;
          console.log("✓ Ready to train");
          resolve();
        }, 500);
      },
      error: function(err) {
        console.error("CSV error:", err);
        reject(err);
      }
    });
  });
}

function convertDate(dateStr) {
  const parts = dateStr.split('-');
  if (parts.length === 3) {
    const y = parseInt(parts[0]);
    const m = parseInt(parts[1]);
    const d = parseInt(parts[2]);
    const inputDate = new Date(y, m - 1, d);
    const refDate = new Date(2018, 0, 1);
    const diff = Math.floor((inputDate - refDate) / (1000 * 60 * 60 * 24));
    return diff;
  }
  return 0;
}

function setupButtons() {
  const trainBtn = select("#train");
  if (trainBtn) {
    trainBtn.mousePressed(startTraining);
  }

  const predictBtn = select("#predict");
  if (predictBtn) {
    predictBtn.mousePressed(makePrediction);
  }
}

function setDate() {
  const today = new Date();
  const dateStr = today.toISOString().split('T')[0];
  select("#date").value(dateStr);
}

function startTraining() {
  if (isTraining) {
    return;
  }

  if (!isModelReady || !dataLoaded) {
    console.log("Model not ready");
    return;
  }

  isTraining = true;
  select("#train").html("Training...");
  
  console.log("Starting training...");

  let opts = {
    epochs: 32,
    batchSize: 32,
  };

  neuralModel.train(opts, trainingProgress, trainingDone);
}

function trainingProgress(epoch, loss) {
  const lossVal = loss.loss ? loss.loss.toFixed(4) : "...";
  console.log(`Epoch: ${epoch} - Loss: ${lossVal}`);
}

function trainingDone() {
  console.log("✓ Training complete!");
  isTraining = false;
  select("#train").html("Trained");
  select("#train").style("background-color", "#31fa03");
}

function makePrediction() {
  if (!isModelReady) {
    console.log("Model not ready");
    return;
  }

  const dateStr = select("#date").value();
  const priceStr = select("#rate").value();
  const volStr = select("#volume").value();

  if (!dateStr || !priceStr || !volStr) {
    console.log("Please fill all fields");
    return;
  }

  const dateVal = convertDate(dateStr);
  const priceVal = parseFloat(priceStr);
  const volVal = parseFloat(volStr);

  if (isNaN(dateVal) || isNaN(priceVal) || isNaN(volVal)) {
    console.log("Invalid values");
    return;
  }

  console.log("Predicting:");
  console.log("  Date:", dateStr, "→", dateVal);
  console.log("  Price:", priceVal);
  console.log("  Volume:", volVal);

  // Send as array
  let inputs = [dateVal, volVal, priceVal];

  console.log("Inputs:", inputs);
  
  neuralModel.classify(inputs, handleResults);
}

function handleResults(error, results) {
  if (error) {
    console.error("Error:", error);
    select("#result").html("Prediction failed");
    return;
  }

  console.log("Results:", results);

  if (!Array.isArray(results) || results.length === 0) {
    select("#result").html("No results");
    return;
  }

  const valid = results.some(r => r.label && r.confidence !== undefined);
  if (!valid) {
    select("#result").html("Invalid results");
    return;
  }

  // Find top prediction
  let top = results[0];
  for (let r of results) {
    if (r.confidence > top.confidence) {
      top = r;
    }
  }

  const label = top.label;
  const conf = (top.confidence * 100).toFixed(1);

  console.log(`✓ ${label} (${conf}%)`);

  let advice = "";
  let emoji = "";

  if (label === "Extreme Fear") {
    advice = "Buy the dip → STRONG BUY";
    emoji = "🔥";
  } else if (label === "Fear") {
    advice = "Good entry point → BUY";
    emoji = "😰";
  } else if (label === "Neutral") {
    advice = "Market stable → HOLD";
    emoji = "😐";
  } else if (label === "Greed") {
    advice = "Consider profits";
    emoji = "😎";
  } else if (label === "Extreme Greed") {
    advice = "Sell high → SELL";
    emoji = "🤑";
  } else {
    advice = "Unknown";
    emoji = "❓";
  }

  const html = `
    <div style="text-align:center;padding:20px;font-size:1.2em;">
      <div style="font-size:3em;">${emoji}</div>
      <div style="font-size:1.8em;margin:10px 0;">${label}</div>
      <div style="margin:10px 0;">Confidence: ${conf}%</div>
      <div style="margin:15px 0;">${advice}</div>
    </div>
  `;

  select("#result").html(html);
}