let model;
let outcome;
let isTraining = false;
let isModelReady = false;
let trainingDataLoaded = false;

// Setup function runs once at start
async function setup() {
  noCanvas();

  showStatus("Initializing Bitcoin Prediction...", "info");

  // Force WebGL backend (more stable than WebGPU)
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log("✓ TensorFlow.js ready with backend:", tf.getBackend());
  } catch (error) {
    console.error("TensorFlow initialization error:", error);
    try {
      await tf.setBackend('cpu');
      await tf.ready();
      console.log("✓ TensorFlow.js ready with CPU backend");
    } catch (e) {
      console.error("TensorFlow CPU fallback failed:", e);
    }
  }

  // Extra delay to ensure backend is fully initialized
  await new Promise(resolve => setTimeout(resolve, 500));

  // Just use CSV - it works!
  await loadCSVData();

  setupButtons();
  setTodaysDate();
}

// Load CSV data (this method works!)
async function loadCSVData() {
  return new Promise((resolve) => {
    showStatus("Loading training data...", "info");
    
    let options = {
      dataUrl: "dataset_btc_fear_greed_copy.csv",
      inputs: ["date", "volume", "rate"],
      outputs: ["prediction"],
      task: "classification",
      debug: false,
    };

    model = ml5.neuralNetwork(options, async () => {
      console.log("✓ Model loaded with training data");
      
      await new Promise(r => setTimeout(r, 500));
      
      try {
        model.normalizeData();
        trainingDataLoaded = true;
        isModelReady = true;
        
        // Get the row count from the CSV
        showStatus("Model ready! Click 'Train Model' to begin.", "success");
        console.log("✓ Ready to train");
      } catch (error) {
        console.error("Error normalizing data:", error);
        showStatus("Error loading data. Please refresh.", "error");
      }
      
      resolve();
    });
  });
}

function setupButtons() {
  const fetchButton = select("#fetchData");
  if (fetchButton) {
    fetchButton.mousePressed(fetchLiveData);
  }

  const trainButton = select("#train");
  if (trainButton) {
    trainButton.mousePressed(trainModel);
  }

  const predictButton = select("#predict");
  if (predictButton) {
    predictButton.mousePressed(classify);
  }
}

function setTodaysDate() {
  const today = new Date();
  const dateStr = today.toISOString().split('T')[0];
  select("#date").value(dateStr);
}

function trainModel() {
  if (isTraining) {
    showStatus("Training already in progress...", "warning");
    return;
  }

  if (!isModelReady || !trainingDataLoaded) {
    showStatus("Model not ready yet. Please wait for data to load...", "warning");
    return;
  }

  isTraining = true;
  select("#train").html("Training...");
  select("#train").style("pointer-events", "none");
  showStatus("Starting neural network training... This will take 20-40 seconds.", "info");

  let trainOptions = {
    epochs: 32,
    batchSize: 32,
  };

  model.train(trainOptions, whileTraining, finishedTraining);
}

function whileTraining(epoch, loss) {
  const lossValue = loss.loss ? loss.loss.toFixed(4) : "calculating";
  showStatus(`Training... Epoch ${epoch}/32 - Loss: ${lossValue}`, "info");
  console.log(`Epoch: ${epoch} - Loss: ${lossValue}`);
}

function finishedTraining() {
  console.log("✓ Training complete!");
  isTraining = false;

  select("#train").html("Trained");
  select("#train").style("background-color", "#31fa03");
  select("#train").style("pointer-events", "auto");
  select("#predict").show();

  showStatus("Training complete! Enter data and click 'Predict' to see results.", "success");
}

async function fetchLiveData() {
  showStatus("Fetching live Bitcoin data...", "info");

  const apis = [
    {
      name: "CoinGecko",
      url: "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true",
      parse: (data) => ({
        price: data.bitcoin.usd,
        volume: data.bitcoin.usd_24h_vol,
        change: data.bitcoin.usd_24h_change
      })
    },
    {
      name: "CryptoCompare",
      url: "https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC&tsyms=USD",
      parse: (data) => ({
        price: data.RAW.BTC.USD.PRICE,
        volume: data.RAW.BTC.USD.VOLUME24HOURTO,
        change: data.RAW.BTC.USD.CHANGEPCT24HOUR
      })
    },
    {
      name: "Blockchain.info",
      url: "https://blockchain.info/ticker",
      parse: (data) => ({
        price: data.USD.last,
        volume: data.USD['15m'] * 144 * 1000000,
        change: 0
      })
    }
  ];

  for (let api of apis) {
    try {
      console.log(`Trying ${api.name}...`);

      const response = await fetch(api.url, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        const parsed = api.parse(data);

        select("#rate").value(Math.round(parsed.price));
        select("#volume").value(Math.round(parsed.volume));

        const now = new Date();
        const changeText = parsed.change ? ` | 24h Change: ${parsed.change.toFixed(2)}%` : '';
        select("#lastUpdate").html(
          `Data from ${api.name} at ${now.toLocaleTimeString()} - Price: $${Math.round(parsed.price).toLocaleString()}${changeText}`
        );

        showStatus(`Live data loaded from ${api.name}!`, "success");
        console.log(`✓ ${api.name} succeeded`);
        return;
      }
    } catch (error) {
      console.log(`✗ ${api.name} failed:`, error.message);
      continue;
    }
  }

  console.error("All APIs failed - using sample data");
  showStatus("API unavailable. Using sample values. You can edit them manually.", "warning");

  select("#rate").value("95000");
  select("#volume").value("45000000000");
  select("#lastUpdate").html("Using sample data (APIs unavailable)");
}

function classify() {
  if (!isModelReady) {
    showStatus("Model not ready yet. Please wait...", "error");
    return;
  }

  const dateStr = select("#date").value();
  const rateStr = select("#rate").value();
  const volumeStr = select("#volume").value();

  if (!dateStr || !rateStr || !volumeStr) {
    showStatus("Please fill in all fields (or click 'Fetch Live Data')", "error");
    return;
  }

  const dateValue = dateToNumeric(dateStr);
  const rateValue = parseFloat(rateStr);
  const volumeValue = parseFloat(volumeStr);

  if (isNaN(dateValue) || isNaN(rateValue) || isNaN(volumeValue)) {
    showStatus("Invalid input values. Please check your data.", "error");
    return;
  }

  console.log("Classifying with:");
  console.log("  - Date string:", dateStr, "→ numeric:", dateValue);
  console.log("  - Rate:", rateValue);
  console.log("  - Volume:", volumeValue);

  // Use object format for CSV-loaded model
  let userInputs = {
    date: dateValue,
    volume: volumeValue,
    rate: rateValue
  };

  console.log("Sending to model:", userInputs);

  showStatus("Making prediction...", "info");
  model.classify(userInputs, gotResults);
}

function dateToNumeric(dateStr) {
  const inputDate = new Date(dateStr);
  const referenceDate = new Date("2018-01-01");
  const daysDiff = Math.floor((inputDate - referenceDate) / (1000 * 60 * 60 * 24));
  return daysDiff;
}

function gotResults(error, results) {
  if (error) {
    console.error("Prediction error:", error);
    showStatus("Prediction failed. Please try again.", "error");
    return;
  }

  console.log("Prediction results:", results);

  if (!results || results.length === 0) {
    console.error("No results returned from model");
    showStatus("No prediction results. Please retrain the model.", "error");
    return;
  }

  const hasValidResults = results.some(r => r.label && r.confidence !== undefined);
  if (!hasValidResults) {
    console.error("Results missing label or confidence:", results);
    showStatus("Invalid prediction results. Please retrain the model.", "error");
    return;
  }

  let topPrediction = results[0];
  for (let result of results) {
    if (result.confidence && result.confidence > (topPrediction.confidence || 0)) {
      topPrediction = result;
    }
  }

  if (!topPrediction.label || topPrediction.confidence === undefined) {
    console.error("Top prediction invalid:", topPrediction);
    showStatus("Could not determine prediction. Please retrain the model.", "error");
    return;
  }

  const label = topPrediction.label;
  const confidence = (topPrediction.confidence * 100).toFixed(1);

  console.log(`Prediction: ${label} (${confidence}% confidence)`);

  let advice = "";
  let emoji = "";
  let actionClass = "";

  switch (label) {
    case "Extreme Fear":
      advice = "Buy the dip → STRONG BUY";
      emoji = "🔥";
      actionClass = "extreme-fear";
      break;
    case "Fear":
      advice = "Good entry point → BUY (especially for long-term holders)";
      emoji = "😰";
      actionClass = "fear";
      break;
    case "Neutral":
      advice = "Market is stable → HOLD or wait for clear signals";
      emoji = "😐";
      actionClass = "neutral";
      break;
    case "Greed":
      advice = "Market is heating up → Consider taking profits";
      emoji = "😎";
      actionClass = "greed";
      break;
    case "Extreme Greed":
      advice = "Buy low, sell high → SELL (take profits)";
      emoji = "🤑";
      actionClass = "extreme-greed";
      break;
    default:
      advice = "Unable to determine market sentiment";
      emoji = "❓";
      actionClass = "unknown";
  }

  const resultHTML = `
    <div class="prediction-result ${actionClass}">
      <div class="prediction-header">
        <span class="prediction-emoji">${emoji}</span>
        <span class="prediction-label">${label}</span>
      </div>
      <div class="confidence">Confidence: ${confidence}%</div>
      <div class="advice">${advice}</div>
      <div class="disclaimer">
        Remember: This is educational only, NOT financial advice!
      </div>
    </div>
  `;

  select("#result").html(resultHTML);
  showStatus("", "");
}

function showStatus(message, type) {
  const statusDiv = select("#trainStatus");
  if (!message) {
    statusDiv.html("");
    return;
  }

  statusDiv.html(message);
  
  statusDiv.removeClass("status-info");
  statusDiv.removeClass("status-success");
  statusDiv.removeClass("status-warning");
  statusDiv.removeClass("status-error");

  if (type) {
    statusDiv.addClass(`status-${type}`);
  }
}