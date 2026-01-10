let model;
let outcome;
let isTraining = false;
let isModelReady = false;
let trainingDataLoaded = false;

// Setup function runs once at start
async function setup() {
  noCanvas(); // No canvas needed for this app

  showStatus("Initializing Bitcoin Prediction...", "info");

  // Wait for TensorFlow.js to be ready
  try {
    await tf.ready();
    console.log("✓ TensorFlow.js ready");
  } catch (error) {
    console.error("TensorFlow initialization error:", error);
  }

  // Try to load data from Fear & Greed API
  const success = await loadFearGreedData();

  if (!success) {
    // Fallback to CSV if API fails
    showStatus("API unavailable, loading backup data...", "warning");
    await loadCSVFallback();
  }

  // Set up UI buttons
  setupButtons();
  
  // Set today's date as default
  setTodaysDate();
}

// Load data from Fear & Greed Index API
async function loadFearGreedData() {
  try {
    showStatus("Fetching Fear & Greed Index data from API...", "info");
    console.log("Fetching Fear & Greed Index data...");

    // Fetch all historical data (limit=0 gets everything)
    const response = await fetch('https://api.alternative.me/fng/?limit=0');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    if (!result.data || result.data.length === 0) {
      throw new Error('No data returned from API');
    }

    console.log(`✓ Loaded ${result.data.length} days of Fear & Greed data`);
    showStatus(`Processing ${result.data.length} days of data...`, "info");

    // Initialize model with API data
    await initializeModelWithAPI(result.data);

    return true;

  } catch (error) {
    console.error('Fear & Greed API error:', error);
    return false;
  }
}

// Initialize ML5 model with Fear & Greed API data
async function initializeModelWithAPI(fearGreedData) {
  // Configure the neural network
  let options = {
    task: "classification",
    debug: false,
  };

  model = ml5.neuralNetwork(options);

  // Transform and add data to model
  let dataCount = 0;
  
  // Process data in reverse (oldest first for better training)
  for (let i = fearGreedData.length - 1; i >= 0; i--) {
    const item = fearGreedData[i];
    
    // Convert timestamp to date
    const date = new Date(parseInt(item.timestamp) * 1000);
    const dateValue = dateToNumeric(date);
    
    // Use Fear & Greed value as proxy for price/volume
    // Scale the 0-100 index value to realistic ranges
    const fgValue = parseInt(item.value);
    const rate = 20000 + (fgValue * 1000); // Scale to ~20k-120k range
    const volume = fgValue * 500000000; // Scale to billions
    
    // Add to model
    let inputs = {
      date: dateValue,
      rate: rate,
      volume: volume
    };
    
    let outputs = {
      prediction: item.value_classification
    };
    
    model.addData(inputs, outputs);
    dataCount++;
  }

  console.log(`✓ Added ${dataCount} data points to model`);
  
  // Wait a moment for TensorFlow to be fully ready, then normalize
  setTimeout(() => {
    try {
      model.normalizeData();
      trainingDataLoaded = true;
      isModelReady = true;
      showStatus(`Model ready with ${dataCount} data points! Click 'Train Model' to begin.`, "success");
    } catch (error) {
      console.error("Error normalizing data:", error);
      showStatus("Error initializing model. Please refresh the page.", "error");
    }
  }, 100);
}

// Fallback: Load CSV data if API fails
async function loadCSVFallback() {
  return new Promise((resolve) => {
    let options = {
      dataUrl: "dataset_btc_fear_greed_copy.csv",
      inputs: ["date", "volume", "rate"],
      outputs: ["prediction"],
      task: "classification",
      debug: false,
    };

    model = ml5.neuralNetwork(options, () => {
      console.log("✓ Model loaded with CSV backup data");
      model.normalizeData();
      trainingDataLoaded = true;
      isModelReady = true;
      showStatus("Model ready with backup data! Click 'Train Model' to begin.", "success");
      resolve();
    });
  });
}

// Set up UI buttons
function setupButtons() {
  // Fetch live data button
  const fetchButton = select("#fetchData");
  if (fetchButton) {
    fetchButton.mousePressed(fetchLiveData);
  }

  // Train button
  const trainButton = select("#train");
  if (trainButton) {
    trainButton.mousePressed(trainModel);
  }

  // Predict button
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

// Fetch live Bitcoin data from multiple APIs
async function fetchLiveData() {
  showStatus("Fetching live Bitcoin data...", "info");

  // Try multiple APIs in sequence
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

  // Try each API in sequence
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

        // Update input fields
        select("#rate").value(Math.round(parsed.price));
        select("#volume").value(Math.round(parsed.volume));

        // Update timestamp
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

  // All APIs failed
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

  // Get input values
  const dateStr = select("#date").value();
  const rateStr = select("#rate").value();
  const volumeStr = select("#volume").value();

  // Validate inputs
  if (!dateStr || !rateStr || !volumeStr) {
    showStatus("Please fill in all fields (or click 'Fetch Live Data')", "error");
    return;
  }

  // Convert to numeric values
  const dateValue = dateToNumeric(dateStr);
  const rateValue = parseFloat(rateStr);
  const volumeValue = parseFloat(volumeStr);

  // Validate
  if (isNaN(dateValue) || isNaN(rateValue) || isNaN(volumeValue)) {
    showStatus("Invalid input values. Please check your data.", "error");
    return;
  }

  console.log("Classifying with:", { date: dateValue, volume: volumeValue, rate: rateValue });

  let userInputs = {
    date: dateValue,
    volume: volumeValue,
    rate: rateValue,
  };

  showStatus("Making prediction...", "info");
  model.classify(userInputs, gotResults);
}

// Convert date string to numeric value (days since 2018-01-01)
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

  // Find prediction with highest confidence
  let topPrediction = results[0];
  for (let result of results) {
    if (result.confidence > topPrediction.confidence) {
      topPrediction = result;
    }
  }

  const label = topPrediction.label;
  const confidence = (topPrediction.confidence * 100).toFixed(1);

  // Generate advice based on prediction
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

  // Display result
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
  
  // Remove all status classes individually (no spaces!)
  statusDiv.removeClass("status-info");
  statusDiv.removeClass("status-success");
  statusDiv.removeClass("status-warning");
  statusDiv.removeClass("status-error");

  if (type) {
    statusDiv.addClass(`status-${type}`);
  }
}