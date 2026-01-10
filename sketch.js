let model;
let outcome;
let isTraining = false;
let isModelReady = false;

function setup() {
  noCanvas(); // We don't need a canvas for this app

  // Configure the neural network
  let options = {
    dataUrl: "dataset_btc_fear_greed_copy.csv",
    inputs: ["date", "volume", "rate"], // Match the CSV column names
    outputs: ["prediction"],
    task: "classification",
    debug: false, // Disable debug to reduce console noise
  };

  // Initialize the model
  model = ml5.neuralNetwork(options, modelReady);

  // Set up the fetch data button
  const fetchButton = select("#fetchData");
  fetchButton.mousePressed(fetchLiveData);

  // Set up the predict button
  const predictButton = select("#predict");
  predictButton.mousePressed(classify);

  // Set up the train button
  const trainButton = select("#train");
  trainButton.mousePressed(trainModel);
  
  // Set today's date as default
  setTodaysDate();
}

function setTodaysDate() {
  const today = new Date();
  const dateStr = today.toISOString().split('T')[0];
  select("#date").value(dateStr);
}

function modelReady() {
  console.log("Model initialized and ready");
  isModelReady = true;
  model.normalizeData();
  showStatus("Model loaded! Click 'Train Model' to start training.", "info");
}

function trainModel() {
  if (isTraining) {
    showStatus("Training already in progress...", "warning");
    return;
  }

  isTraining = true;
  select("#train").html("⏳ Training...");
  showStatus("Training neural network... This may take a moment.", "info");

  let trainOptions = {
    epochs: 50, // Increased for better accuracy
    batchSize: 32,
  };

  model.train(trainOptions, whileTraining, finishedTraining);
}

function whileTraining(epoch, loss) {
  // Show training progress
  const lossValue = loss.loss ? loss.loss.toFixed(4) : "calculating";
  showStatus(`Training... Epoch ${epoch} - Loss: ${lossValue}`, "info");
  console.log(`Epoch: ${epoch} - Loss: ${lossValue}`);
}

function finishedTraining() {
  console.log("Training complete!");
  isTraining = false;
  
  select("#train").html("Trained");
  select("#train").style("background-color", "#31fa03");
  select("#predict").show();
  
  showStatus("Training complete! Enter data and click 'Predict' to see results.", "success");
}

async function fetchLiveData() {
  showStatus("Fetching live Bitcoin data...", "info");
  
  // Try multiple APIs in order for better reliability
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
        volume: data.USD['15m'] * 144 * 1000000, // Approximate 24h volume
        change: 0 // This API doesn't provide change
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
        
        // Update input fields with live data
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
        return; // Success! Exit function
      }
    } catch (error) {
      console.log(`✗ ${api.name} failed:`, error.message);
      continue; // Try next API
    }
  }
  
  // All APIs failed - use sample data
  console.error("All APIs failed - using sample data");
  showStatus("API unavailable. Using sample values. You can edit them manually.", "warning");
  
  // Use realistic sample data
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

  // Convert date to a numeric value (days since Unix epoch)
  const dateValue = dateToNumeric(dateStr);
  const rateValue = parseFloat(rateStr);
  const volumeValue = parseFloat(volumeStr);

  // Validate numeric values
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

function dateToNumeric(dateStr) {
  // Convert date string (YYYY-MM-DD) to numeric value
  // Using days since 2018-01-01 as reference
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

  // Find the prediction with highest confidence
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

  switch(label) {
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
  showStatus("", ""); // Clear status message
}

function showStatus(message, type) {
  const statusDiv = select("#trainStatus");
  if (!message) {
    statusDiv.html("");
    return;
  }
  
  statusDiv.html(message);
  statusDiv.removeClass("status-info status-success status-warning status-error");
  
  if (type) {
    statusDiv.addClass(`status-${type}`);
  }
}