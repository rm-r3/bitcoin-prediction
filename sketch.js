let model;
let isTraining = false;
let isModelReady = false;
let trainingDataLoaded = false;

async function setup() {
  noCanvas();
  showStatus("Initializing...", "info");

  // Initialize TensorFlow
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log("✓ TensorFlow ready:", tf.getBackend());
  } catch (error) {
    await tf.setBackend('cpu');
    await tf.ready();
    console.log("✓ TensorFlow ready (CPU)");
  }

  await new Promise(resolve => setTimeout(resolve, 500));
  await loadAndProcessCSV();
  setupButtons();
  setTodaysDate();
}

function loadAndProcessCSV() {
  return new Promise((resolve, reject) => {
    showStatus("Loading training data from CSV...", "info");
    
    Papa.parse("dataset_btc_fear_greed_copy.csv", {
      download: true,
      header: true,
      dynamicTyping: true,
      complete: function(results) {
        console.log(`✓ Loaded ${results.data.length} rows from CSV`);
        
        model = ml5.neuralNetwork({
          task: 'classification',
          debug: false,
          inputs: 3,
          outputs: ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
        });
        
        console.log("✓ Created model: 3 inputs, 5 outputs");
        
        let addedCount = 0;
        results.data.forEach(row => {
          if (row.date && row.volume && row.rate && row.prediction) {
            let dateNum = dateToNumeric(row.date);
            model.addData(
              [dateNum, row.volume, row.rate],
              [row.prediction]
            );
            addedCount++;
          }
        });
        
        console.log(`✓ Added ${addedCount} training examples`);
        
        setTimeout(() => {
          model.normalizeData();
          trainingDataLoaded = true;
          isModelReady = true;
          showStatus(`Ready! ${addedCount} examples loaded.`, "success");
          console.log("✓ Ready to train");
          resolve();
        }, 500);
      },
      error: function(error) {
        console.error("CSV error:", error);
        showStatus("Error loading CSV", "error");
        reject(error);
      }
    });
  });
}

function dateToNumeric(dateStr) {
  const parts = dateStr.split('-');
  if (parts.length === 3) {
    const year = parseInt(parts[0]);
    const month = parseInt(parts[1]);
    const day = parseInt(parts[2]);
    const inputDate = new Date(year, month - 1, day);
    const referenceDate = new Date(2018, 0, 1);
    const daysDiff = Math.floor((inputDate - referenceDate) / (1000 * 60 * 60 * 24));
    return daysDiff;
  }
  return 0;
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
    showStatus("Already training...", "warning");
    return;
  }

  if (!isModelReady || !trainingDataLoaded) {
    showStatus("Model not ready", "warning");
    return;
  }

  isTraining = true;
  select("#train").html("Training...");
  select("#train").style("pointer-events", "none");
  showStatus("Training neural network (20-40 seconds)...", "info");

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

  showStatus("Training complete! Click 'Predict'.", "success");
}

async function fetchLiveData() {
  showStatus("Fetching live Bitcoin data...", "info");

  const apis = [
    {
      name: "CoinGecko",
      url: "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true",
      parse: (data) => ({
        price: data.bitcoin.usd,
        vol: data.bitcoin.usd_24h_vol,
        change: data.bitcoin.usd_24h_change
      })
    },
    {
      name: "CryptoCompare",
      url: "https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC&tsyms=USD",
      parse: (data) => ({
        price: data.RAW.BTC.USD.PRICE,
        vol: data.RAW.BTC.USD.VOLUME24HOURTO,
        change: data.RAW.BTC.USD.CHANGEPCT24HOUR
      })
    }
  ];

  for (let api of apis) {
    try {
      console.log(`Trying ${api.name}...`);
      const response = await fetch(api.url);

      if (response.ok) {
        const data = await response.json();
        const parsed = api.parse(data);

        select("#rate").value(Math.round(parsed.price));
        select("#volume").value(Math.round(parsed.vol));

        const now = new Date();
        const changeText = parsed.change ? ` | 24h: ${parsed.change.toFixed(2)}%` : '';
        select("#lastUpdate").html(
          `${api.name} - $${Math.round(parsed.price).toLocaleString()}${changeText}`
        );

        showStatus(`Live data from ${api.name}`, "success");
        console.log(`✓ ${api.name} OK`);
        return;
      }
    } catch (error) {
      console.log(`✗ ${api.name} failed`);
      continue;
    }
  }

  console.error("All APIs failed");
  showStatus("API unavailable. Using sample data.", "warning");
  select("#rate").value("90000");
  select("#volume").value("15000000000");
}

function classify() {
  if (!isModelReady) {
    showStatus("Model not ready", "error");
    return;
  }

  const dateStr = select("#date").value();
  const rateStr = select("#rate").value();
  const volStr = select("#volume").value();

  if (!dateStr || !rateStr || !volStr) {
    showStatus("Please fill all fields", "error");
    return;
  }

  const dateValue = dateToNumeric(dateStr);
  const rateValue = parseFloat(rateStr);
  const volValue = parseFloat(volStr);

  if (isNaN(dateValue) || isNaN(rateValue) || isNaN(volValue)) {
    showStatus("Invalid values", "error");
    return;
  }

  console.log("Predicting:");
  console.log("  Date:", dateStr, "→", dateValue);
  console.log("  Rate:", rateValue);
  console.log("  Volume:", volValue);

  let userInputs = [dateValue, volValue, rateValue];

  console.log("Sending:", userInputs);
  showStatus("Making prediction...", "info");
  
  model.classify(userInputs, gotResults);
}

function gotResults(error, results) {
  if (error) {
    console.error("Prediction error:", error);
    showStatus("Prediction failed: " + error.message, "error");
    return;
  }

  console.log("Results:", results);

  if (!Array.isArray(results) || results.length === 0) {
    showStatus("No results", "error");
    return;
  }

  const hasValid = results.some(r => r.label && r.confidence !== undefined);
  if (!hasValid) {
    showStatus("Invalid results", "error");
    return;
  }

  let topPrediction = results[0];
  for (let result of results) {
    if (result.confidence > topPrediction.confidence) {
      topPrediction = result;
    }
  }

  const label = topPrediction.label;
  const confidence = (topPrediction.confidence * 100).toFixed(1);

  console.log(`✓ ${label} (${confidence}%)`);

  let advice = "";
  let emoji = "";

  switch (label) {
    case "Extreme Fear":
      advice = "Buy the dip → STRONG BUY";
      emoji = "🔥";
      break;
    case "Fear":
      advice = "Good entry point → BUY";
      emoji = "😰";
      break;
    case "Neutral":
      advice = "Market stable → HOLD";
      emoji = "😐";
      break;
    case "Greed":
      advice = "Consider taking profits";
      emoji = "😎";
      break;
    case "Extreme Greed":
      advice = "Buy low, sell high → SELL";
      emoji = "🤑";
      break;
    default:
      advice = "Unable to determine";
      emoji = "❓";
  }

  const resultHTML = `
    <div style="text-align:center;padding:20px;">
      <div style="font-size:4em;">${emoji}</div>
      <div style="font-size:2em;margin:10px 0;">${label}</div>
      <div style="font-size:1.5em;margin:10px 0;">Confidence: ${confidence}%</div>
      <div style="font-size:1.3em;margin:15px 0;">${advice}</div>
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