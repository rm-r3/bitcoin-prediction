/* sketch.js — Bitcoin Prediction (ml5 + p5 + PapaParse + CoinGecko)
   Robust against ml5 callback/promise signature differences.
*/

let neuralModel = null;

let isTraining = false;
let isModelReady = false; // dataset loaded + model created + normalized
let isTrained = false; // training complete

let tf = null;

/* -----------------------------
   p5 entry point (NOT async)
------------------------------ */
function setup() {
  noCanvas();
  init().catch((err) => {
    console.error("Fatal init error:", err);
    setTrainStatus(`Init failed: ${err?.message ?? err}`, "status-error");
  });
}

/* -----------------------------
   Main init (async)
------------------------------ */
async function init() {
  console.log("Starting init...");

  // Guard: libs loaded
  if (typeof ml5 === "undefined") {
    throw new Error("ml5 is not loaded. Check your <script> order.");
  }
  if (typeof Papa === "undefined") {
    throw new Error("PapaParse is not loaded. Add papaparse CDN in index.html.");
  }

  // Use the TF instance that ml5 uses (prevents version mismatch)
  tf = ml5.tf;
  if (!tf) {
    throw new Error("ml5.tf not available. ml5 may not have loaded correctly.");
  }

  // Backend init
  await initTensorFlowBackend();

  // UI
  setupButtons();
  setDate();
  setTrainStatus("Loading dataset…", "status-info");

  // Data
  await loadCSVData();

  isModelReady = true;
  setTrainStatus("Dataset loaded. Ready to train.", "status-success");
  console.log("✓ Init complete");
}

/* -----------------------------
   TF backend init
------------------------------ */
async function initTensorFlowBackend() {
  // Prefer WebGL, fallback to CPU
  try {
    await tf.setBackend("webgl");
    await tf.ready();
    console.log("✓ TensorFlow backend:", tf.getBackend());
  } catch (e) {
    console.warn("WebGL backend failed, switching to CPU:", e);
    await tf.setBackend("cpu");
    await tf.ready();
    console.log("✓ TensorFlow backend (CPU):", tf.getBackend());
  }

  // Small stabilization pause
  await sleep(200);
}

/* -----------------------------
   CSV loading
------------------------------ */
function loadCSVData() {
  return new Promise((resolve, reject) => {
    console.log("Loading CSV…");

    Papa.parse("./dataset_btc_fear_greed_copy.csv", {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const rows = Array.isArray(results.data) ? results.data : [];
        console.log(`✓ Loaded ${rows.length} rows from CSV`);

        // Create model
        neuralModel = ml5.neuralNetwork({
          task: "classification",
          debug: false,
          inputs: ["date", "volume", "rate"],
          outputs: ["label"],
        });

        // Add training data
        let count = 0;

        for (const row of rows) {
          if (!row) continue;

          const dateStr = String(row.date ?? "").trim();
          const volume = Number(row.volume);
          const rate = Number(row.rate);
          const label = String(row.prediction ?? "").trim();

          if (!dateStr || !label) continue;
          if (!Number.isFinite(volume) || !Number.isFinite(rate)) continue;

          const dateNum = convertDate(dateStr);

          // Object inputs/outputs = most compatible across ml5 versions
          neuralModel.addData(
            { date: dateNum, volume, rate },
            { label }
          );

          count++;
        }

        console.log(`✓ Added ${count} training examples`);

        // Normalize
        try {
          neuralModel.normalizeData();
          console.log("✓ Normalized data");
        } catch (e) {
          console.warn("normalizeData failed (continuing):", e);
        }

        resolve();
      },
      error: (err) => {
        console.error("CSV parse error:", err);
        reject(err);
      },
    });
  });
}

/* -----------------------------
   UI setup
------------------------------ */
function setupButtons() {
  select("#train")?.mousePressed(startTraining);
  select("#predict")?.mousePressed(makePrediction);
  select("#fetchData")?.mousePressed(fetchLiveData);
}

function setDate() {
  const today = new Date();
  const dateStr = today.toISOString().split("T")[0];
  select("#date")?.value(dateStr);
}

function setTrainStatus(text, className = "status-info") {
  const el = select("#trainStatus");
  if (!el) return;

  el.removeClass("status-info");
  el.removeClass("status-success");
  el.removeClass("status-warning");
  el.removeClass("status-error");

  el.addClass(className);
  el.html(text);
}

function setResult(html) {
  select("#result")?.html(html);
}

/* -----------------------------
   Training
------------------------------ */
function startTraining() {
  if (isTraining) return;

  if (!isModelReady || !neuralModel) {
    setTrainStatus("Model not ready yet — wait for dataset load.", "status-warning");
    return;
  }

  isTraining = true;
  isTrained = false;

  select("#train")?.html("Training…");
  setTrainStatus("Training started… (see console for epoch/loss)", "status-info");

  console.log("Starting training…");

  const opts = {
    epochs: 32,
    batchSize: 32,
  };

  neuralModel.train(
    opts,
    trainingProgress,
    trainingDone
  );
}

function trainingProgress(epoch, loss) {
  const lossVal =
    loss && typeof loss.loss === "number" ? loss.loss.toFixed(4) : "…";
  console.log(`Epoch: ${epoch} - Loss: ${lossVal}`);
}

function trainingDone() {
  console.log("✓ Training complete!");

  isTraining = false;
  isTrained = true;

  select("#train")?.html("Trained");
  select("#train")?.style("background-color", "#31fa03");

  // Show Predict button
  select("#predict")?.style("display", "inline-block");

  setTrainStatus("Training complete. You can predict now.", "status-success");
}

/* -----------------------------
   Prediction
------------------------------ */
function makePrediction() {
  if (!isModelReady || !neuralModel) {
    setTrainStatus("Model not ready.", "status-warning");
    return;
  }
  if (!isTrained) {
    setTrainStatus("Train the model first.", "status-warning");
    return;
  }

  const dateStr = String(select("#date")?.value() ?? "").trim();
  const rateStr = String(select("#rate")?.value() ?? "").trim();
  const volStr = String(select("#volume")?.value() ?? "").trim();

  if (!dateStr || !rateStr || !volStr) {
    setTrainStatus("Please fill Date, Price, and Volume.", "status-warning");
    return;
  }

  const dateVal = convertDate(dateStr);
  const rateVal = Number(rateStr);
  const volVal = Number(volStr);

  if (!Number.isFinite(dateVal) || !Number.isFinite(rateVal) || !Number.isFinite(volVal)) {
    setTrainStatus("Invalid input values.", "status-warning");
    return;
  }

  const input = { date: dateVal, volume: volVal, rate: rateVal };
  console.log("Classifying with:", input);

  // Robust classify handling:
  // - works with callback
  // - works with Promise return
  // - works if ml5 passes results as first arg
  try {
    const maybePromise = neuralModel.classify(input, handleResults);

    if (maybePromise && typeof maybePromise.then === "function") {
      maybePromise
        .then((res) => handleResults(null, res))
        .catch((err) => handleResults(err, null));
    }
  } catch (err) {
    handleResults(err, null);
  }
}

function handleResults(error, results) {
  // Some ml5 versions pass results array as first arg
  if (Array.isArray(error) && results == null) {
    results = error;
    error = null;
  }

  if (error) {
    console.error("Prediction error:", error);
    setTrainStatus("Prediction failed (see console).", "status-error");
    setResult(
      `<div class="prediction-result"><div class="prediction-label">Prediction failed</div></div>`
    );
    return;
  }

  console.log("Results:", results);

  if (!Array.isArray(results) || results.length === 0) {
    setTrainStatus("No prediction results.", "status-warning");
    setResult(
      `<div class="prediction-result"><div class="prediction-label">No results</div></div>`
    );
    return;
  }

  // Pick top confidence
  let top = results[0];
  for (const r of results) {
    if (typeof r?.confidence === "number" && r.confidence > (top?.confidence ?? -1)) {
      top = r;
    }
  }

  const label = String(top?.label ?? "Unknown");
  const conf = typeof top?.confidence === "number" ? (top.confidence * 100).toFixed(1) : "0.0";

  const { advice, emoji, cssClass } = labelToAdvice(label);

  setTrainStatus(`Prediction: ${label} (${conf}%)`, "status-success");

  setResult(`
    <div class="prediction-result ${cssClass}">
      <div class="prediction-header">
        <div class="prediction-emoji">${emoji}</div>
        <div class="prediction-label">${label}</div>
      </div>
      <div class="confidence">Confidence: ${conf}%</div>
      <div class="advice">${advice}</div>
      <div class="disclaimer">Educational only — not financial advice.</div>
    </div>
  `);
}

/* -----------------------------
   Fetch live data (CoinGecko)
------------------------------ */
async function fetchLiveData() {
  const btn = select("#fetchData");
  const lastUpdate = select("#lastUpdate");

  btn?.attribute("disabled", "");
  btn?.html("Fetching…");
  lastUpdate?.html("Fetching CoinGecko…");

  try {
    const url =
      "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true";

    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`CoinGecko HTTP ${res.status}`);

    const data = await res.json();
    const price = data?.bitcoin?.usd;
    const vol = data?.bitcoin?.usd_24h_vol;

    if (typeof price !== "number" || typeof vol !== "number") {
      throw new Error("Unexpected CoinGecko response format");
    }

    select("#rate")?.value(Math.round(price));
    select("#volume")?.value(Math.round(vol));

    const now = new Date();
    lastUpdate?.html(`Last update: ${now.toLocaleString()}`);
    console.log("✓ CoinGecko live data:", { price, vol });
  } catch (err) {
    console.error("Fetch Live Data failed:", err);
    lastUpdate?.html(`Fetch failed: ${err?.message ?? err}`);
  } finally {
    btn?.removeAttribute("disabled");
    btn?.html("Fetch Live Data");
  }
}

/* -----------------------------
   Helpers
------------------------------ */
function convertDate(dateStr) {
  // expects YYYY-MM-DD
  const parts = String(dateStr).split("-");
  if (parts.length !== 3) return 0;

  const y = Number(parts[0]);
  const m = Number(parts[1]);
  const d = Number(parts[2]);

  if (!Number.isFinite(y) || !Number.isFinite(m) || !Number.isFinite(d)) return 0;

  const inputDate = new Date(y, m - 1, d);
  const refDate = new Date(2018, 0, 1);

  const diffDays = Math.floor((inputDate - refDate) / (1000 * 60 * 60 * 24));
  return Number.isFinite(diffDays) ? diffDays : 0;
}

function labelToAdvice(label) {
  switch (label) {
    case "Extreme Fear":
      return { advice: "Buy the dip → STRONG BUY", emoji: "🔥", cssClass: "extreme-fear" };
    case "Fear":
      return { advice: "Good entry point → BUY", emoji: "😰", cssClass: "fear" };
    case "Neutral":
      return { advice: "Market stable → HOLD", emoji: "😐", cssClass: "neutral" };
    case "Greed":
      return { advice: "Consider taking profits", emoji: "😎", cssClass: "greed" };
    case "Extreme Greed":
      return { advice: "Sell high → SELL", emoji: "🤑", cssClass: "extreme-greed" };
    default:
      return { advice: "Unknown sentiment", emoji: "❓", cssClass: "" };
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}