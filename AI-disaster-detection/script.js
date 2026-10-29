console.log("AI Disaster Detection System loaded successfully.");


// ======================================================
// ELEMENTS
// ======================================================

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const analyzeButton = document.getElementById("analyzeButton");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const predictionResult = document.getElementById("predictionResult");

const predictionName = document.getElementById("predictionName");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const riskLevel = document.getElementById("riskLevel");
const warningMessage = document.getElementById("warningMessage");
const recommendedAction = document.getElementById("recommendedAction");


// ======================================================
// IMAGE PREVIEW
// ======================================================

if (imageInput) {

    imageInput.addEventListener("change", function () {

        const file = this.files[0];

        if (file) {

            const reader = new FileReader();

            reader.onload = function (event) {

                imagePreview.src = event.target.result;
                previewContainer.style.display = "block";

            };

            reader.readAsDataURL(file);
        }

    });

}


// ======================================================
// SAVE PREDICTION TO HISTORY
// ======================================================

function savePredictionHistory(fileName, prediction, confidence, risk) {

    const history =
        JSON.parse(localStorage.getItem("predictionHistory")) || [];


    const predictionData = {

        image: fileName,

        prediction: prediction,

        confidence: confidence,

        risk: risk,

        status: prediction.toLowerCase() === "normal"
            ? "Normal"
            : "Detected",

        time: new Date().toLocaleString()

    };


    history.unshift(predictionData);


    // Keep latest 20 predictions
    if (history.length > 20) {

        history.pop();

    }


    localStorage.setItem(
        "predictionHistory",
        JSON.stringify(history)
    );

}


// ======================================================
// ANALYZE IMAGE
// ======================================================

if (analyzeButton) {

    analyzeButton.addEventListener("click", async function () {

        if (!imageInput.files.length) {

            alert("Please select a disaster image first.");

            return;

        }


        const file = imageInput.files[0];


        // Show analyzing status

        analyzeButton.disabled = true;

        analyzeButton.innerHTML = "⏳ Analyzing...";


        // Create FormData

        const formData = new FormData();

        formData.append("image", file);


        try {

            // Send image to Flask

            const response = await fetch(
                "http://127.0.0.1:5000/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


            const data = await response.json();


            // Check for Flask error

            if (!response.ok) {

                throw new Error(
                    data.error || "Prediction failed."
                );

            }


            // ==================================================
            // SHOW RESULT
            // ==================================================

            resultPlaceholder.classList.add("d-none");

            predictionResult.classList.remove("d-none");


            // Convert model class name to readable text

            let readablePrediction = data.prediction
                .replace(/_/g, " ")
                .replace(/\b\w/g, function (letter) {
                    return letter.toUpperCase();
                });


            // Prediction

            predictionName.textContent =
                readablePrediction;


            // Confidence

            confidenceValue.textContent =
                data.confidence + "%";


            confidenceBar.style.width =
                data.confidence + "%";


            // Risk

            riskLevel.textContent =
                data.risk;


            // Warning

            warningMessage.textContent =
                data.message;


            // ==================================================
            // RECOMMENDED ACTION
            // ==================================================

            if (data.prediction === "normal") {

                recommendedAction.textContent =
                    "No immediate disaster condition detected. Continue monitoring the situation.";

            }

            else if (data.risk === "HIGH") {

                recommendedAction.textContent =
                    "Take appropriate safety precautions and follow local emergency instructions.";

            }

            else if (data.risk === "MEDIUM") {

                recommendedAction.textContent =
                    "Stay alert, verify the situation, and monitor local emergency information.";

            }

            else {

                recommendedAction.textContent =
                    "Continue monitoring the situation and verify the prediction if necessary.";

            }


            // ==================================================
            // SAVE RESULT TO DASHBOARD HISTORY
            // ==================================================

            savePredictionHistory(
                file.name,
                readablePrediction,
                data.confidence,
                data.risk
            );


            console.log("Prediction:", data);

        }


        catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            alert(
                "Could not connect to the AI model. Make sure Flask is running."
            );

        }


        // Restore button

        analyzeButton.disabled = false;

        analyzeButton.innerHTML =
            "🔍 Analyze Image";

    });

}


// ======================================================
// LOAD DASHBOARD HISTORY
// ======================================================

function loadDashboard() {

    const predictionTable =
        document.getElementById("predictionTable");


    const imagesAnalyzed =
        document.getElementById("imagesAnalyzed");


    const detections =
        document.getElementById("detections");


    const highRisk =
        document.getElementById("highRisk");


    // If these elements don't exist,
    // we are not on the dashboard page.

    if (
        !predictionTable ||
        !imagesAnalyzed ||
        !detections ||
        !highRisk
    ) {

        return;

    }


    // Get history

    const history =
        JSON.parse(localStorage.getItem("predictionHistory")) || [];


    // ==================================================
    // UPDATE STATISTICS
    // ==================================================

    imagesAnalyzed.textContent =
        history.length;


    const detectionCount =
        history.filter(function (item) {

            return item.prediction.toLowerCase() !== "normal";

        }).length;


    detections.textContent =
        detectionCount;


    const highRiskCount =
        history.filter(function (item) {

            return item.risk === "HIGH";

        }).length;


    highRisk.textContent =
        highRiskCount;


    // ==================================================
    // EMPTY HISTORY
    // ==================================================

    if (history.length === 0) {

        predictionTable.innerHTML = `

            <tr>

                <td colspan="5" class="empty-history">

                    No predictions yet.

                    <br>

                    <a href="detection.html">
                        Analyze an image →
                    </a>

                </td>

            </tr>

        `;

        return;

    }


    // ==================================================
    // CREATE TABLE
    // ==================================================

    predictionTable.innerHTML = "";


    history.slice(0, 10).forEach(function (item) {

        let riskClass = "";


        if (item.risk === "HIGH") {

            riskClass = "text-danger fw-bold";

        }

        else if (item.risk === "MEDIUM") {

            riskClass = "text-warning fw-bold";

        }

        else {

            riskClass = "text-success fw-bold";

        }


        let statusClass = "";


        if (item.status === "Detected") {

            statusClass = "text-danger fw-bold";

        }

        else {

            statusClass = "text-success fw-bold";

        }


        const row = document.createElement("tr");


        row.innerHTML = `

            <td>

                ${item.image}

            </td>


            <td>

                ${item.prediction}

            </td>


            <td>

                ${item.confidence}%

            </td>


            <td class="${riskClass}">

                ${item.risk}

            </td>


            <td class="${statusClass}">

                ${item.status}

            </td>

        `;


        predictionTable.appendChild(row);

    });

}


// ======================================================
// RUN DASHBOARD FUNCTION
// ======================================================

loadDashboard();