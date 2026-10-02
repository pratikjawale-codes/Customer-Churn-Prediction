const form = document.querySelector("#customer-form");
const statusText = document.querySelector("#service-status");
const statusDot = document.querySelector("#status-dot");
const predictButton = document.querySelector("#predict-button");
const emptyState = document.querySelector("#empty-state");
const resultState = document.querySelector("#result-state");
const errorBox = document.querySelector("#form-error");

async function updateServiceStatus() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    const ready = response.ok && health.status === "healthy";
    statusText.textContent = ready ? "Model ready" : "Train model to begin";
    statusDot.classList.toggle("ready", ready);
    statusDot.classList.toggle("offline", !response.ok);
  } catch {
    statusText.textContent = "Service unavailable";
    statusDot.classList.add("offline");
  }
}

function updateDependentServices() {
  const internet = form.elements.InternetService.value;
  const dependentFields = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
  ];
  for (const name of dependentFields) {
    const field = form.elements[name];
    if (internet === "No") field.value = "No internet service";
    else if (field.value === "No internet service") field.value = "No";
  }

  const multipleLines = form.elements.MultipleLines;
  if (form.elements.PhoneService.value === "No") multipleLines.value = "No phone service";
  else if (multipleLines.value === "No phone service") multipleLines.value = "No";
}

form.elements.InternetService.addEventListener("change", updateDependentServices);
form.elements.PhoneService.addEventListener("change", updateDependentServices);
updateDependentServices();
updateServiceStatus();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  predictButton.disabled = true;
  predictButton.querySelector("span:first-child").textContent = "Reading profile…";

  const profile = Object.fromEntries(new FormData(form).entries());
  profile.SeniorCitizen = Number(profile.SeniorCitizen);
  profile.tenure = Number(profile.tenure);
  profile.MonthlyCharges = Number(profile.MonthlyCharges);
  profile.TotalCharges = Number(profile.TotalCharges);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Prediction could not be completed.");

    const prediction = payload.predictions[0];
    const percentage = Math.round(prediction.churn_probability * 100);
    const riskClass = prediction.risk_level.split(" ")[0].toLowerCase();
    document.querySelector("#score").textContent = `${percentage}%`;
    document.querySelector("#risk-label").textContent = prediction.risk_level;
    document.querySelector("#risk-label").className = `risk-label ${riskClass}`;
    document.querySelector("#probability-bar").style.width = `${percentage}%`;
    document.querySelector("#probability-bar").dataset.risk = riskClass;
    document.querySelector("#probability-meter").setAttribute("aria-valuenow", percentage);
    document.querySelector("#prediction-copy").textContent =
      `The model predicts ${prediction.churn_prediction.toLowerCase()} churn for this profile.`;
    document.querySelector("#recommendation-text").textContent = prediction.recommended_action;
    document.querySelector("#customer-ref").textContent = prediction.customerID;
    emptyState.hidden = true;
    resultState.hidden = false;
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    predictButton.disabled = false;
    predictButton.querySelector("span:first-child").textContent = "Reveal churn outlook";
  }
});
