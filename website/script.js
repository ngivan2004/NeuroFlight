document.addEventListener("DOMContentLoaded", () => {
  document
    .getElementById("predictionForm")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      const resultDiv = document.getElementById("result");
      resultDiv.style.display = "none";
      const formData = {
        startingAirport: document.getElementById("startingAirport").value,
        destinationAirport: document.getElementById("destinationAirport").value,
        segmentsAirlineCode: document.getElementById("segmentsAirlineCode")
          .value,
        travelDurationHours: parseInt(
          document.getElementById("travelDurationHours").value
        ),
        travelDurationMinutes: parseInt(
          document.getElementById("travelDurationMinutes").value
        ),
        totalFare: parseFloat(document.getElementById("totalFare").value),
        flightDate: formatDate(document.getElementById("flightDate").value),
      };

      document.getElementById("loadingScreen").style.display = "flex";

      try {
        const response = await fetch(
          "https://1zkf8hrdfk.execute-api.us-east-1.amazonaws.com/default/airline-api",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(formData),
          }
        );

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        const data = await response.json();
        displayResult(data);
      } catch (error) {
        console.error("Error:", error);
        alert(
          "An error occurred while fetching the prediction. Please try again."
        );
      } finally {
        document.getElementById("loadingScreen").style.display = "none";
      }
    });

  function formatDate(dateString) {
    const date = new Date(dateString);
    return `${date.getDate().toString().padStart(2, "0")}/${(
      date.getMonth() + 1
    )
      .toString()
      .padStart(2, "0")}/${date.getFullYear()}`;
  }

  function displayResult(data) {
    const resultDiv = document.getElementById("result");
    resultDiv.style.display = "block";

    resultDiv.classList.add("fade-in");

    let res = "";
    const totalFare = parseFloat(document.getElementById("totalFare").value);
    const predictedLowestPrice = parseFloat(data.predictedLowestPrice);
    const absPriceDifference = Math.abs(totalFare - predictedLowestPrice);
    const priceDifference = totalFare - predictedLowestPrice;

    if (priceDifference < 0) {
      res =
        "<strong>The current price is even lower than our predicted minimum. Purchase now.</strong>";
    } else if (absPriceDifference < 10) {
      res =
        "<strong>The current price is close to the predicted minimum. Purchase now.</strong>";
    } else if (data.priceTrend === "Already lowest") {
      res =
        "<strong>However, the optimal time to purchase has likely passed already.</strong>";
    } else {
      res =
        "<strong>Prices are expected to drop to the minimum in the future. Consider waiting.</strong>";
    }

    let comparisonText;
    if (priceDifference > 0) {
      comparisonText = "higher than";
    } else if (priceDifference < 0) {
      comparisonText = "lower than";
    } else {
      comparisonText = "equal to";
    }

    document.getElementById(
      "predictedPrice"
    ).innerHTML = `The current price is <strong>$${absPriceDifference.toFixed(
      2
    )} ${comparisonText} </strong> our predicted minimum ($${predictedLowestPrice.toFixed(
      2
    )}).`;

    document.getElementById("priceTrend").innerHTML = res;

    document.getElementById(
      "confidenceText"
    ).innerHTML = `The model is <strong>${parseFloat(data.confidence).toFixed(
      2
    )}%</strong> confident in this prediction.`;

    const confidenceBarFill = document.getElementById("confidenceBarFill");
    confidenceBarFill.style.width = "0%";
    setTimeout(() => {
      confidenceBarFill.style.width = `${data.confidence}%`;
    }, 100);

    resultDiv.querySelectorAll("p").forEach((p, index) => {
      p.classList.add("slide-in");
      p.style.animationDelay = `${index * 0.1}s`;
    });

    const scrollingElement = document.scrollingElement || document.body;
    scrollingElement.scrollTop = scrollingElement.scrollHeight;
  }

  const today = new Date();
  const minDate = new Date(2024, 3, 16); // months are 0-indexed (why??????????)
  const maxDate = new Date(2024, 9, 5);
  const flightDateInput = document.getElementById("flightDate");

  if (today > maxDate) {
    flightDateInput.value = maxDate.toISOString().split("T")[0];
  } else if (today < minDate) {
    flightDateInput.value = minDate.toISOString().split("T")[0];
  } else {
    flightDateInput.value = today.toISOString().split("T")[0];
  }

  const sampleFlight = {
    startingAirport: "JFK",
    destinationAirport: "LAX",
    segmentsAirlineCode: "B6",
    travelDurationHours: 6,
    travelDurationMinutes: 30,
    totalFare: 299.99,
    flightDate: "2024-07-15",
  };

  function fillSampleFlight() {
    document.getElementById("startingAirport").value =
      sampleFlight.startingAirport;
    document.getElementById("destinationAirport").value =
      sampleFlight.destinationAirport;
    document.getElementById("segmentsAirlineCode").value =
      sampleFlight.segmentsAirlineCode;
    document.getElementById("travelDurationHours").value =
      sampleFlight.travelDurationHours;
    document.getElementById("travelDurationMinutes").value =
      sampleFlight.travelDurationMinutes;
    document.getElementById("totalFare").value = sampleFlight.totalFare;
    document.getElementById("flightDate").value = sampleFlight.flightDate;
  }

  document
    .getElementById("sampleFlightButton")
    .addEventListener("click", fillSampleFlight);
});
