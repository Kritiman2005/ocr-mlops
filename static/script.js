async function runOCR() {
    const fileInput = document.getElementById("fileInput");
    const output = document.getElementById("output");
    const confidence = document.getElementById("confidence");
    const loader = document.getElementById("loader");
  
    if (!fileInput.files.length) {
      alert("Please select an image");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
  
    loader.style.display = "block";
  
    const response = await fetch("/ocr", {
      method: "POST",
      body: formData
    });
  
    const data = await response.json();
  
    loader.style.display = "none";
    output.textContent = data.prediction;
    confidence.textContent = (data.confidence * 100).toFixed(2) + "%";
  }
  