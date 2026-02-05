const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('resultCard');

// State
let currentFile = null;

// Drag & Drop
dropArea.addEventListener('click', () => fileInput.click());
dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.classList.add('active'); });
dropArea.addEventListener('dragleave', () => dropArea.classList.remove('active'));
dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('active');
    handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

function handleFile(file) {
    if (!file || !file.type.startsWith('audio/')) return alert('Please upload an audio file');
    currentFile = file;
    dropArea.querySelector('h3').innerText = file.name;
    detectBtn.disabled = false;
    resultCard.hidden = true;
}

detectBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    loader.hidden = false;
    detectBtn.disabled = true;
    resultCard.hidden = true;

    try {
        const base64Audio = await toBase64(currentFile);
        const apiKey = document.getElementById('apiKey').value;
        const lang = document.getElementById('languageSelect').value;

        const res = await fetch('http://127.0.0.1:8000/detect-voice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey
            },
            body: JSON.stringify({
                audio_base64: base64Audio.split(',')[1], // Remove type prefix
                language: lang
            })
        });

        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || 'Analysis Failed');

        showResult(data);

    } catch (err) {
        alert(err.message);
    } finally {
        loader.hidden = true;
        detectBtn.disabled = false;
    }
});

function toBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

function showResult(data) {
    resultCard.hidden = false;

    const verdictEl = document.getElementById('verdict');
    const fillEl = document.getElementById('confidenceFill');

    verdictEl.innerText = data.result;
    verdictEl.className = `verdict ${data.result}`; // Add class for color

    const confPercent = (data.confidence * 100).toFixed(1) + '%';
    fillEl.style.width = confPercent;
    fillEl.style.backgroundColor = data.result === "AI_GENERATED" ? "#ef4444" : "#10b981";
    document.getElementById('confidenceText').innerText = `Confidence: ${confPercent}`;

    document.getElementById('expEntropy').innerText = data.explainability.spectral_entropy;
    document.getElementById('expTemp').innerText = data.explainability.temporal_consistency;
    document.getElementById('expPhase').innerText = data.explainability.phase_anomalies;
}
