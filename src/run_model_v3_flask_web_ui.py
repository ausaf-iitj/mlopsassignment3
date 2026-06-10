# run_model.py — Version 3: Flask Web Interface
# Purpose: Interactive web UI at http://localhost:8080 for zero-shot classification
# Run: python run_model_v3_flask_web_ui.py
# Docker: docker run --rm -p 0.0.0.0:8080:8080 -v "/mnt/f/models/gliclass-llama-1.3B:/models/gliclass-llama-1.3B:ro" -v ~/.netrc:/root/.netrc:ro gliclass-model
# Then open: http://localhost:8080
# Result: Web server running, browser UI accessible

import torch
import wandb
from flask import Flask, request, jsonify, render_template_string
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

app = Flask(__name__)

# Load model once at startup
tokenizer = AutoTokenizer.from_pretrained('/models/gliclass-llama-1.3B')
model = GLiClassModel.from_pretrained('/models/gliclass-llama-1.3B',
                                      torch_dtype=torch.float32)
model = model.float()

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cpu'
)
print('Model ready!')

HTML = '''
<!DOCTYPE html>
<html>
<head>
  <title>GLiClass Classifier</title>
  <style>
    body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
    textarea, input { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
    button { background: #4CAF50; color: white; padding: 12px 24px; border: none; cursor: pointer; font-size: 16px; }
    .result { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .bar { background: #4CAF50; height: 22px; border-radius: 3px; transition: width 0.3s; }
  </style>
</head>
<body>
  <h1>GLiClass Zero-Shot Classifier</h1>
  <textarea id="text" rows="4" placeholder="Enter text to classify...">Apple announces new iPhone</textarea>
  <input id="labels" placeholder="Labels (comma separated)" value="technology,sports,politics,business,health">
  <input id="threshold" type="number" step="0.05" min="0" max="1" value="0.1" placeholder="Threshold">
  <button onclick="classify()">Classify</button>
  <div id="results"></div>
  <script>
    async function classify() {
      const text = document.getElementById("text").value;
      const labels = document.getElementById("labels").value.split(",").map(l => l.trim());
      const threshold = parseFloat(document.getElementById("threshold").value);
      document.getElementById("results").innerHTML = "<p>Running...</p>";
      const res = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, labels, threshold })
      });
      const data = await res.json();
      let html = "<h3>Results:</h3>";
      data.results.forEach(r => {
        const pct = (r.score * 100).toFixed(1);
        html += `<div class="result">
          <b>${r.label}</b>: ${pct}%
          <div style="background:#ddd;border-radius:3px">
            <div class="bar" style="width:${pct}%"></div>
          </div>
        </div>`;
      });
      document.getElementById("results").innerHTML = html;
    }
  </script>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HTML)


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    text = data['text']
    labels = data['labels']
    threshold = data.get('threshold', 0.1)

    result = pipeline(text, labels, threshold=threshold)

    scores = {}
    for item in result:
        for pred in item:
            scores[pred['label']] = pred['score']

    try:
        wandb.log(scores)
    except Exception:
        pass

    return jsonify({'results': [{'label': k, 'score': v} for k, v in scores.items()]})


if __name__ == '__main__':
    print('Starting web server at http://localhost:8080')
    app.run(host='0.0.0.0', port=8080)
