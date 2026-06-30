# IMDB Sentiment Classification — End-to-End Deep Learning Project

A complete NLP sentiment classification project (positive/negative movie reviews)
covering training, experiment tracking, API serving, containerization, CI/CD,
and Kubernetes deployment — built to run on **Windows** using **Google Colab**
(for GPU training) and **VS Code** (for everything else).

## 1. Project Overview

- **Problem statement:** Classify IMDB movie reviews as positive or negative.
- **Real-world significance:** Sentiment analysis powers review moderation, customer
  feedback analysis, brand monitoring, and recommendation systems.
- **Dataset:** [IMDB Movie Reviews](https://huggingface.co/datasets/imdb) — 50,000
  labeled reviews (25k train / 25k test), loaded automatically via the
  Hugging Face `datasets` library (no manual download needed).
- **Expected outcome:** A deployed REST API that returns a sentiment label and
  confidence score for arbitrary review text.

| | Model V1 (Baseline) | Model V2 (Improved) |
|---|---|---|
| Architecture | Bidirectional LSTM, trained from scratch | Fine-tuned DistilBERT (transfer learning) |
| Tokenization | Whitespace + custom vocab | Subword (WordPiece) |
| Expected accuracy | ~80-85% | ~90-93% |

---

## 2. Project Structure

```
sentiment-project/
├── README.md
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci-cd.yml
├── src/
│   ├── data_loader.py        # dataset loading + cleaning
│   ├── model_v1.py           # LSTM architecture
│   └── model_v2.py           # DistilBERT wrapper
├── training/
│   ├── train_v1.py           # trains + logs Model V1 to MLflow
│   ├── train_v2.py           # trains + logs Model V2 to MLflow
│   └── compare_models.py     # generates comparison report
├── deployment/
│   └── app.py                # FastAPI app (/, /health, /predict)
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   └── colab_training.ipynb  # run this in Google Colab for GPU training
└── tests/
    └── test_api.py
```

---

## 3. How This Maps to Windows + Colab + VS Code

Since the original guidelines assume a Linux terminal, here's the Windows-friendly
equivalent for each part:

| Task | Where to run it |
|---|---|
| Model training (V1 + V2) | **Google Colab** (free GPU, no local setup needed) |
| Writing/editing code | **VS Code** on Windows |
| MLflow tracking UI | VS Code terminal (PowerShell) or Colab + ngrok |
| FastAPI serving | VS Code terminal (PowerShell), then Docker |
| Docker build/run | **Docker Desktop for Windows** |
| Kubernetes | **Minikube** on Windows (via Docker Desktop driver) |
| Git/GitHub | VS Code's built-in Git, or Git Bash/PowerShell |
| CI/CD | GitHub Actions (runs in the cloud — works the same regardless of your OS) |

---

## 4. Step-by-Step Setup

### A. Local environment (VS Code, PowerShell)

```powershell
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### B. Train the models on Google Colab (recommended — free GPU)

1. Push this project to a GitHub repo first (see Section F).
2. Open `notebooks/colab_training.ipynb` in Google Colab
   (`File > Upload notebook`, or open directly from GitHub).
3. Set **Runtime > Change runtime type > GPU (T4)**.
4. Update the `git clone` URL in the first cell to your repo.
5. Run all cells. This trains Model V1, then Model V2, logs everything to
   MLflow, and generates `artifacts/model_comparison_report.csv`.
6. Run the final cell to **download `trained_artifacts.zip`** to your Windows machine.
7. Extract it into your local project folder so you have:
   ```
   sentiment-project/artifacts/model_v2/...
   sentiment-project/mlruns/...
   ```

> Alternative: if you have a decent CPU and patience, you can instead run
> `python training/train_v1.py` and `python training/train_v2.py` directly
> in VS Code's PowerShell terminal — just expect V2 (DistilBERT) to be slow on CPU.

### C. View MLflow experiment tracking (locally, in VS Code terminal)

```powershell
mlflow ui
```
Open `http://127.0.0.1:5000` in your browser. Take screenshots of:
- The experiment list / run comparison table
- An individual run's parameters and metrics
- The registered models page (Models tab)

### D. Run the FastAPI server locally

```powershell
uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger API docs.
Test `/predict` with a JSON body: `{"text": "This movie was absolutely fantastic!"}`

### E. Docker (Docker Desktop for Windows)

Install **Docker Desktop for Windows** (enable WSL2 backend or Hyper-V when prompted
by the installer — this is just Docker's internal engine, you won't need to use
a Linux terminal yourself).

```powershell
docker build -t final-project:v1 .
docker run -p 8000:8000 final-project:v1
```
Visit `http://localhost:8000/health` to confirm it's running.

### F. Git & GitHub (VS Code built-in or PowerShell)

```powershell
git init
git add .
git commit -m "Initial commit: full sentiment classification project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```
VS Code's Source Control panel (Ctrl+Shift+G) can do all of this through the UI too.

### G. CI/CD with GitHub Actions

The workflow at `.github/workflows/ci-cd.yml` runs automatically on every push to
`main`. It installs dependencies, lints, runs tests, and builds the Docker image —
**entirely on GitHub's cloud runners**, so nothing extra is needed on Windows.

To enable the optional Docker Hub push job, add these repo secrets
(Settings > Secrets and variables > Actions):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Take a screenshot of the green checkmark / pipeline run under the **Actions** tab.

### H. Kubernetes with Minikube on Windows

Install **Minikube** ([minikube.exe](https://minikube.sigs.k8s.io/docs/start/)) and
point it at Docker Desktop as its driver:

```powershell
minikube start --driver=docker

# Build the image directly inside Minikube's Docker environment
minikube image build -t final-project:v1 .

kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

kubectl get pods
kubectl get deployments
kubectl get services

# Open the service in your browser
minikube service sentiment-api-service
```

> Note: `kubernetes/deployment.yaml` uses `imagePullPolicy: Never` so it uses the
> image you just built locally inside Minikube instead of trying to pull from a registry.

---

## 5. Deliverables Checklist

- [ ] Training code (`training/train_v1.py`, `training/train_v2.py`)
- [ ] Comparison report (`artifacts/model_comparison_report.csv`)
- [ ] MLflow screenshots (experiment runs, run comparison, registered model)
- [ ] Working FastAPI app + Swagger docs screenshot
- [ ] Dockerfile + running container screenshot
- [ ] GitHub repo with full structure + commit history
- [ ] GitHub Actions pipeline execution screenshot
- [ ] Minikube: running pods, deployments, and services screenshots
- [ ] 5–10 minute demonstration video covering all of the above

---

## 6. Troubleshooting (Windows-specific)

- **`venv\Scripts\activate` fails with "running scripts is disabled"**: run
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell first.
- **Docker Desktop won't start**: ensure WSL2 is installed (`wsl --install` in an
  admin PowerShell) and virtualization is enabled in your BIOS.
- **Minikube stuck starting**: try `minikube delete` then `minikube start --driver=docker` again.
- **Out of memory training V2 on CPU**: reduce `sample_size` and `batch_size` in
  `training/train_v2.py`, or just train it on Colab instead (recommended).
