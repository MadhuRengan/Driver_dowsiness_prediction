# Driver Drowsiness Prediction 🚗💤

A deep-learning-based **driver drowsiness prediction system** that analyzes facial images and classifies the driver's state using a Convolutional Neural Network (CNN).

The project includes a Streamlit web application, a trained TensorFlow/Keras model, Docker support, and a GitHub Actions CI/CD pipeline designed for deployment on an AWS EC2 instance.

> **Project status:** The CI/CD and containerization foundation is implemented. Before the application can be deployed successfully, the trained model artifact expected by the application must be added to the repository or supplied through a model-artifact mechanism.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Machine Learning Model](#-machine-learning-model)
- [Web Application](#-web-application)
- [Local Setup](#-local-setup)
- [Run the Application](#-run-the-application)
- [Docker Setup](#-docker-setup)
- [CI/CD Pipeline](#-cicd-pipeline)
- [AWS EC2 Deployment](#-aws-ec2-deployment)
- [GitHub Actions Secrets](#-github-actions-secrets)
- [Deployment Workflow](#-deployment-workflow)
- [Health Check](#-health-check)
- [Troubleshooting](#-troubleshooting)
- [Security Considerations](#-security-considerations)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

Driver fatigue can reduce alertness, reaction time, and decision-making ability and can increase the risk of road accidents.

This project explores a **vision-based, non-intrusive approach** to driver-state prediction. Instead of relying on vehicle telemetry or physical sensors, the system uses facial-image information processed by a CNN.

The application provides a simple web interface where a user can upload a driver's image and receive a model prediction.

### High-level flow

\`\`\`text
Driver Image
     │
     ▼
Streamlit Web Application
     │
     ▼
Image Preprocessing
     │
     ▼
CNN / TensorFlow Model
     │
     ▼
Prediction
     │
     ▼
Driver State
\`\`\`

---

## ✨ Features

- 🧠 CNN-based image classification
- 👁️ Vision-based driver-state analysis
- 🖼️ Upload JPG, JPEG, and PNG images
- 🌐 Streamlit web interface
- 🐳 Dockerized application
- 🔄 GitHub Actions CI/CD
- 📦 GitHub Container Registry support
- ☁️ AWS EC2 deployment support
- ❤️ Application health check
- 🔁 Automatic container replacement during deployment

---

## 🏗️ Project Architecture

The project is designed around the following deployment architecture:

\`\`\`text
                         ┌─────────────────────┐
                         │      Developer      │
                         │                     │
                         │     git push main   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     GitHub          │
                         │     Repository      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   GitHub Actions    │
                         │                     │
                         │  1. Validate code  │
                         │  2. Build Docker    │
                         │  3. Push image      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ GitHub Container    │
                         │ Registry (GHCR)     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      AWS EC2        │
                         │                     │
                         │  docker pull        │
                         │  stop old container │
                         │  start new          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit :8501     │
                         │ Driver Drowsiness   │
                         │ Application         │
                         └─────────────────────┘
\`\`\`

---

## 📁 Repository Structure

\`\`\`text
Driver_dowsiness_prediction/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── CNN.ipynb
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── drowsiness_model.h5       # Required by app.py
└── README.md
\`\`\`

### File descriptions

| File | Purpose |
|---|---|
| \`CNN.ipynb\` | Model development/training notebook |
| \`app.py\` | Streamlit inference application |
| \`drowsiness_model.h5\` | Trained TensorFlow/Keras model expected by the application |
| \`requirements.txt\` | Python dependencies |
| \`Dockerfile\` | Container build instructions |
| \`.dockerignore\` | Files excluded from Docker build context |
| \`.github/workflows/ci-cd.yml\` | CI/CD automation |
| \`README.md\` | Project documentation |

---

## 🧠 Machine Learning Model

The model development work is contained in \`CNN.ipynb\`.

The notebook contains the workflow for:

1. Loading and preparing image data.
2. Preparing training and test datasets.
3. Building a CNN.
4. Training the model.
5. Generating predictions.
6. Evaluating predictions using a classification report.

The notebook output indicates that the model was trained for **four classes**.

### Important model/application consideration

The current notebook produces a four-class prediction output, while the current \`app.py\` interprets the first output as a single binary drowsiness probability.

Therefore, the class mapping should be explicitly confirmed before production deployment.

For example, a production implementation should follow this pattern:

\`\`\`python
prediction = model.predict(img_resized)
predicted_class = np.argmax(prediction, axis=1)[0]
confidence = float(np.max(prediction))
\`\`\`

The exact class names and their numerical mapping must match the training dataset used by the notebook.

**Do not assume that class indices 0, 1, 2, and 3 correspond to particular driver states unless that mapping has been verified from the training data.**

---

## 🌐 Web Application

The application is implemented using Streamlit.

The current application:

1. Loads the trained model.
2. Accepts an uploaded image.
3. Converts the image into an OpenCV-compatible format.
4. Converts it to grayscale.
5. Resizes it to 224 × 224.
6. Normalizes pixel values.
7. Passes the processed image to the CNN.
8. Displays the prediction.

### Current expected model input

\`\`\`text
Image
  │
  ├── Grayscale
  │
  ├── Resize → 224 × 224
  │
  └── Normalize → pixel values / 255
\`\`\`

The resulting tensor is expected to have the shape:

\`\`\`text
(1, 224, 224, 1)
\`\`\`

---

# 💻 Local Setup

## 1. Clone the repository

\`\`\`bash
git clone https://github.com/MadhuRengan/Driver_dowsiness_prediction.git
cd Driver_dowsiness_prediction
\`\`\`

## 2. Create a virtual environment

### Windows

\`\`\`bash
python -m venv .venv
.venv\\Scripts\\activate
\`\`\`

### Linux/macOS

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

## 3. Install dependencies

\`\`\`bash
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## 4. Add the trained model

The application currently expects:

\`\`\`text
drowsiness_model.h5
\`\`\`

in the project root.

The file must be present before starting the application.

---

# ▶️ Run the Application

Start Streamlit:

\`\`\`bash
streamlit run app.py
\`\`\`

The application normally becomes available at:

\`\`\`text
http://localhost:8501
\`\`\`

Open the address in a browser and upload a supported image.

---

# 🐳 Docker Setup

Docker allows the application to run with the same environment locally and on AWS EC2.

## Build the image

\`\`\`bash
docker build -t driver-drowsiness .
\`\`\`

## Run the container

\`\`\`bash
docker run -d \
  --name driver-drowsiness \
  -p 8501:8501 \
  driver-drowsiness
\`\`\`

Open:

\`\`\`text
http://localhost:8501
\`\`\`

## View container logs

\`\`\`bash
docker logs driver-drowsiness
\`\`\`

## Stop the container

\`\`\`bash
docker stop driver-drowsiness
\`\`\`

## Remove the container

\`\`\`bash
docker rm driver-drowsiness
\`\`\`

---

# 🔄 CI/CD Pipeline

The repository contains:

\`\`\`text
.github/workflows/ci-cd.yml
\`\`\`

The workflow is triggered by:

- Pull requests targeting \`main\`
- Pushes to \`main\`

## CI stage

The CI stage:

1. Checks out the source code.
2. Installs Python.
3. Validates the Python application using \`py_compile\`.
4. Verifies that the trained model artifact exists.
5. Builds the Docker image.

## Publish stage

For a successful push to \`main\`, the workflow:

1. Authenticates with GitHub Container Registry.
2. Builds the Docker image.
3. Pushes the image to GHCR.
4. Creates both \`latest\` and commit-SHA tags.

Example:

\`\`\`text
ghcr.io/madhurengan/driver_dowsiness_prediction:latest
ghcr.io/madhurengan/driver_dowsiness_prediction:<commit-sha>
\`\`\`

## Deployment stage

After the image is published, GitHub Actions connects to the EC2 instance over SSH and:

1. Logs into GHCR.
2. Pulls the latest image.
3. Stops the existing container.
4. Removes the old container.
5. Starts the new container.
6. Waits for the application to start.
7. Performs a Streamlit health check.
8. Removes unused Docker images.

---

# ☁️ AWS EC2 Deployment

## EC2 prerequisites

Create an EC2 instance with a Linux distribution supported by your deployment setup.

Install:

- Docker
- curl

Example for an Amazon Linux-based system:

\`\`\`bash
sudo dnf update -y
sudo dnf install -y docker curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
\`\`\`

Log out and log back in after adding the user to the Docker group.

Verify:

\`\`\`bash
docker --version
\`\`\`\`

---

## 🔐 EC2 Security Group

The EC2 Security Group must allow inbound TCP traffic to port:

\`\`\`text
8501
\`\`\`

For a production environment, it is preferable to place the application behind a reverse proxy such as Nginx and expose HTTPS on port 443 instead of directly exposing Streamlit on port 8501.

---

# 🔑 GitHub Actions Secrets

The CI/CD workflow expects the following repository secrets.

| Secret | Description |
|---|---|
| \`EC2_HOST\` | Public IPv4 address or DNS name of EC2 |
| \`EC2_USERNAME\` | SSH username, e.g. \`ec2-user\` or \`ubuntu\` |
| \`EC2_SSH_KEY\` | Private SSH key used to access EC2 |
| \`GHCR_USERNAME\` | GitHub username used by EC2 for GHCR |
| \`GHCR_TOKEN\` | GitHub token with permission to pull the private container image |

If the GHCR package is public, the EC2 GHCR authentication requirement can potentially be simplified.

### Never commit these values to the repository.

Do not put:

- SSH private keys
- GitHub tokens
- AWS access keys
- passwords

inside source code, Dockerfiles, notebooks, or GitHub workflow files.

---

# 🚀 Deployment Workflow

Once EC2 and GitHub Actions secrets are configured:

\`\`\`bash
git add .
git commit -m "update application"
git push origin main
\`\`\`

GitHub Actions automatically performs:

\`\`\`text
Push
 ↓
CI validation
 ↓
Docker build
 ↓
GHCR push
 ↓
SSH → EC2
 ↓
Docker pull
 ↓
Replace running container
 ↓
Health check
 ↓
Deployment complete
\`\`\`

This means application updates can be deployed without manually copying source code to the EC2 server.

---

# ❤️ Health Check

The Docker container exposes Streamlit's health endpoint:

\`\`\`text
/_stcore/health
\`\`\`

The CI/CD workflow verifies:

\`\`\`bash
curl --fail http://127.0.0.1:8501/_stcore/health
\`\`\`

A successful response indicates that the Streamlit server is responding.

---

# 🛠️ Troubleshooting

## Container exits immediately

Check:

\`\`\`bash
docker logs driver-drowsiness
\`\`\`

A common cause is a missing model file.

Verify:

\`\`\`bash
docker exec -it driver-drowsiness ls -lh /app
\`\`\`

---

## Model file not found

If you see:

\`\`\`text
FileNotFoundError: drowsiness_model.h5
\`\`\`

make sure the trained model is available at the expected location.

The CI/CD workflow intentionally fails when the model artifact is missing.

---

## Streamlit cannot be accessed

Check whether the container is running:

\`\`\`bash
docker ps
\`\`\`

Check the port mapping:

\`\`\`text
0.0.0.0:8501->8501/tcp
\`\`\`

Then verify the EC2 Security Group allows inbound TCP 8501.

---

## GitHub Actions deployment fails

Check the following:

1. \`EC2_HOST\` is correct.
2. \`EC2_USERNAME\` is correct.
3. \`EC2_SSH_KEY\` contains the complete private key.
4. Docker is running on EC2.
5. EC2 can access the internet.
6. GHCR credentials are valid.
7. The EC2 Security Group allows SSH.
8. Port 8501 is allowed for application access.

---

# 🔒 Security Considerations

For a production deployment, consider the following improvements:

### 1. HTTPS

Use a domain name and TLS certificate instead of exposing the Streamlit server directly.

### 2. Reverse proxy

Use Nginx or an AWS load balancer in front of Streamlit.

### 3. Restricted SSH access

Limit SSH access to trusted IP addresses where possible.

### 4. Secret management

Use GitHub Actions secrets, AWS Secrets Manager, or another dedicated secret-management system.

### 5. Container hardening

Consider:

- Running as a non-root user.
- Pinning all dependencies.
- Scanning Docker images for vulnerabilities.
- Keeping the base image updated.

### 6. Model artifact management

Large ML model files should generally be managed through dedicated artifact storage rather than repeatedly storing large binaries in Git history.

Possible approaches include:

- Amazon S3
- Git LFS
- Hugging Face Hub
- GitHub Releases
- Container image layers

---

# 🔮 Future Improvements

The project can be extended with:

- 🎥 Real-time webcam/video inference
- 👁️ Eye Aspect Ratio (EAR) analysis
- 😮 Yawn detection
- ⏱️ Continuous drowsiness monitoring
- 🔔 Audible driver alerts
- 📊 Prediction confidence dashboard
- 📈 Monitoring and logging
- 🧪 Automated unit/integration tests
- 🔐 HTTPS
- 🌐 Custom domain
- ⚖️ Better class balancing and validation
- 📦 Dedicated model registry
- ☁️ S3-based model storage
- 🔁 Blue/green or rolling deployments
- 📡 CloudWatch monitoring
- 🛡️ Container vulnerability scanning

---

# 📊 Model Evaluation

The training notebook includes evaluation using a classification report.

When reporting model performance, distinguish between:

- Training accuracy
- Validation accuracy
- Test accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Very high training/test metrics should be interpreted carefully and validated against an independent dataset before making claims about real-world performance.

---

# ⚠️ Disclaimer

This project is intended for **educational, research, and demonstration purposes**.

A prediction from a computer-vision model should not be treated as a definitive measurement of a driver's fitness to drive or as a replacement for responsible driving practices, safety systems, or validated automotive safety technology.

Real-world deployment should include rigorous testing across different lighting conditions, camera positions, driver characteristics, image quality, and environmental conditions.

---

## 👨‍💻 Author

**Madhu Rengan**

GitHub: [MadhuRengan](https://github.com/MadhuRengan)

Project: [Driver Drowsiness Prediction](https://github.com/MadhuRengan/Driver_dowsiness_prediction)

---

## ⭐ Project Goal

The goal of this project is to demonstrate how a machine-learning model can be taken from **model development → application → Docker → CI/CD → cloud deployment** as an end-to-end machine-learning deployment workflow.

\`\`\`text
Machine Learning
      ↓
CNN Model
      ↓
Streamlit Application
      ↓
Docker
      ↓
GitHub Actions
      ↓
GitHub Container Registry
      ↓
AWS EC2
      ↓
Deployed ML Application
\`\`\`
