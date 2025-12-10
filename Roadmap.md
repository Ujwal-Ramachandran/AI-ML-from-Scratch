# 🚀 THE AI/ML MASTERY ROADMAP
---

## 📘 MODULE 1: FOUNDATIONS (Math → Classical ML → NN Core)

### ✅ L1: Linear Algebra with NumPy
**Concepts:**
* Vectors, matrices, tensor shapes ($N, D$).
* Matrix multiplication, dot products, broadcasting rules.
* Vectorization vs Python loops (Big O implications).
* **Build:**
* **Implement Dense Layer (Forward Pass):**
    * Input X, Weights W, Bias b.
    * Compute $Y = XW + b$ using pure NumPy broadcasting.
    * **ReLU Activation:** Implement $f(x) = max(0, x)$.
* **Benchmark:** Compare runtime of Naive `for-loop` vs `np.dot` for batch sizes $1 \to 10,000$.

### ✅ L2: Classical ML – Regression & Classification
**Concepts:**
* Linear/Logistic Regression theory (Gradient Descent derivation).
* Bias–Variance tradeoff, Overfitting vs Underfitting.
* Evaluation: Precision, Recall, F1, ROC-AUC.
* **Build:**
* **Titanic Survival Prediction (Scikit-Learn):**
    * Preprocessing: Handle missing values, One-Hot Encoding.
    * Train Logistic Regression vs Decision Tree.
    * Plot Learning Curves to visualize Bias vs Variance.

### ✅ L3: Ensemble Methods
**Concepts:**
* **Bagging:** Random Forest (Variance Reduction).
* **Boosting:** XGBoost/LightGBM (Bias Reduction).
* Feature Importance interpretation.
* **Build:**
* **House Price Prediction:**
    * Baseline: Linear Regression.
    * Challenger: XGBoost Regressor.
    * Analyze: Which features matter most? (SHAP values or Gain).

### ✅ L4: Neural Networks from Scratch
**Concepts:**
* Multi-Layer Perceptron (MLP) architecture.
* **The Chain Rule** & Backpropagation derivation.
* Loss Functions: MSE vs Cross-Entropy.
* **Build:**
* **2-Layer NN in NumPy:**
    * Forward: Linear $\to$ ReLU $\to$ Linear $\to$ Sigmoid.
    * Backward: Calculate gradients manually.
    * Optimizer: Update weights ($W = W - \alpha \cdot dW$).
    * Train on "Moons" dataset (non-linear boundary).

### ✅ L5: PyTorch Fundamentals
**Concepts:**
* Tensors, Autograd (Computation Graphs).
* `nn.Module`, `nn.Sequential`, `DataLoader`.
* GPU acceleration (CUDA basics).
* **Build:**
* **MNIST Digit Classifier:**
    * Create custom `Dataset` class.
    * Training Loop: Forward $\to$ Loss $\to$ Backward $\to$ Step.
    * **Hardware Check:** Verify `torch.cuda.is_available()` on RTX 4060.

### ✅ L6: Optimization Techniques
**Concepts:**
* Stochastic Gradient Descent (SGD) vs Adam.
* Momentum, RMSProp.
* Learning Rate Schedulers (Cosine Annealing).
* **Build:**
* **Convergence Battle:**
    * Train MNIST model with SGD, SGD+Momentum, and Adam.
    * Plot Loss vs Epochs for all three on one graph.

### ✅ L7: Regularization
**Concepts:**
* L1/L2 Weight Decay.
* Dropout (Training vs Inference mode).
* Batch Normalization (Covariate Shift).
* **Build:**
* **CIFAR-10 Classifier (preventing overfitting):**
    * Baseline: Overfit a deep MLP.
    * Fix: Add `nn.Dropout` and `nn.BatchNorm2d`.
    * Compare Validation Accuracy.

### ✅ L8: MLflow Setup & Tracking
**Concepts:**
* Experiment Tracking (Parameters, Metrics, Artifacts).
* Model Versioning.
* **Build:**
* **Instrument CIFAR-10 Script:**
    * Log Hyperparams (LR, Batch Size).
    * Log Metrics (Loss, Accuracy per epoch).
    * Save best model artifact (`.pth`) automatically.

---

## 👁️ MODULE 2: COMPUTER VISION (Representation → Detection)

### ✅ L9: CNN Fundamentals
**Concepts:**
* Convolution (Kernels, Stride, Padding).
* Pooling (Max vs Average).
* Receptive Fields.
* **Build:**
* **CNN from Scratch:**
    * Implement `Conv2d` forward pass logic (loops) to understand sliding windows.
    * Train a standard CNN on CIFAR-10.
    * **Vis:** Visualize Feature Maps (what filters actually see).

### ✅ L10: Modern CNN Architectures
**Concepts:**
* ResNet (Skip Connections / Residual Blocks).
* Transfer Learning (Feature Extraction vs Fine-tuning).
* **Build:**
* **Malware Classifier:**
    * Dataset: Malware screenshots vs benign apps.
    * Model: Pre-trained ResNet-18.
    * Action: Freeze backbone, train head, then unfreeze layer 4.

### ✅ L11: Object Detection (YOLO)
**Concepts:**
* Bounding Boxes ($x, y, w, h$).
* IoU (Intersection over Union) & NMS (Non-Max Suppression).
* mAP (Mean Average Precision).
* **Build:**
* **Custom Object Detector (YOLOv8):**
    * Annotate a small dataset (e.g., "Hard Hats").
    * Train YOLOv8n (Nano) on GPU.
    * Run inference on webcam video.

### ✅ L12: Image Segmentation & Advanced CV
**Concepts:**
* Semantic Segmentation (U-Net).
* Encoder-Decoder Architectures.
* **Build:**
* **U-Net from Scratch:**
    * Implement U-Net architecture (skip connections between encoder/decoder).
    * Train on Medical Nuclei Segmentation dataset.

---

## 🗣️ MODULE 3: NLP & LLM CORE (Sequence → Transformers)

### ✅ L13: Text Processing & Embeddings
**Concepts:**
* Tokenization (Subword, BPE).
* Word2Vec / GloVe intuition.
* Transformer Embeddings (Sentence-BERT).
* **Build:**
* **Semantic Classifier:**
    * Embed text using `sentence-transformers`.
    * Train a Logistic Regression head on top.
    * Compare performance vs TF-IDF.

### ✅ L14: Sequence Models (RNN/LSTM)
**Concepts:**
* Recurrent Neural Networks (RNN).
* Vanishing Gradient Problem.
* LSTM/GRU Gates (Forget, Input, Output).
* **Build:**
* **Character-Level Text Gen:**
    * Train an LSTM to generate text style (e.g., Shakespeare).
    * Observe how it fails (repetitive loops) vs succeeds.

### ✅ L15: Transformer Architecture
**Concepts:**
* **Attention is All You Need.**
* Self-Attention ($Q, K, V$).
* Multi-Head Attention & Positional Encoding.
* **Build:**
* **Mini-Transformer:**
    * Implement `SelfAttention` class in PyTorch.
    * Train a small sequence-to-sequence model (e.g., reverse a string).

### ✅ L16: BERT & Fine-Tuning
**Concepts:**
* Encoder-only models (BERT).
* Masked Language Modeling (MLM).
* **Build:**
* **Sentiment Analysis (DistilBERT):**
    * Load pre-trained DistilBERT.
    * Fine-tune on IMDb dataset using Hugging Face `Trainer`.
    * Evaluate F1 Score.

---

## 🗄️ MODULE 4: VECTOR DATABASES & RAG (Memory → Retrieval)

### ✅ L17: Vector Search Fundamentals
**Concepts:**
* Vector Spaces.
* Cosine Similarity vs Euclidean Distance.
* ANN Algorithms (HNSW, IVF).
* **Build:**
* **FAISS Implementation:**
    * Create random vectors.
    * Implement "Brute Force" search (exact).
    * Implement "IVF" search (approximate) and compare speed.

### ✅ L18: Vector Databases
**Concepts:**
* ChromaDB / Weaviate architecture.
* Metadata Filtering.
* **Build:**
* **Knowledge Base Ingestion:**
    * Parse PDFs.
    * Chunk text (RecursiveCharacterTextSplitter).
    * Ingest into ChromaDB (Local).

### ✅ L19: RAG Implementation
**Concepts:**
* Retrieval Augmented Generation (RAG).
* Context Injection.
* **Build:**
* **"Chat with Docs":**
    * Retrieve Top-3 chunks for a query.
    * Construct Prompt: "Context: {chunks}, Question: {q}".
    * Generate Answer using Llama 3 (Ollama).

### ✅ L20: RAG Optimization
**Concepts:**
* Hybrid Search (Keyword + Vector).
* Re-ranking (Cross-Encoders).
* **Build:**
* **Advanced RAG Pipeline:**
    * Add a Re-ranker step (using `bi-encoder` vs `cross-encoder`).
    * Measure "Retrieval Recall" (Did we get the right chunk?).

---

## 🤖 MODULE 5: AGENTIC AI (Reasoning → Workflows)

### ✅ L21: Agent Fundamentals
**Concepts:**
* ReAct Pattern (Reason + Act).
* Tool Usage (Function Calling).
* **Build:**
* **Scratchpad Agent:**
    * Build a loop: Prompt LLM $\to$ Parse "Action:" $\to$ Run Python Tool $\to$ Feed Output back.

### ✅ L22: Agent Workflows
**Concepts:**
* LangGraph / State Machines.
* Human-in-the-loop.
* **Build:**
* **Support Bot Router:**
    * State 1: Classify Intent.
    * State 2: Retrieve Policy (RAG).
    * State 3: Draft Email.

### ✅ L23: Multi-Agent Systems
**Concepts:**
* Orchestration (Manager vs Worker).
* CrewAI Framework.
* **Build:**
* **Research Crew:**
    * Agent A: Search Web (DeepSeek-R1).
    * Agent B: Summarize Findings.
    * Agent C: Critic/Editor.

### ✅ L24: Production Agents
**Concepts:**
* Reliability (Retries, Fallbacks).
* Cost Tracking.
* **Build:**
* **Agent API:**
    * Wrap agent in FastAPI.
    * Implement Timeout/Retry logic.

---

## ⚙️ MODULE 6: PRODUCTION ML (Serving → CI/CD)

### ✅ L25: FastAPI & Serving
**Concepts:**
* REST API Design.
* Pydantic Data Validation.
* **Build:**
* **Prediction Endpoint:**
    * `/predict` route taking JSON.
    * Validate input schema.
    * Return prediction + confidence.

### ✅ L26: Containerization (Docker)
**Concepts:**
* Dockerfiles (Layers, Entrypoints).
* Docker Compose.
* **Build:**
* **Full Stack Container:**
    * Service 1: FastAPI Model.
    * Service 2: ChromaDB.
    * `docker-compose up` to launch both.

### ✅ L27: CI/CD for ML
**Concepts:**
* GitHub Actions.
* Linting & Testing ML code.
* **Build:**
* **Automated Pipeline:**
    * On Push: Run `pytest` (Unit tests for data processing).
    * On Merge: Build Docker Image.

---

## 🧬 MODULE 7: DATA ENGINEERING (Data → Quality)

### ✅ L28: Data Pipelines & DVC
**Concepts:**
* Data Version Control (DVC).
* Reproducibility.
* **Build:**
* **Versioning:**
    * Initialize DVC in a repo.
    * Track a large `.csv` file.
    * Modify data, commit new version, switch back/forth.

### ✅ L29: Data Quality & Validation
**Concepts:**
* Data Drift.
* Schema Validation (Great Expectations / Pandera).
* **Build:**
* **"Break-Fix" Pipeline:**
    * Corrupt the Titanic dataset (add nulls, outliers).
    * Write a validation script that fails.
    * Fix data until validation passes.

---

## 🧠 CAPSTONE 1: INTELLIGENT RESEARCH ASSISTANT
*(Positioned here to utilize RAG, Agents, and Data Engineering skills)*

### ✅ L30: Backend & Orchestration
* **Stack:** FastAPI + LangGraph + DeepSeek-R1 (Ollama).
* * **Build:** Create the "Brain". It receives a query, decides to Search Web or Query Vector DB.

### ✅ L31: Frontend & Integration
* **Stack:** Streamlit or Chainlit.
* * **Build:** A clean UI that renders markdown, citations, and "Thought Process" (ReAct steps).

### ✅ L32: Monitoring & Polish
* **Stack:** LangSmith or basic logging.
* * **Build:** Log every token usage and latency. Optimize prompt for speed.

---

## 🧪 MODULE 8: MODEL EVALUATION & ERROR ANALYSIS

### ✅ L33: Systematic Error Analysis
**Concepts:**
* Confusion Matrix Deep Dive.
* Slice-based Analysis (Performance by category).
* **Build:**
* **Failure Analysis:**
    * Take your best model. Find the top 10 worst errors.
    * Categorize them (e.g., "Blurry Image", "Sarcasm").

---

## 🛡️ MODULE 9: ML SECURITY (Offensive)

### ✅ L34: Adversarial ML
**Concepts:**
* Prompt Injection.
* Adversarial Examples (Pixel noise).
* **Build:**
* **Red Teaming:**
    * Try to "jailbreak" your Research Assistant to generate banned content.
    * Document the prompts that worked.

---

## ⚡ MODULE 10: EFFICIENT TRAINING (RTX 4060 Optimization)

### ✅ L35: Hardware Stress Test
**Concepts:**
* VRAM Management.
* Quantization (4-bit vs 8-bit).
* **Build:**
* **Load DeepSeek-R1 (Distill 8B):**
    * Use `Ollama` or `transformers`.
    * Measure VRAM usage. Ensure ~2GB free for gradients.

### ✅ L36: Fine-Tuning Setup (Unsloth)
**Concepts:**
* LoRA (Low-Rank Adaptation).
* Unsloth (Optimization library).
* **Build:**
* **Fine-Tune Mistral/DeepSeek:**
    * Dataset: "Alpaca-Cleaned" (Small subset).
    * Config: 4-bit loading, LoRA rank 16.
    * Run training for 50 steps.

### ✅ L37: Export & GGUF
**Concepts:**
* Model Merging (Adapter + Base).
* GGUF Format (CPU execution).
* **Build:**
* **Run on CPU:**
    * Export fine-tuned model to GGUF.
    * Run it on your laptop CPU using `llama.cpp`.

---

## 🔒 MODULE 11: ADVANCED LLM SECURITY & FINE-TUNING

### ✅ L38: LLM Guardrails
**Concepts:**
* Input/Output Filtering.
* NeMo Guardrails / LlamaGuard.
* **Build:**
* **Secured RAG:**
    * Add a "Guard" layer before the LLM.
    * Filter out PII (Personally Identifiable Information).

### ✅ L39: Advanced Fine-Tuning
**Concepts:**
* Instruction Tuning vs Continued Pre-training.
* Chat Templates.
* **Build:**
* **Security Specialist Model:**
    * Fine-tune DeepSeek/Mistral on a **Cybersecurity Q&A Dataset**.
    * Evaluate: Does it know CVEs better than the base model?

---

## 🎯 MODULE 12: FINAL CAPSTONE – "CYBER-SENTINEL"

### ✅ L40: The Fuzzy Logic Core
**Concepts:**
* Fuzzy Sets (Low, Med, High).
* Rule-based Aggregation.
* * **Build:**
* **Risk Engine:**
    * Input: Vision Confidence (0.8), NLP Threat Score (0.6).
    * Rule: `IF Vision IS High AND NLP IS Med THEN Risk IS High`.
    * Output: Final Risk Score.

### ✅ L41: The Modules (Vision + NLP + Agent)
* **Build:**
* **Vision:** Phishing Screenshot Detector (CNN).
* **NLP:** Log Anomaly Detector (BERT).
* **Agent:** Threat Intel Lookup (RAG).

### ✅ L42: Full Integration
* **Build:**
* **The Platform:**
    * Docker Compose file launching all 3 services + Risk Engine + Dashboard.
    * Final "Red Team" test against the full system.

---

## 📋 MODULE 13: JOB PREP

### ✅ L43: Portfolio & Blog
* **Deliverable:** Write "How I Built Cyber-Sentinel" on Medium.
* **Repo:** Clean up GitHub, add READMEs.

### ✅ L44: Mock Interviews
* **Focus:** System Design ("Design a Phishing Detector").
* **Focus:** Theory ("Explain Backpropagation").

---

**Foundations → Models → Systems → Production → Security → Final System**