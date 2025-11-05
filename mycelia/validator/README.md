# Validator

Validator is a two-process setup for continuously evaluating miners and serving updated models on a decentralized network.

## Overview

A validator runs two cooperating components:

1. **Constant Evaluation**: discovers miners ready for evaluation, continuously evaluates their submissions, aggregates results, and publishes scores to the chain (and shares with peer validators).
2. **Model Serving**: serves updated models to two audiences:

   * **Peers/Clients** who may need the **full model**
   * **Miners** who may need a **partial model** (e.g. 2 out of the 8 experts & shared weights)

Both read the same `config.json`.

---

## Prerequisites

* Python **3.10+**
* PyTorch (CUDA optional)
* Project dependencies installed in a virtual environment:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## Paths & Layout
All the miner/validator configuration are controlled via a config.json file. 

You may create a template config file by running 
```python mycelia/shared/config.py --get_template <validator/miner> --coldkey_name <your coldkey name> --hotkey_name <your hotkey name> --run_name <your naming to specify this run>```

Then, you may modify any specifics in the config.json file if needed.

Afterwards, when you are running the validator, simply use `--path` to point to the validator’s **config file**:

```
~/subnet-MoE/checkpoints/validator/<your hotkey>/<run name>/config.json
```
> when a path is not provided, it would use the default config from .../mycelia/config.py

---

## 1) Constant Evaluation

Continuously gathers miners to evaluate, runs standardized evaluation, aggregates results, and publishes to the chain (and shares results with other validators).

```bash
python mycelia/validator/run.py \
  --path /home/isabella/crucible/subnet-MoE/checkpoints/validator/hk3/foundation/config.json
```

**What it does (typical flow):**

* Discovers miners ready for evaluation (queue/polling).
* Load model submissions (with resume/retry).
* Evaluates on a dataloader.
* Aggregates scores (per UID/hotkey) and handles hotkey changes correctly (history reset).
* Publishes scores to the chain and optionally exposes a summary for peers.

---

## 2) Model Serving (Updated Model)

Serves the updated model to two groups:

* **(a) Other validators/clients** — download the **full model**
* **(b) Miners** — download the **partial model** required for training (e.g., shared weights only)

```bash
python3 mycelia/shared/server.py \
  --path /home/isabella/crucible/subnet-MoE/checkpoints/validator/hk3/foundation/config.json
```

**Notes:**

* Starts an HTTP (or RPC) server as configured.
* Exposes endpoints for full vs partial model artifacts.
* Supports basic authentication/authorization if enabled in `config.json`.
* Logs requests and maintains simple indices under `serve/`.

---

## Running Both Together

Use two terminals (or `tmux`/`screen`):

```bash
# Terminal A: constant evaluation
python mycelia/validator/run.py --path .../mycelia/validator/hk3/foundation/config.json 

# Terminal B: model serving
python3 mycelia/shared/server.py --path .../mycelia/validator/hk3/foundation/config.json
```

---

## Tips

* Keep both processes pointed at the **same** `config.json`.
* Separate directories per validator/hotkey keep artifacts clean (`hk3/`, `hk4/`, …).
* Consider enabling **log rotation** for long-running services.
* If serving public endpoints, place the server behind a reverse proxy (nginx/traefik) and enable TLS.

---


## Troubleshooting

* **Evaluation stalls / no miners fetched**

* **Download errors / timeouts**

* **Dataloader worker exited unexpectedly**

* **High GPU memory usage**

* **Serve endpoints not reachable**

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your proposal. Remember to update tests and docs as appropriate.

