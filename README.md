# pilotJSP
Code repository accompanying the LION20 conference paper:

**Learning from Expert Optimization: Expert‑Like Lookahead Policies via Pilot Heuristics**

This project implements an active imitation‑learning pipeline (DAgger) for the Job‑Shop Scheduling Problem (JSP). The goal is to learn a dispatching model that behaves like an expert MIP solver and can be used as a pilot heuristic inside a limited‑lookahead scheduling procedure.

---

## 🔧 Overview of the Pipeline

The pipeline performs the following steps:

1. **Generate benchmark JSP instances** using the official Colorado State University JSP generator:  
   https://www.cs.colostate.edu/sched/generator/

2. **Query an optimal expert (MIP) solver**  
   Gurobi (v13.0 or newer) is used to compute optimal dispatch decisions at each scheduling step.

3. **Extract 13 standard dispatching features**  
   Computed at each decision point (100 steps for a 10×10 JSP).

4. **Construct pairwise preference datasets**  
   Expert decisions are transformed into ordinal regression targets.

5. **Train a dispatching model**  
   A learned model is embedded inside a pilot heuristic with limited lookahead.

---

## 📁 Repository Structure

```
pilotJSP/
├── config/                # YAML configuration files
├── data/                  # <gitignored> generated JSP instances
├── code/                  # main codebase
└── README.md
```

---

## ⚙️ Configuration

All experiment parameters are controlled using a YAML file, e.g.:
```
config/experiment.yaml
```
The configuration file is **self‑documenting**, so the README does not duplicate parameter descriptions.

Run an experiment:
```bash
make fspgen # downloads and builds the problem generator
make generate_problems # creates the problem instances and labels them
```

---

## 📦 Generating JSP Instances

This repository uses two families of instances:

- **jsp-rnd** — uniform random processing times
- **jsp-rndn** — narrow distribution (reduced variance)

You can create instances automatically using our script. Run:
```bash
bash scripts/generate_instances.sh
```
This downloads and extracts the desired number of instances.

---

## 🗂️ Data Directory
All generated data is stored under:
```
data/
    jsp-rnd/
    jsp-rndn/
```


---

## ▶️ Running an Experiment

```bash
python src/train.py --config config/experiment.yaml
```
This performs:
- instance loading/generation
- expert querying
- feature computation
- dataset construction
- model training
- evaluation with a pilot heuristic

---

## 📖 Citation
If you use this repository, please cite the LION20 paper:
```
[Insert citation entry]
```

---
