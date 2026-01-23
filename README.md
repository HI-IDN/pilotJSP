# ALICE Scheduling Framework

This repository is the C++ implementation of the ALICE framework described in the PhD thesis 
Ingimundardóttir, H. (2016). ALICE: Analysis & Learning Iterative Consecutive Executions 
(Doctoral dissertation).

The goal is to provide a modular library for building scheduling experiments, starting with Job-Shop
Scheduling (JSP) and extending to other shop types later on.

The current focus is data ingestion + configuration so we can load benchmark instances and wire them
to reusable scheduling components.

---

## Repository Structure

```
jsp/
+-- config/                # YAML configuration files
+-- data/                  # local sample JSP instances
+-- examples/              # small CLI examples
+-- include/               # public C++ headers
+-- src/                   # library sources
+-- CMakeLists.txt
+-- README.md
```

---

## Configuration

The library reads a YAML experiment file (currently `config/experiment.yaml`). The loader is
intentionally small and strict, so the config keys should match this schema:

```
domain: jsp
features: [makespan, slack, wrm]
data:
  name: j.rnd.4x5
  generator: taillard
  instance_size: { jobs: 4, machines: 5 }
  instances: 1
  durationLB: 1
  durationUB: 99
  set: train 
  file: data/Raw/j.rnd.4x5.train.txt
```

Notes:

- `domain` is the scheduling family (for example `jsp`).
- `features` toggles the feature set by name. If omitted or empty, all features for the domain are
  enabled by default.
- `file` is optional; if omitted, the loader builds a filename from `name` or `generator` +
  `instance_size`.

---

## Build and Run (Example)

```
cmake -S . -B build
cmake --build build
./build/alice_jsp_inspect config/experiment.yaml
```

The example reads the config, locates the data file, and prints basic dataset info.

---

## Citation

If you use this repository, cite the thesis (TODO update to new publication):

```
Ingimundardóttir, H. (2016). ALICE: Analysis & Learning Iterative Consecutive Executions (Doctoral dissertation).
```








