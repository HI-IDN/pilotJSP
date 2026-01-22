# Makefile
SHELL := /bin/bash

# -------------------------------------------------------------------
# CONFIG — Read from YAML using yq
# -------------------------------------------------------------------
YQ       := yq -r
CFG      := ../config/experiment.yaml

GENERATOR := $(shell $(YQ) '.data.generator' $(CFG))
JOBS      := $(shell $(YQ) '.data.instance_size.jobs' $(CFG))
MACHINES  := $(shell $(YQ) '.data.instance_size.machines' $(CFG))
COUNT     := $(shell $(YQ) '.data.instances' $(CFG))
NAME      := $(shell $(YQ) '.data.name' $(CFG))

OUT_ROOT  := ../data/$(NAME)

# -------------------------------------------------------------------
# fspgen download + build into external/fspgen
# -------------------------------------------------------------------
GEN_DIR     := ../external/fspgen
FSPGEN_BIN  := $(GEN_DIR)/fspgen
FSPGEN_TGZ  := $(GEN_DIR)/fspgen.tar.gz
FSPGEN_URL  := https://www.cs.colostate.edu/sched/generator/fspgen.tar.gz

$(FSPGEN_BIN):
	@echo "[INFO] Building fspgen..."
	mkdir -p $(GEN_DIR)
	curl -L -o $(FSPGEN_TGZ) $(FSPGEN_URL)
	tar -xzf $(FSPGEN_TGZ) -C $(GEN_DIR)
	$(MAKE) -C $(GEN_DIR)

.PHONY: fspgen
fspgen: $(FSPGEN_BIN)
	@echo "[OK] fspgen is ready at $(FSPGEN_BIN)"
clean-fspgen:
	rm -rf $(GEN_DIR)



# -------------------------------------------------------------------
# Generate data
# -------------------------------------------------------------------
.ONESHELL:
$(OUT_ROOT)/problem_%.txt: $(FSPGEN_BIN)
	seed=$$( printf "%d" $$(echo $* | sed 's/[^0-9]*//g') ); \
	mkdir -p $(OUT_ROOT); \
	lb="$$( $(YQ) '.data.durationLB // ""' $(CFG) )"; \
	ub="$$( $(YQ) '.data.durationUB // ""' $(CFG) )"; \
	extra=" -durationLB=$$lb -durationUB=$$ub "; \
	echo "[GEN] $(NAME) seed=$$seed LB=$$lb UB=$$ub"; \
	$(FSPGEN_BIN) $(GENERATOR) $(JOBS) $(MACHINES) $$seed $$extra > $@;
	echo "[OK] Generated $@"


$(OUT_ROOT)/problem_%.json: $(OUT_ROOT)/problem_%.txt | label_problem_data.py
	@echo "[LABEL] Converting $< -> $@"
	python3 label_problem_data.py --input $< --output $@
	@echo "[OK] Converted $< to $@"

# -------------------------------------------------------------------
# Build the list of expected files
# -------------------------------------------------------------------
ALL_TARGETS := $(foreach n,$(shell seq -f "%04g" 1 $(COUNT)), \
                     $(OUT_ROOT)/problem_$(n).txt)

.PHONY: generate_problems
generate_problems: fspgen
generate_problems: $(ALL_TARGETS)
	@echo "[OK] All data generated."


# ---------------------------------------------------
# Clean up old data
# ---------------------------------------------------
clean:
	rm -rf data


