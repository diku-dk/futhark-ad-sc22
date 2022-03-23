RESULTS=results.json
JAC_SPEEDUP=True
OBJ_SPEEDUP=False
RUNS=10
FUTHARK=futhark
FUTHARK_BACKEND=cuda
FUTHARK_BENCH_OPTIONS=--pass-option=--default-tile-size=$(FUTHARK_TILE_SIZE) --pass-option=--default-reg-tile=$(FUTHARK_REG_TILE)
PYTHONPATH=../..:..:$PYTHONPATH
PYTHON_CMD=PYTHONPATH=../../:../ $(PYTHON)

%.json: %.py
	$(PYTHON_CMD) -c 'import $(basename  $<); $(basename $<).bench_all(runs=$(RUNS), output="$(basename $<).json")'

%.json: %.fut
	$(FUTHARK) bench $< $(FUTHARK_TUNING) $(FUTHARK_BENCH_OPTIONS) -r $(RUNS) --backend=$(FUTHARK_BACKEND) --json $@
ifeq ($(NAME),$(basename $<))
	$(PYTHON_CMD) -c 'import benchmark; benchmark.process_futhark("$(basename $<).json", "$<")'
else
	TYPE=$(patsubst *_%, %, $<)
	$(PYTHON_CMD) -c 'import benchmark; benchmark.process_futhark("$(basename $<).json", "$<", "$(TYPE)")'
endif

gen_results: $(shell find . -type f \( -name "*.json" ! -name "$(RESULTS)" \))
	$(PYTHON_CMD) -c 'import benchmark; benchmark.dump("$(RESULTS)", jac_speedup=$(JAC_SPEEDUP), obj_speedup=$(OBJ_SPEEDUP))'

gen_latex: $(shell find . -type f \( -name "*.json" ! -name "$(RESULTS)" \))
	@$(PYTHON_CMD) -c 'import benchmark; benchmark.latex("$(NAME)")'

.PHONY: clean

clean:
	rm -rf *.json *.c *.actual *.expected $(NAME) __pycache__  $(find . -executable -type f)
