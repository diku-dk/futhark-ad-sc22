ifeq ($(GPU), A100)
	RESULTS=results-a100.json
	FUTHARK_BACKEND=cuda
else ifeq ($(GPU), 2080TI)
	RESULTS=results-2080ti.json
	FUTHARK_BACKEND=cuda
else ifeq ($(GPU), Q6000)
	RESULTS=results-q6000.json
	FUTHARK_BACKEND=cuda
else ifeq ($(GPU), MI100)
	RESULTS=results-mi100.json
        FUTHARK_BACKEND=opencl
else
	RESULTS=results.json
	FUTHARK_BACKEND=cuda
endif
JAC_SPEEDUP=True
OBJ_SPEEDUP=False
RUNS=10
FUTHARK=futhark
FUTHARK_BENCH_OPTIONS=--pass-option=--default-tile-size=$(FUTHARK_TILE_SIZE) --pass-option=--default-reg-tile=$(FUTHARK_REG_TILE)
PYTHONPATH=../..:..:$PYTHONPATH
PYTHON_CMD=PYTHONPATH=../../:../ $(PYTHON)
PRECISION=f32

%.json: %.py
	$(PYTHON_CMD) -c 'import $(basename  $<); $(basename $<).bench_all(runs=$(RUNS), output="$(basename $<).json", prec="$(PRECISION)")'

%.json: %.fut
	$(FUTHARK) bench $< $(FUTHARK_TUNING) $(FUTHARK_BENCH_OPTIONS) -r $(RUNS) --backend=$(FUTHARK_BACKEND) --json $@
	$(PYTHON_CMD) -c 'import benchmark; benchmark.process_futhark("$(NAME)", "$(basename $<).json", "$<", "$(patsubst *_%, %, $(basename $<))")'

gen_results:
	$(PYTHON_CMD) -c 'import benchmark; benchmark.dump("$(RESULTS)", jac_speedup=$(JAC_SPEEDUP), obj_speedup=$(OBJ_SPEEDUP))'

latex:
	@$(PYTHON_CMD) -c 'import benchmark; benchmark.latex("$(NAME)", $(JAC_SPEEDUP), $(OBJ_SPEEDUP), "$(RESULTS)")'

table:
	@$(PYTHON_CMD) -c 'import benchmark; benchmark.table("$(NAME)", "$(RESULTS)")'

.PHONY: clean

clean:
	rm -rf *.json *.c *.actual *.expected __pycache__
