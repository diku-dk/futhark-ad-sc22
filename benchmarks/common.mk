RESULTS=results.json
JAC_SPEEDUP=True
OBJ_SPEEDUP=False
RUNS=10
FUTHARK=futhark
FUTHARK_BACKEND=cuda
FUTHARK_BENCH_OPTIONS=--pass-option=--default-tile-size=$(FUTHARK_TILE_SIZE) --pass-option=--default-reg-tile=$(FUTHARK_REG_TILE)
PYTHONPATH=../..:..:$PYTHONPATH

%.json: %.py
	python3 -c 'import $(basename  $<); $(basename $<).benchmarks(runs=$(RUNS), output="$(basename $<).json")'

%.json: %.fut
	$(FUTHARK) bench $< $(FUTHARK_TUNING) $(FUTHARK_BENCH_OPTIONS) -r $(RUNS) --backend=$(FUTHARK_BACKEND) --json $@
ifeq ($(NAME),$(basename $<))
	python3 -c 'import benchmark; benchmark.process_futhark("$(basename $<).json", "$<")'
else
	TYPE=$(patsubst *_%, %, $<)
	python3 -c 'import benchmark; benchmark.process_futhark("$(basename $<).json", "$<", "$(TYPE)")'
endif

gen_results: $(shell find . -type f \( -name "*.json" ! -name "$(RESULTS)" \))
	python3 -c 'import benchmark; benchmark.dump("$(RESULTS)", jac_speedup=$(JAC_SPEEDUP), obj_speedup=$(OBJ_SPEEDUP))'

gen_latex: $(shell find . -type f \( -name "*.json" ! -name "$(RESULTS)" \))
	@python3 -c 'import benchmark; benchmark.latex_$(NAME)()'

.PHONY: clean

clean:
	rm -rf *.json *.c *.actual *.expected