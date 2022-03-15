FUTHARK_BENCH_RUNS=10
FUTHARK=futhark
FUTHARK_BACKEND=cuda
FUTHARK_BENCH_OPTIONS=--pass-option=--default-tile-size=$(FUTHARK_TILE_SIZE) --pass-option=--default-reg-tile=$(FUTHARK_REG_TILE)

DATA_DIR=data
PY_RESULTS=lstm_py.json
FUT_RESULTS=lstm_fut.json
RESULTS=results.json

ifeq ($(shell if lshw -C display 2>/dev/null | grep -q 'A100'; then echo yes; fi),yes)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f64
else
	FUTHARK_TUNING=--tuning=.tuning.2080ti
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f32
endif

data:
	mkdir -p $(DATA_DIR)
	python3 -c 'import lstm; lstm.gen_data(data_dir="$(DATA_DIR)", prec="$(PRECISION)")'

$(PY_RESULTS): lstm.py data
	python3 -c 'import lstm; lstm.benchmarks(runs=$(FUTHARK_BENCH_RUNS), data_dir="$(DATA_DIR)", output="$(PY_RESULTS)", prec="$(PRECISION)")'

$(FUT_RESULTS): lstm.fut data
	@sed -E -r 's/f64|f32/$(PRECISION)/' $<
	$(FUTHARK) bench $^ $(FUTHARK_TUNING) $(FUTHARK_BENCH_OPTIONS) -r $(FUTHARK_BENCH_RUNS) --backend=$(FUTHARK_BACKEND) --json $@
	python3 -c 'import benchmark; benchmark.process_futhark("$(FUT_RESULTS)")'

results: $(PY_RESULTS) $(FUT_RESULTS) 
	python3 -c 'import benchmark; benchmark.dump(["$(PY_RESULTS)", "$(FUT_RESULTS)"], "$(RESULTS)")'

latex: $(PY_RESULTS) $(FUT_RESULTS) 
	@python3 -c 'import benchmark; benchmark.latex_lstm("$(PY_RESULTS)", "$(FUT_RESULTS)")'

.PHONY: clean

clean:
	rm -rf data *.json
