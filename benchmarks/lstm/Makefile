include ../common.mk
NAME=lstm
PRECISION=f32
ifeq ($(GPU), A100)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=4
else  ifeq ($(GPU), MI100)
	FUTHARK_TUNING=--tuning=.tuning.mi100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=4
else  ifeq ($(GPU), 2080TI)
	FUTHARK_TUNING=--tuning=.tuning.2080ti
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=4
else
        FUTHARK_BENCH_OPTIONS=
endif

results: data lstm.json lstm_pytorch.json lstm_jax.json
	make gen_results

data:
	mkdir -p data
	$(PYTHON_CMD) -c 'import lstm_pytorch; lstm_pytorch.gen_data(data_dir="data", prec="$(PRECISION)")'
