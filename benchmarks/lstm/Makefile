include ../common.mk
NAME=lstm

ifeq ($(shell if lshw -C display 2>/dev/null | grep -q 'A100'; then echo yes; fi),yes)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f64
else
	FUTHARK_TUNING=--tuning=.tuning.2080ti
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=4
        PRECISION=f32
endif

.PHONY: results
results: data lstm.json lstm_pytorch.json
	make gen_results

data:
	mkdir -p data
	python3 -c 'import lstm_pytorch; lstm_pytorch.gen_data(data_dir="data", prec="$(PRECISION)")'
