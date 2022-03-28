include ../common.mk
NAME=gmm
CONVERT=true
ifeq ($(GPU), A100)
	FUTHARK_TUNING=--tuning=.tuning.a100
	FUTHARK_TILE_SIZE=16
	FUTHARK_REG_TILE=3
        PRECISION=f64
else  
ifeq ($(GPU), 2080TI)
	FUTHARK_TUNING=--tuning=.tuning.2080ti
	FUTHARK_TILE_SIZE=8
	FUTHARK_REG_TILE=4
        PRECISION=f32
else
        FUTHARK_BENCH_OPTIONS=
endif
endif

.PHONY: convert
convert:
	sed -E -r -i 's/f64|f32/$(PRECISION)/g' $(NAME).fut

$(RESULTS): convert gmm.json gmm_pytorch.json
	make gen_results
