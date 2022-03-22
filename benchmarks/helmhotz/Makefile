include ../common.mk
NAME=helmholtz

ifeq ($(shell if lshw -C display 2>/dev/null | grep -q 'A100'; then echo yes; fi),yes)
	FUTHARK_TUNING=--tuning=.tuning
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f32
else
	FUTHARK_TUNING=--tuning=.tuning
	FUTHARK_TILE_SIZE=4
	FUTHARK_REG_TILE=4
        PRECISION=f32
endif

.PHONY: results
results: $(NAME).json $(NAME)_pytorch.json
