CFLAGS?=-O3
LDFLAGS?=-lOpenCL -lm

lbm: main.c lbm.c ocl.c parboil_opencl.c
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: run

run: lbm
	./lbm 100 -o out -i 120_120_150_ldc.of
