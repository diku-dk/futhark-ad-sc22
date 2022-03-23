include ../../common.mk
NAME=kmeans
JAC_SPEEDUP=False
OBJ_SPEEDUP=True
FUTHARK_BENCH_OPTIONS=

$(RESULTS): kmeans.json kmeans_pytorch.json kmeans_manual.json kmeans_jax.json
	make gen_results
