include ../../common.mk
NAME=kmeans_sparse
JAC_SPEEDUP=False
OBJ_SPEEDUP=True
FUTHARK_BENCH_OPTIONS=
ifeq ($(GPU), A100)
        JAX=TRUE
endif

.PHONY: results

results: kmeans_sparse.json kmeans_sparse_pytorch.json kmeans_sparse_manual.json $(if $(JAX),kmeans_sparse_jax.json)
	make gen_results

