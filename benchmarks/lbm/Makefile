FUTHARK=futhark
FUTHARK_BACKEND=cuda
RUNS=10
NAME=lbm

.PHONY: results clean
results: $(NAME).json

clean:
	rm -f $(NAME).json

$(NAME).json: $(NAME).fut
	$(FUTHARK) bench --backend=$(FUTHARK_BACKEND) --runs=$(RUNS) lbm.fut --json $@ $<
