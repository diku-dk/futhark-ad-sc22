bin/futhark:
	cd futhark && nix-build --argstr suffix ad
	mkdir -p bin
	tar --extract -C bin/ --strip-components 2 -f futhark/result/futhark-ad.tar.xz futhark-ad/bin/futhark
