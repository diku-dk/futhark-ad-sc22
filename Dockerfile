FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y sudo git curl git-lfs
RUN apt-get install -y python3 python3-pip

RUN adduser --gecos '' --disabled-password bench
RUN echo "bench ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers
USER bench

RUN pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install futhark-data prettytable

RUN [ "/bin/bash", "-c", "sh <(curl -L https://nixos.org/nix/install) --no-daemon" ]
RUN . /home/bench/.nix-profile/etc/profile.d/nix.sh
ENV PATH=/home/bench/.local/bin:/home/bench/.nix-profile/bin:$PATH
RUN nix-channel --update

RUN git lfs install
WORKDIR /home/bench/
RUN git clone https://github.com/diku-dk/futhark-ad-sc22
WORKDIR /home/bench/futhark-ad-sc22
RUN git submodule update --init ADBench
RUN git submodule update --init futhark
RUN make -B bin/futhark
