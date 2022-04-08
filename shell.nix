{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/2e8743b8e53638d8af54c74c023e0bb317557afb.tar.gz") {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.pkgconfig
    pkgs.powershell
    pkgs.cmake
    pkgs.cacert
    pkgs.dotnet-sdk
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pyopencl
  ];
}
