let
  pkgs = import <nixpkgs> {};
in
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
