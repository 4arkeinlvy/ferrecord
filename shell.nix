# Dev shell for NixOS: `nix-shell --run "cargo test"`
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = with pkgs; [
    pkg-config
    alsa-lib     # cpal (tools/vad-live) via the ALSA/PipeWire bridge
    ffmpeg       # ffprobe for media checks in tests
  ];
}
