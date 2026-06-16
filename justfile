set shell := ["bash", "-cu"]

default:
    @just --list

init:
    cargo check

run *args:
    cargo run -- {{args}}

run-json name="world":
    cargo run -- --name {{name}} --json

build:
    cargo build

release:
    cargo build --release

test:
    cargo test

nextest:
    cargo nextest run

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

lint:
    cargo clippy --all-targets --all-features -- -D warnings

check:
    cargo check --all-targets --all-features

docs:
    cargo doc --no-deps

coverage:
    cargo tarpaulin --out Html

coverage-llvm:
    cargo tarpaulin --engine llvm --out Html

watch:
    cargo watch -x check -x test

bacon:
    bacon

bench:
    cargo bench

nix-build:
    nix build

flake-check:
    nix flake check

ci: fmt-check lint test flake-check

clean:
    cargo clean
    rm -rf tarpaulin-report.html coverage
