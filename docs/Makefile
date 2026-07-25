# Documentation build for SLCETools.jl.
#
#   make build        build the HTML site into build/
#   make serve        build, then serve it locally with live reload (LiveServer.jl)
#   make open         open the built site in a browser
#   make clean        remove the build/ output
#
# The first run resolves docs/Project.toml (Documenter + LiveServer + SLCETools +
# SLCE, both by path). SLCE must sit at ../../SLCE.jl.

.PHONY: build serve open clean help

build:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); include("make.jl")'

serve: build
	julia --project=. -e 'using LiveServer; serve(dir = "build")'

open:
	open build/index.html

clean:
	rm -rf build/

help:
	@echo "build  - build the HTML documentation into build/"
	@echo "serve  - build and serve locally (live reload)"
	@echo "open   - open the built site in a browser"
	@echo "clean  - remove build/"
