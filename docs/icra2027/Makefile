.PHONY: all figures check
all: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper_a.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper_b.tex
	cp build/paper_a.pdf learning_to_open_and_remove.pdf
	cp build/paper_b.pdf roboevolve.pdf
figures:
	python make_figures.py
check:
	python check_papers.py
