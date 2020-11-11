SHELL := /bin/bash
fdir = ./Manuscript/Figures
tdir = ./common/templates
pan_common = -F pandoc-crossref -C --filter=$(tdir)/figure-filter.py -f markdown ./Manuscript/Text/*.md

flist = 1 2 3 4 5 6 S1 S2 S4 S5 S7

.PHONY: clean test all testcover autopep spell

all: ckine/ckine.so Manuscript/Manuscript.pdf Manuscript/Manuscript.docx spell.txt

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

$(fdir)/figure%.svg: venv genFigures.py ckine/ckine.so graph_all.svg ckine/figures/figure%.py
	mkdir -p ./Manuscript/Figures
	. venv/bin/activate && ./genFigures.py $*

$(fdir)/figure%pdf: $(fdir)/figure%svg
	rsvg-convert --keep-image-data -f pdf $< -o $@

$(fdir)/figure%eps: $(fdir)/figure%svg
	rsvg-convert --keep-image-data -f eps $< -o $@

graph_all.svg: ckine/data/graph_all.gv
	dot $< -Tsvg -o $@

Manuscript/Manuscript.pdf: Manuscript/Text/*.md $(patsubst %, $(fdir)/figure%.pdf, $(flist)) Manuscript/gatingFigure.pdf
	pandoc -s $(pan_common) --fail-if-warnings --template=$(tdir)/default.latex --pdf-engine=xelatex -o $@

ckine/ckine.so: gcSolver/model.cpp gcSolver/model.hpp gcSolver/reaction.hpp gcSolver/makefile
	cd ./gcSolver && make ckine.so
	cp ./gcSolver/ckine.so ckine/ckine.so

Manuscript/Manuscript.docx: Manuscript/Text/*.md $(patsubst %, $(fdir)/figure%.eps, $(flist))
	cp -R $(fdir) ./
	pandoc -s $(pan_common) -o $@
	rm -r ./Figures

autopep:
	autopep8 -i -a --max-line-length 200 ckine/*.py ckine/figures/*.py

clean:
	rm -f ./Manuscript/Manuscript.* $(fdir)/figure* ckine/ckine.so .coverage coverage.xml graph_all.svg
	rm -rf venv

spell.txt: Manuscript/Text/*.md
	pandoc --lua-filter common/templates/spell.lua Manuscript/Text/*.md | sort | uniq -ic > spell.txt

test: venv ckine/ckine.so
	. venv/bin/activate && pytest

coverage.xml: venv ckine/ckine.so
	. venv/bin/activate && THEANO_FLAGS='mode=FAST_COMPILE' pytest --cov=ckine --cov-report=xml --cov-config=.github/workflows/coveragerc
