SHELL := /bin/bash
fdir = ./Manuscript/Figures
tdir = ./common/templates
pan_common = -F pandoc-crossref --natbib --filter=$(tdir)/figure-filter.py -f markdown+raw_attribute

flist = 1 2 3 4 5 6 S1 S2 S4 S5 S7

.PHONY: clean test all testcover autopep spell

all: ckine/ckine.so Manuscript/Manuscript.pdf pylint.log spell.txt Manuscript/Supplement.pdf

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

graph_all.svg: ckine/data/graph_all.gv
	dot $< -Tsvg -o $@

Manuscript/Manuscript.tex: Manuscript/Text/*.md
	pandoc -s $(pan_common) ./Manuscript/Text/*.md --template=Manuscript/pnas.tex -o $@

Manuscript/Manuscript.pdf: Manuscript/Manuscript.tex $(patsubst %, $(fdir)/figure%.pdf, $(flist)) Manuscript/gatingFigure.pdf
	pdflatex -interaction=batchmode -output-directory=Manuscript Manuscript.tex
	cd Manuscript && bibtex Manuscript
	pdflatex -interaction=batchmode -output-directory=Manuscript Manuscript.tex
	pdflatex -interaction=batchmode -output-directory=Manuscript Manuscript.tex
	rm -f Manuscript/Manuscript.aux Manuscript/Manuscript.bbl Manuscript/Manuscript.blg

Manuscript/Supplement.pdf: Manuscript/Supplement.md $(patsubst %, $(fdir)/figure%.pdf, $(flist))
	pandoc -s $(pan_common) Manuscript/Supplement.md --template=Manuscript/pnasSI.tex --pdf-engine=pdflatex -o $@

ckine/ckine.so: gcSolver/model.cpp gcSolver/model.hpp gcSolver/reaction.hpp gcSolver/makefile
	cd ./gcSolver && make ckine.so
	mv ./gcSolver/ckine.so ckine/ckine.so

autopep:
	autopep8 -i -a --max-line-length 200 ckine/*.py ckine/figures/*.py

clean:
	rm -f ./Manuscript/Manuscript.* Manuscript/Supplement.pdf
	rm -f $(fdir)/figure* ckine/ckine.so stats.dat .coverage nosetests.xml coverage.xml ckine/cppcheck testResults.xml
	rm -rf html doxy.log graph_all.svg venv
	find -iname "*.pyc" -delete

cleanms:
	rm -f ./Manuscript/Manuscript.* Manuscript/Supplement.pdf

spell.txt: Manuscript/Text/*.md
	pandoc --lua-filter common/templates/spell.lua Manuscript/Text/*.md | sort | uniq -ic > spell.txt

test: venv ckine/ckine.so
	. venv/bin/activate && pytest

testcover: venv ckine/ckine.so
	. venv/bin/activate && THEANO_FLAGS='mode=FAST_COMPILE' pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc ckine > pylint.log || echo "pylint3 exited with $?")
