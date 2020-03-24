SHELL := /bin/bash
fdir = ./Manuscript/Figures
tdir = ./common/templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown 

flist = 1 2 3 4 5 6 S1 S2 S4 S5 S7

.PHONY: clean test all testcover autopep spell

all: ckine/ckine.so Manuscript/Manuscript.pdf Manuscript/Manuscript.docx pylint.log spell.txt

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

Manuscript/Manuscript.pdf: Manuscript/Text/*.md $(patsubst %, $(fdir)/figure%.pdf, $(flist)) Manuscript/gatingFigure.pdf
	pandoc -s $(pan_common) ./Manuscript/Text/*.md --fail-if-warnings --template=Manuscript/pnas.tex --pdf-engine=xelatex -o $@

Manuscript/Supplement.pdf: Manuscript/Supplement.md $(patsubst %, $(fdir)/figure%.pdf, $(flist))
	pandoc -s $(pan_common) Manuscript/Supplement.md --fail-if-warnings --template=Manuscript/pnasSI.tex --pdf-engine=xelatex -o $@

ckine/ckine.so: gcSolver/model.cpp gcSolver/model.hpp gcSolver/reaction.hpp gcSolver/makefile
	cd ./gcSolver && make ckine.so
	cp ./gcSolver/ckine.so ckine/ckine.so

autopep:
	autopep8 -i -a --max-line-length 200 ckine/*.py ckine/figures/*.py

clean:
	rm -f ./Manuscript/Manuscript.*
	rm -f $(fdir)/figure* ckine/ckine.so profile.p* stats.dat .coverage nosetests.xml coverage.xml ckine.out ckine/cppcheck testResults.xml
	rm -rf html ckine/*.dSYM doxy.log graph_all.svg valgrind.xml callgrind.out.* cprofile.svg venv
	find -iname "*.pyc" -delete

spell.txt: Manuscript/Text/*.md
	pandoc --lua-filter common/templates/spell.lua Manuscript/Text/*.md | sort | uniq -ic > spell.txt

test: venv ckine/ckine.so
	. venv/bin/activate && pytest

testcover: venv ckine/ckine.so
	. venv/bin/activate && THEANO_FLAGS='mode=FAST_COMPILE' pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc ckine > pylint.log || echo "pylint3 exited with $?")
