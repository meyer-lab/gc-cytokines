SHELL := /bin/bash
fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown ./Manuscript/Text/*.md
compile_opts = -std=c++14 -mavx -march=native -Wall -pthread

flist = 1 2 3 4 5 S1 S2 S3 S4 S5 B1 B2 B3 B4 B5

.PHONY: clean test all testprofile testcover doc testcpp autopep

all: ckine/ckine.so Manuscript/index.html Manuscript/Manuscript.pdf Manuscript/Manuscript.docx Manuscript/CoverLetter.docx pylint.log

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LINKFLAG = -Wl,-rpath=./ckine
endif

CPPLINKS = -I/usr/include/eigen3/ -I/usr/local/include/eigen3/ -lm -ladept -lsundials_cvodes -lsundials_cvode -lsundials_nvecserial -lcppunit

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

$(fdir)/figure%.svg: venv genFigures.py ckine/ckine.so graph_all.svg
	mkdir -p ./Manuscript/Figures
	. venv/bin/activate; ./genFigures.py $*

$(fdir)/figure%pdf: $(fdir)/figure%svg
	rsvg-convert -f pdf $< -o $@

$(fdir)/figure%eps: $(fdir)/figure%svg
	rsvg-convert -f eps $< -o $@

graph_all.svg: ckine/data/graph_all.gv
	dot $< -Tsvg -o $@

Manuscript/Manuscript.pdf: Manuscript/Manuscript.tex $(patsubst %, $(fdir)/figure%.pdf, $(flist))
	(cd ./Manuscript && latexmk -xelatex -f -quiet)
	rm -f ./Manuscript/Manuscript.b* ./Manuscript/Manuscript.aux ./Manuscript/Manuscript.fls

ckine/ckine.so: ckine/model.cpp ckine/model.hpp ckine/jacobian.hpp ckine/reaction.hpp
	clang++    $(compile_opts) -O3 $(CPPLINKS) ckine/model.cpp --shared -fPIC -o $@

ckine/libckine.debug.so: ckine/model.cpp ckine/model.hpp ckine/jacobian.hpp ckine/reaction.hpp
	clang++ -g $(compile_opts) -O3 $(CPPLINKS) ckine/model.cpp --shared -fPIC -o $@

ckine/cppcheck: ckine/libckine.debug.so ckine/model.hpp ckine/cppcheck.cpp ckine/jacobian.hpp ckine/reaction.hpp
	clang++ -g $(compile_opts) -L./ckine ckine/cppcheck.cpp $(CPPLINKS) -lckine.debug $(LINKFLAG) -o $@

Manuscript/index.html: Manuscript/Text/*.md $(patsubst %, $(fdir)/figure%.svg, $(flist))
	pandoc -s $(pan_common) -t html5 --mathjax -c ./Templates/kultiad.css --template=$(tdir)/html.template -o $@

Manuscript/Manuscript.docx: Manuscript/Text/*.md $(patsubst %, $(fdir)/figure%.eps, $(flist))
	mkdir -p ./Manuscript/Figures
	cp -R $(fdir) ./
	pandoc -s $(pan_common) -o $@
	rm -r ./Figures

Manuscript/Manuscript.tex: Manuscript/Text/*.md Manuscript/index.html
	pandoc -s $(pan_common) --template=$(tdir)/default.latex --pdf-engine=xelatex -o $@

Manuscript/CoverLetter.docx: Manuscript/CoverLetter.md
	pandoc -f markdown $< -o $@

Manuscript/CoverLetter.pdf: Manuscript/CoverLetter.md
	pandoc --pdf-engine=xelatex --template=/Users/asm/.pandoc/letter-templ.tex $< -o $@

autopep:
	autopep8 -i -a --max-line-length 200 ckine/*.py ckine/figures/*.py

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html Manuscript/CoverLetter.docx Manuscript/CoverLetter.pdf ckine/libckine.debug.so
	rm -f $(fdir)/Figure* ckine/ckine.so profile.p* stats.dat .coverage nosetests.xml coverage.xml ckine.out ckine/cppcheck testResults.xml
	rm -rf html ckine/*.dSYM doxy.log graph_all.svg valgrind.xml callgrind.out.* cprofile.svg venv
	find -iname "*.pyc" -delete

test: venv ckine/ckine.so
	. venv/bin/activate; pytest

testcover: venv ckine/ckine.so
	. venv/bin/activate; pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

testcpp: ckine/cppcheck
	valgrind ckine/cppcheck

cppcheck: ckine/cppcheck
	ckine/cppcheck
	
pylint.log: .pylintrc
	(pylint3 --rcfile=.pylintrc ckine > pylint.log || echo "pylint3 exited with $?")

doc:
	doxygen Doxyfile
