SHELL := /bin/bash
fdir = ./Manuscript/Figures
tdir = ./common/templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown ./Manuscript/Text/*.md
compile_opts = -std=c++14 -mavx -march=native -Wall -pthread

flist = 1 2 3 4 5 S1 S2 S4 S5 B1 B2 B3 B4 B5

.PHONY: clean test all testprofile testcover doc testcpp autopep spell leaks

all: ckine/ckine.so Manuscript/Manuscript.pdf Manuscript/Manuscript.docx Manuscript/CoverLetter.docx pylint.log

CPPLINKS = -I/usr/include/eigen3/ -I/usr/local/include/eigen3/ -lm -ladept -lsundials_cvodes -lsundials_cvode -lsundials_nvecserial -lstdc++ -lcppunit

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv --system-site-packages venv
	. venv/bin/activate && pip install -Ur requirements.txt
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
	pandoc -s $(pan_common) --template=$(tdir)/default.latex --pdf-engine=xelatex -o $@

ckine/ckine.so: ckine/model.cpp ckine/model.hpp ckine/reaction.hpp
	g++ $(compile_opts) -O3 $(CPPLINKS) ckine/model.cpp --shared -fPIC $(CPPLINKS) -o $@

ckine/libckine.debug.so: ckine/model.cpp ckine/model.hpp ckine/reaction.hpp
	g++ -g $(compile_opts) -O3 $(CPPLINKS) ckine/model.cpp --shared -fPIC $(CPPLINKS) -o $@

ckine/cppcheck: ckine/libckine.debug.so ckine/model.hpp ckine/cppcheck.cpp ckine/reaction.hpp
	g++ -g $(compile_opts) -L./ckine ckine/cppcheck.cpp $(CPPLINKS) -lckine.debug -Wl,-rpath=./ckine -o $@

Manuscript/Manuscript.docx: Manuscript/Text/*.md $(patsubst %, $(fdir)/figure%.eps, $(flist))
	cp -R $(fdir) ./
	pandoc -s $(pan_common) -o $@
	rm -r ./Figures

Manuscript/CoverLetter.docx: Manuscript/CoverLetter.md
	pandoc -f markdown $< -o $@

Manuscript/CoverLetter.pdf: Manuscript/CoverLetter.md
	pandoc --pdf-engine=xelatex --template=/Users/asm/.pandoc/letter-templ.tex $< -o $@

autopep:
	autopep8 -i -a --max-line-length 200 ckine/*.py ckine/figures/*.py

clean:
	rm -f ./Manuscript/Manuscript.* Manuscript/CoverLetter.docx Manuscript/CoverLetter.pdf ckine/libckine.debug.so
	rm -f $(fdir)/figure* ckine/ckine.so profile.p* stats.dat .coverage nosetests.xml coverage.xml ckine.out ckine/cppcheck testResults.xml
	rm -rf html ckine/*.dSYM doxy.log graph_all.svg valgrind.xml callgrind.out.* cprofile.svg venv
	find -iname "*.pyc" -delete

spell: Manuscript/Text/*.md
	pandoc --lua-filter common/templates/spell.lua Manuscript/Text/*.md | sort | uniq -ic

test: venv ckine/ckine.so
	. venv/bin/activate && pytest

testcover: venv ckine/ckine.so
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

testcpp: venv ckine/cppcheck
	valgrind --tool=callgrind ckine/cppcheck
	. venv/bin/activate && gprof2dot -f callgrind -n 1.0 callgrind.out.* | dot -Tsvg -o cprofile.svg

leaks: venv ckine/cppcheck
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --trace-children=yes ckine/cppcheck

cppcheck: ckine/cppcheck
	ckine/cppcheck
	
pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc ckine > pylint.log || echo "pylint3 exited with $?")

doc:
	doxygen Doxyfile
