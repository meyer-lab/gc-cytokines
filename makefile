fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown ./Manuscript/Text/*.md

.PHONY: clean test all testprofile testcover doc testcpp

all: ckine/ckine.so Manuscript/index.html Manuscript/Manuscript.pdf Manuscript/Manuscript.docx Manuscript/CoverLetter.docx

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LINKFLAG = -Wl,-rpath=./ckine
endif

CPPLINKS = -lm -lsundials_cvode -lsundials_nvecserial -lcppunit

$(fdir)/Figure%.svg: genFigures.py
	mkdir -p ./Manuscript/Figures
	python3 genFigures.py $*

$(fdir)/Figure%pdf: $(fdir)/Figure%svg
	rsvg-convert -f pdf $< -o $@

$(fdir)/Figure%eps: $(fdir)/Figure%svg
	rsvg-convert -f eps $< -o $@

Manuscript/Manuscript.pdf: Manuscript/Manuscript.tex
	(cd ./Manuscript && latexmk -xelatex -f -quiet)
	rm -f ./Manuscript/Manuscript.b* ./Manuscript/Manuscript.aux ./Manuscript/Manuscript.fls

ckine/ckine.so: ckine/model.cpp ckine/model.hpp
	clang++    -std=c++11 -mavx -march=native -O3 $(CPPLINKS) ckine/model.cpp --shared -fPIC -o $@

ckine/libckine.debug.so: ckine/model.cpp ckine/model.hpp
	clang++ -g -std=c++11 -mavx -march=native -O3 $(CPPLINKS) ckine/model.cpp --shared -fPIC -o $@

ckine/cppcheck: ckine/libckine.debug.so ckine/model.hpp ckine/cppcheck.cpp
	clang++ -g -std=c++11 -L./ckine ckine/cppcheck.cpp $(CPPLINKS) -lckine.debug $(LINKFLAG) -o $@

Manuscript/index.html: Manuscript/Text/*.md
	pandoc -s $(pan_common) -t html5 --mathjax -c ./Templates/kultiad.css --template=$(tdir)/html.template -o $@

Manuscript/Manuscript.docx: Manuscript/Text/*.md
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

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html Manuscript/CoverLetter.docx Manuscript/CoverLetter.pdf
	rm -f $(fdir)/Figure* ckine/ckine.so profile.p* stats.dat .coverage nosetests.xml coverage.xml ckine.out ckine/cppcheck testResults.xml
	rm -rf docs/build/* docs/build/.buildinfo docs/build/.doctrees docs/build/.nojekyll docs/source/ckine* docs/source/modules.rst
	rm -rf ckine/*.dSYM
	rm -f ckine/libckine.debug.so

test: ckine/ckine.so
	nosetests3 -s --with-timer --timer-top-n 5

testcover: ckine/ckine.so
	nosetests3 --with-xunit --with-xcoverage --cover-package=ckine -s --with-timer --timer-top-n 5

stats.dat: ckine/ckine.so
	nosetests3 -s --with-cprofile

testprofile: stats.dat
	pyprof2calltree -i stats.dat -k

testcpp: ckine/cppcheck
	ckine/cppcheck

doc: ckine/ckine.so
	sphinx-apidoc -o docs/source ckine
	sphinx-build docs/source docs/build
