fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown ./Manuscript/Text/*.md

.PHONY: clean test all testprofile testcover

all: ckine/ckine.so Manuscript/index.html Manuscript/Manuscript.pdf Manuscript/Manuscript.docx Manuscript/CoverLetter.docx

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

ckine/ckine.so: ckine/model.cpp
	g++ -std=c++11 ckine/model.cpp -O3 --shared -fPIC -lsundials_cvode -lsundials_nvecserial -lm -o ckine/ckine.so

Manuscript/index.html: Manuscript/Text/*.md
	pandoc -s $(pan_common) -t html5 --mathjax -c ./Templates/kultiad.css --template=$(tdir)/html.template -o $@

Manuscript/Manuscript.docx: Manuscript/Text/*.md
	mkdir -p ./Manuscript/Figures
	cp -R $(fdir) ./
	pandoc -s $(pan_common) -o $@
	rm -r ./Figures

Manuscript/Manuscript.tex: Manuscript/Text/*.md Manuscript/index.html
	pandoc -s $(pan_common) --template=$(tdir)/default.latex --latex-engine=xelatex -o $@

Manuscript/CoverLetter.docx: Manuscript/CoverLetter.md
	pandoc -f markdown $< -o $@

Manuscript/CoverLetter.pdf: Manuscript/CoverLetter.md
	pandoc --latex-engine=xelatex --template=/Users/asm/.pandoc/letter-templ.tex $< -o $@

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html Manuscript/CoverLetter.docx Manuscript/CoverLetter.pdf
	rm -f $(fdir)/Figure* ckine/ckine.so profile.p* stats.dat .coverage nosetests.xml coverage.xml ckine.out

test: ckine/ckine.so
	nosetests3 -s --with-timer --timer-top-n 5

testcover: ckine/ckine.so
	nosetests3 --with-xunit --with-xcoverage --cover-package=ckine -s --with-timer --timer-top-n 5

stats.dat: ckine/ckine.so
	nosetests3 -s --with-cprofile

testprofile: stats.dat
	pyprof2calltree -i stats.dat -k
