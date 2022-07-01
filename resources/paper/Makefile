NAME=thesis
all: $(NAME).pdf

SECTIONS := $(shell echo ./sections/*)

$(NAME).pdf: $(NAME).tex bibliography.bib $(SECTIONS)
	pdflatex $(NAME)
	bibtex $(NAME).aux || true  #bibtex geeft foutmeldingen als nederlands wordt gebruikt met `|| true' zijn die te negeren, wees hier voorzichtig mee en verwijder het als je in het engels gaat schrijven
	pdflatex $(NAME)
	pdflatex $(NAME)
	rm $(NAME).aux $(NAME).bbl  $(NAME).blg  $(NAME).log

clean:
	rm $(NAME).aux $(NAME).bbl  $(NAME).blg  $(NAME).log
	rm $(NAME).pdf
