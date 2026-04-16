# Simulating Lexical Semantic Change from Sense-Annotated Data

- - -
Siehe unten für die deutsche Version.
- - -

### Type

Dataset

### Authors

Dominik Schlechtweg, Sabine Schulte im Walde

### Description

This data collection supplementing the paper referenced below contains:

- a lemmatized English text corpus pair (SEMCOR1, SEMCOR2) based on [SemCor](https://nltk.github.com/nltk_data/packages/corpora/semcor.zip) in which lexical semantic change has been simulated (`corpora/`)
- a lexical semantic change detection testset containing 148 lemmas with frequencies >=50 in both SEMCOR1 and SEMCOR2 (`testset/`)

	The file `testset.tsv` contains the following information:

	- __lemma__: lemma
	- __T1__: sense frequency distribution in SEMCOR1
	- __T2__: sense frequency distribution in SEMCOR2
	- __freq1__: lemma frequency in SEMCOR1
	- __freq2__: lemma frequency in SEMCOR2
	- __freq_error__: relative frequency error of annotated frequency against final lemma frequency
	- __poly__: maximal number of senses in SEMCOR1 and SEMCOR2
	- __freq__: normalized frequency difference between freq1 and freq2
	- __graded__: graded change score of lemma, *G(lemma)*
	- __binary__: binary change score of lemma, *B(lemma)*

	The files `poly.tsv` and `freq.tsv` contain the scores for the polysemy and frequency baselines from the paper.

### Reference

Dominik Schlechtweg and Sabine Schulte im Walde. 2020. [Simulating Lexical Semantic Change from Sense-Annotated Data](https://brussels.evolang.org/proceedings/paper.html?nr=9). In Ravignani, A. and Barbieri, C. and Martins, M. and Flaherty, M. and Jadoul, Y. and Lattenkamp, E. and Little, H. and Mudd, K. and Verhoef, T. (Eds.): The Evolution of Language: Proceedings of the 13th International Conference (EvoLang13).

### Download

The resources are [freely available for education, research and other non-commercial purposes](https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/semcor_lsc.zip). More information can be requested via email to the authors.

- - -

# Simulation von Bedeutungswandel mit bedeutungsannotierten Daten

### Typ

Datensatz

### Autoren

Dominik Schlechtweg, Sabine Schulte im Walde

### Beschreibung

Diese Datensammlung ergänzt den unten zitierten Artikel und enthält:

- ein lemmatisiertes englisches Textkorpuspaar (SEMCOR1, SEMCOR2) basierend [SemCor](nltk.github.com/nltk_data/packages/corpora/semcor.zip), in dem Bedeutungswandel simuliert wurde (`corpora/`)
- einen Testdatensatz für Bedeutungswandelerkennung, der 148 Lemmata mit Frequenzen >=50 in sowohl SEMCOR1 als auch SEMCOR2 enthält (`testset/`)

	Die Datei `testset.tsv` enthält die folgenden Informationen:

	- __lemma__: Lemma
	- __T1__: Bedeutungsfrequenzverteilung in SEMCOR1
	- __T2__: Bedeutungsfrequenzverteilung in SEMCOR2
	- __freq1__: Lemma-Frequenz in SEMCOR1
	- __freq2__: Lemma-Frequenz in SEMCOR2
	- __freq_error__: relative Frequenzabweichung zwischen Frequenz annotierter Vorkommen und Gesamtfrequenz
	- __poly__: maximale Anzahl der Bedeutungen in SEMCOR1 und SEMCOR2
	- __freq__: normalisierte Frequenz-Differenz zwischen freq1 und freq2
	- __graded__: gradierter Bedeutungswandelwert für Lemma, *G(Lemma)*
	- __binary__: binärer Bedeutungswandelwert für Lemma, *B(Lemma)*

	Die Dateien `poly.tsv` und `freq.tsv` enthalten die Werte für die Polysemie-und Frequenz-Baselines aus dem Artikel.

### Reference

Dominik Schlechtweg and Sabine Schulte im Walde. 2020. [Simulating Lexical Semantic Change from Sense-Annotated Data](https://brussels.evolang.org/proceedings/paper.html?nr=9). In Ravignani, A. and Barbieri, C. and Martins, M. and Flaherty, M. and Jadoul, Y. and Lattenkamp, E. and Little, H. and Mudd, K. and Verhoef, T. (Eds.): The Evolution of Language: Proceedings of the 13th International Conference (EvoLang13).

### Download

Die Ressourcen sind [frei verfügbar für Lehre, Forschung sowie andere nicht-kommerzielle Zwecke](https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/semcor_lsc.zip). Für weitere Informationen schreiben Sie bitte eine E-Mail an die Autoren.
