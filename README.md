******
Bayesian Multilateration - BMLAT
******

**BMLAT** is Bayesian approach to the multilateration task.

you can create a development environment in conda with all dependencies through command

	conda create --name <env> --file ./src/requirements.txt

## source code

[Python example](src/example.py)<br>
[Jupyter Notebook example](src/example.ipynb)<br>
[Analysis of the parameters of the Nakagami distribution reparametrization adopted](src/nakagami_spread_parameter.ipynb)<br>
[Stan model](src/stan/bmlat.stan)<br>

## Pseudo-codes

[bootstrap sampler](pseudocode/alg01.png)<br>
[BMLAT sampler](pseudocode/alg02.png)<br>

---



Citation
========

To cite BMLAT in publications use:

Alencar, Alisson; Mattos, César; Gomes, João; Mesquita, Diego (2021): Bayesian Multilateration. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.16806688.v1 

Or use the following BibTeX entry::

	@article{Alencar2021,
		author = "Alisson Alencar and César Mattos and João Gomes and Diego Mesquita",
		title = "{Bayesian Multilateration}",
		year = "2021",
		month = "10",
		url = "https://www.techrxiv.org/articles/preprint/Bayesian_Multilateration/16806688",
		doi = "10.36227/techrxiv.16806688.v1"
	}

