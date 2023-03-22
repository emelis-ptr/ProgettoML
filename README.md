# ProgettoML
Progetto per il corso di ML

# Configurazione Python - PyCharm

Prerequisiti
Bisogna creare un virtual environment per evitare conflitti con altre versioni di python installate.

**Ubuntu**: Se non hai pip, installalo con 

    $ sudo apt install python3-pip

installa virtualenv con 

    $ sudo apt install python3-virtualenv

**Windows**: Installa i virtualenv dalla versione attuale di python, ad esempio 

    $ python[3] -m pip install virtualenv

N.B.: Fuori dal virtualenv, usa `python3`, dentro puoi usare `python`

Su Pycharm:
- premi Ctrl+Alt+S
- Cerca "interpreter"
- Su Project Interpreter: <NomeProgetto> | Python Intepreter, clicca il link "Add interpreter" e poi "Add Local Interpreter"
- Se non hai creato alcun virtual environment (cartella /venv), allora basta premere OK. Altrimenti devi selezionare il virtual environment esistente.

Per disattivare il virtual environment:

    $ deactivate

Il comando deactivate è disponibile solo quando è attivo il virtual environment, cioè quando vedi (venv) nella linea di comando.

IMPORTANTE: su PyCharm marca la cartella `code` come Source Root e `tests` come Test Source Root, altrimenti le importazioni non funzioneranno.

# Test

Per lanciare i test, scaricare prima pytest:

    python -m pip install pytest

Dopodiché nella cartella root del progetto esegui:

    pytest

Ecco un esempio di test per una funzione:

```python
import pytest

def sum(a,b):
    return a+b

def sum_test():
    assert sum(1,2) == 3
    assert sum(0,0) == 0
    assert sum(-1,1) == 0
    with pytest.raises(TypeError):    
        sum("a", "b")
```
