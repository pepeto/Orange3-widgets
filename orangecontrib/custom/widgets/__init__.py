from .gmm_widget import OWGMM
from .hmm_widget import OWHMM
from .yf import OWCustYahooFinance
from .zigzag import OWZigZag
from .lgbm_optuna import OWLightGBMOptuna

# Lista de widgets a registrar:
WIDGETS = [OWGMM, OWHMM, OWCustYahooFinance, OWZigZag, OWLightGBMOptuna]
