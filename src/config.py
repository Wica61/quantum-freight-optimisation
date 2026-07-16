# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

GUROBI_OPTIONS = {
    "WLSACCESSID": os.environ["GRB_WLSACCESSID"],
    "WLSSECRET": os.environ["GRB_WLSSECRET"],
    "LICENSEID": int(os.environ["GRB_LICENSEID"]),
}