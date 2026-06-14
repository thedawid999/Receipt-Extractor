import re 
from rapidfuzz import fuzz

def extract_data(results: list[list[dict[str, int, int]]]):
