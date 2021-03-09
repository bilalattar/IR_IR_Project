import pyterrier as pt
from nltk.stem import PorterStemmer

if not pt.started():
    pt.init()

# Remove stopwords for queries
def _remove_stops(q):
    terms = q["query"].split(" ")
    terms = [t for t in terms if not t in stops]
    return " ".join(terms)

porter = PorterStemmer()
stops=set(["is", "for", "of", "are", "to", "a", "do", "in", "the", "and", "what",
           "how", "an", "why", "who", "from", "was", "when", "does", "can", "did", "or", "you"])