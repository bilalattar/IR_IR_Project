import pyterrier as pt
from src.L2R import filter_topics
from src.preprocessing import _remove_stops

if not pt.started():
    pt.init()

dataset = pt.get_dataset("trec-deep-learning-passages")
index = pt.IndexFactory.of("./passage_index_correct/data.properties")
BM25_br = pt.apply.query(_remove_stops) >> pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % 1000

# Filter topic list
topics = dataset.get_topics("test-2019")
qrels = dataset.get_qrels("test-2019")
newTopics = filter_topics(topics, qrels)

# Run experiment for whole dataset
# result = pt.Experiment(
#     [BM25_br],
#     newTopics,
#     qrels,
#     eval_metrics=["num_q", "recip_rank", "ndcg_cut_10", "map"])
#

# Run experiment per query
result = pt.Experiment(
    [BM25_br],
    newTopics,
    qrels,
    eval_metrics=["ndcg_cut_30", "map", "recip_rank"],
    perquery=True)