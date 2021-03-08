import pandas as pd
import pyterrier as pt
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing import _remove_stops

if not pt.started():
    pt.init()

# Filter topic list
def filter_topics(t, q):
    data = []
    uniqueList = q['qid'].unique()
    for index, row in t.iterrows():
        if row['qid'] in uniqueList:
            data.append([row['qid'], row['query']])
    newTopics = pd.DataFrame(data, columns=['qid', 'query'])
    print("done filter")
    return newTopics

if __name__ == '__main__':

    # Load dataset and wmodels
    dataset = pt.get_dataset("trec-deep-learning-passages")
    index = pt.IndexFactory.of("./passage_index_correct/data.properties")
    BM25_br = pt.apply.query(_remove_stops) >> pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % 1000
    TF_IDF = pt.BatchRetrieve(index, controls={"wmodel": "TF_IDF"}, verbose=True)
    PL2 = pt.BatchRetrieve(index, controls={"wmodel": "PL2"}, verbose=True)
    # BB2 = pt.BatchRetrieve(index, controls={"wmodel": "BB2"}, verbose=True)
    DirichletLM = pt.BatchRetrieve(index, controls={"wmodel": "DirichletLM"}, verbose=True)
    Dl = pt.BatchRetrieve(index, controls={"wmodel": "Dl"}, verbose=True)


    # Create datasets
    # validation_topics = dataset.get_topics(dev)
    # validation_qrels = dataset.get_qrels(dev)
    training_topics = dataset.get_topics('train').sample(10000)
    training_qrels = dataset.get_qrels('train')
    test_topics = dataset.get_topics("test-2019")
    test_qrels = dataset.get_qrels("test-2019")

    # Filter topic
    filtered_test_topics = filter_topics(test_topics, test_qrels)
    filtered_training_topics = filter_topics(training_topics, training_qrels)

    # Make pipeline
    pipe = BM25_br >> (pt.transformer.IdentityTransformer() ** TF_IDF ** PL2 ** Dl ** DirichletLM)

    # Make and train random forest regressor
    rf = RandomForestRegressor(n_estimators=400, verbose=1)
    BaselineLTR = pipe >> pt.ltr.apply_learned_model(rf)
    BaselineLTR.fit(filtered_training_topics, training_qrels)

    # Save results
    results = pt.pipelines.Experiment([BaselineLTR], filtered_test_topics, test_qrels,
                                      eval_metrics=["ndcg_cut_30", "ndcg_cut_10", "recip_rank", "map"],
                                      perquery=True)
    results.to_csv('dfd.csv')