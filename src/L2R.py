import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyterrier as pt
from sklearn.ensemble import RandomForestRegressor

if not pt.started():
    pt.init()


def filter_topics(t, q):
    data = []
    sum = 0
    for index, row in t.iterrows():
        if row['qid'] in q['qid'].unique():
            sum += 1
            data.append([row['qid'], row['query']])
    newTopics = pd.DataFrame(data, columns=['qid', 'query'])

    return newTopics


if __name__ == '__main__':
    dataset = pt.get_dataset("trec-deep-learning-passages")
    index = pt.IndexFactory.of("./passage_index_correct/data.properties")
    BM25_br = pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % 1000
    TF_IDF = pt.BatchRetrieve(index, controls={"wmodel": "TF_IDF"}, verbose=True)
    PL2 = pt.BatchRetrieve(index, controls={"wmodel": "PL2"}, verbose=True)
    # BB2 = pt.BatchRetrieve(index, controls={"wmodel": "BB2"}, verbose=True)
    DirichletLM = pt.BatchRetrieve(index, controls={"wmodel": "DirichletLM"}, verbose=True)
    Dl = pt.BatchRetrieve(index, controls={"wmodel": "Dl"}, verbose=True)

    dev = "dev"

    # validation_topics = dataset.get_topics(dev)
    # validation_qrels = dataset.get_qrels(dev)
    training_topics = dataset.get_topics('train').head(1000)
    training_qrels = dataset.get_qrels('train')

    test_topics = dataset.get_topics("test-2019")
    test_qrels = dataset.get_qrels("test-2019")
    # # # #
    filtered_test_topics = filter_topics(test_topics, test_qrels)
    filtered_training_topics = filter_topics(training_topics, training_qrels).head(1000)
    #
    # # print(filtered_training_topics)
    #
    pipe = BM25_br >> (pt.transformer.IdentityTransformer() ** TF_IDF ** PL2 ** Dl ** DirichletLM)

    print((pipe % 2).transform('what slows down the flow of blood'))
    print(type(pipe))
    rf = RandomForestRegressor(n_estimators=400, verbose=1)
    BaselineLTR = pipe >> pt.ltr.apply_learned_model(rf)
    BaselineLTR.fit(filtered_training_topics, training_qrels)
    results = pt.pipelines.Experiment([BaselineLTR], filtered_test_topics, test_qrels, eval_metrics=["ndcg_cut_30", "ndcg_cut_10", "recip_rank", "map"],
                                      perquery=True)
    results.to_csv('dfd.csv')

#old way of calculating
    # # pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:Tf"])
    # # rf = RandomForestRegressor(n_estimators=400, verbose=1)
    # # rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)
    # # rf_pipe.fit(training_topics, training_qrels)
    # # results = pt.pipelines.Experiment([rf_pipe], filtered_test_topics, test_qrels, eval_metrics=["ndcg_cut_30"],
    # #                                   perquery=True)
    # results.to_csv('dfd.csv')


    # results = pd.read_csv('plot_files/bm25, tfidf, pl2, dl.csv')
    # print(results.dtypes)
    # filtered_test_topics.astype({'qid': 'int64'}).dtypes
    # filtered_test_topics["qid"] = pd.to_numeric(filtered_test_topics["qid"])
    # print(filtered_test_topics.dtypes)
    # results = (filtered_test_topics.set_index('qid').join(results.set_index('qid'))).sort_values(by='value')
    # results = results[results['measure'] == 'map']
    # print(results['value'])
    # ax = results.plot.barh(x='query', y='value', rot=0, figsize=(15, 20), color='orange')
    # plt.ylabel('query')
    # plt.xlabel('NDCG@30')
    # plt.grid(axis='x')
    # plt.locator_params(axis='x', nbins=20)
    # mean = results["value"].mean()
    # ax.axvline(mean)
    # plt.xlim([0, 1])
    # plt.show()