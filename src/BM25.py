import pyterrier as pt
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pt.init()
    dataset = pt.get_dataset("trec-deep-learning-passages")
    index = pt.IndexFactory.of("./passage_index_correct/data.properties")
    BM25_br = pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % 1000

    # Filter topic list
    topics = dataset.get_topics("test-2019")
    qrels = dataset.get_qrels("test-2019")
    data = []
    sum = 0
    for index, row in topics.iterrows():
        if row['qid'] in qrels['qid'].unique():
            sum += 1
            data.append([row['qid'], row['query']])
    newTopics = pd.DataFrame(data, columns=['qid', 'query'])

    # Run experiment for whole dataset
    # result = pt.Experiment(
    #     [BM25_br],
    #     newTopics,
    #     qrels,
    #     eval_metrics=["num_q", "recip_rank", "ndcg_cut_10", "map"])


    # Run experiment per query
    result = pt.Experiment(
        [BM25_br],
        newTopics,
        qrels,
        eval_metrics=["map"],
        perquery=True)


    # Plot horizontal bar graph
    result = (newTopics.set_index('qid').join(result.set_index('qid'))).sort_values(by='value')
    result.plot.barh(x='query', y='value', rot=0, figsize=(10,20))
    plt.ylabel('query')
    plt.xlabel('MAP')
    plt.show()

    # print(result.to_string())