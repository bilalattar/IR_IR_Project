import pandas as pd
import matplotlib.pyplot as plt
import pyterrier as pt

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
    # name of the csv file to load from
    name = 'plot_files/bm25, tfidf, pl2, dl.csv'

    dataset = pt.get_dataset("trec-deep-learning-passages")

    test_topics = dataset.get_topics("test-2019")
    test_qrels = dataset.get_qrels("test-2019")

    filtered_test_topics = filter_topics(test_topics, test_qrels)

    results = pd.read_csv('')
    filtered_test_topics.astype({'qid': 'int64'}).dtypes
    filtered_test_topics["qid"] = pd.to_numeric(filtered_test_topics["qid"])
    results = (filtered_test_topics.set_index('qid').join(results.set_index('qid'))).sort_values(by='value')
    results = results[results['measure'] == 'map']
    ax = results.plot.barh(x='query', y='value', rot=0, figsize=(15, 20), color='orange')
    plt.ylabel('query')
    plt.xlabel('NDCG@30')
    plt.grid(axis='x')
    plt.locator_params(axis='x', nbins=20)
    mean = results["value"].mean()
    ax.axvline(mean)
    plt.xlim([0, 1])
    plt.show()