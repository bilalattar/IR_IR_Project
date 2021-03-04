import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal


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
    name = 'plot_files/bm25, tfidf, pl2, dl, dirichletlm.csv'

    test_topics = pd.read_csv('test_topics.csv')
    test_qrels = pd.read_csv('test_qrels.csv')

    filtered_test_topics = filter_topics(test_topics, test_qrels)

    results = pd.read_csv(name)
    filtered_test_topics.astype({'qid': 'int64'}).dtypes
    filtered_test_topics["qid"] = pd.to_numeric(filtered_test_topics["qid"])
    results = (filtered_test_topics.set_index('qid').join(results.set_index('qid'))).sort_values(by='value')
    results = results[results['measure'] == 'map']
    my_colors = [(x / len(filtered_test_topics), x / (2 * len(filtered_test_topics)), 0.75) for x in
                 range(len(filtered_test_topics))]
    ax = results.plot.barh(x='query', y='value', rot=0, figsize=(15, 20), color=my_colors)
    plt.ylabel('query')
    plt.xlabel('NDCG@30')
    plt.grid(axis='x')
    plt.locator_params(axis='x', nbins=20)
    mean = results["value"].mean()
    ax.axvline(mean)
    for p in ax.patches:
        ax.annotate(round(Decimal(str(p.get_width())), 3), (p.get_width()*1.005, p.get_y()))
    plt.xlim([0, 1])
    plt.show()
