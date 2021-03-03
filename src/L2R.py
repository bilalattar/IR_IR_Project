import numpy as np
import pandas as pd
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

  dataset = pt.get_dataset("trec-deep-learning-passages")
  index = pt.IndexFactory.of("./passage_index_correct/data.properties")
  # BM25_br = pt.BatchRetrieve(index, wmodel="BM25", verbose=True) % 1000

  dev = "dev"

  validation_topics = dataset.get_topics(dev)
  validation_qrels = dataset.get_qrels(dev)
  training_topics = dataset.get_topics('train').head(10000)
  training_qrels = dataset.get_qrels('train')

  test_topics = dataset.get_topics("test-2019")
  test_qrels = dataset.get_qrels("test-2019")

  filtered_test_topics = filter_topics(test_topics, test_qrels)
  filtered_training_topics = filter_topics(training_topics, training_qrels).head(1000)
