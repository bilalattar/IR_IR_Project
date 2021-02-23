from pyserini.search import get_topics, SimpleSearcher, get_topics_with_reader

def run_all_queries(file, topics, searcher):
    with open(file, 'w') as runfile:
        cnt = 0
        print('Running {} queries in total'.format(len(topics)))
        for id in topics:
            query = topics[id]['title']
            hits = searcher.search(query, 1000)
            for i in range(0, len(hits)):
                _ = runfile.write('{} Q0 {} {} {:.6f} Anserini\n'.format(id, hits[i].docid, i+1, hits[i].score))
            cnt += 1
            if cnt % 100 == 0:
                print(f'{cnt} queries completed')

if __name__ == '__main__':
    file = open('collection.tsv')
    topics = get_topics_with_reader(reader_class='io.anserini.search.topicreader.TsvIntTopicReader', file='msmarco-test2019-queries.tsv')
    searcher = SimpleSearcher('index-msmarco-passage-20191117-0ed488/')
    run_all_queries('bm25-passage-baseline.txt', topics, searcher)


