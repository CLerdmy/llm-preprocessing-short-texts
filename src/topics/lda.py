from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.models import LdaModel
 
from src.topics.base import MAX_DF, MIN_DF, TOP_N, TopicModel
 
K = 20
SEED = 42
PASSES = 10
ITERATIONS = 400
ALPHA = "symmetric"
ETA = None


class LDA(TopicModel):
    name = "lda"
 
    def __init__(self, k=K, seed=SEED, passes=PASSES, iterations=ITERATIONS, alpha=ALPHA, eta=ETA):
        super().__init__(k, seed, passes=passes, iterations=iterations, alpha=alpha, eta=eta)
 
    def _fit(self, docs, texts, embeddings):
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=MIN_DF, no_above=MAX_DF, keep_n=None)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
 
        model = LdaModel(corpus, num_topics=self.k, id2word=dictionary, random_state=self.seed,
                         passes=self.params["passes"], iterations=self.params["iterations"],
                         alpha=self.params["alpha"], eta=self.params["eta"], minimum_probability=0.0)
 
        topics = [[(w, float(p)) for w, p in model.show_topic(i, topn=TOP_N)] for i in range(self.k)]
        doc_topic = corpus2dense(model[corpus], num_terms=self.k).T
        self.model = model
        vocab = [dictionary[i] for i in range(len(dictionary))]
        return topics, doc_topic, model.get_topics(), vocab