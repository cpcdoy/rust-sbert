from time import time

from sentence_transformers import SentenceTransformer


def test_sbert():
    print("Loading SBert...")
    t1 = time()
    sbert_model = SentenceTransformer("./models/distiluse-base-multilingual-cased")
    t2 = time()

    print("Loaded model in", t2 - t1, "s")

    texts = ["TTThis player needs tp be reported lolz."] * 100

    print("Encoding", texts[0], "...")
    t = 0
    n = 10
    for i in range(n):
        t1 = time()
        output = sbert_model.encode(texts)
        t2 = time()
        t += t2 - t1
    t /= n

    print("Elapsed time", t, "s")
    print(output[0][:5])

    return output


test_sbert()
