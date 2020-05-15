from time import time
import string
import random

from sentence_transformers import SentenceTransformer

def randomString(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(random.randint(0, size)))

def test_sbert():
    print("Loading SBert...")
    t1 = time()
    sbert_model = SentenceTransformer("./models/distiluse-base-multilingual-cased")
    t2 = time()

    print("Loaded model in", t2 - t1, "s")

    #texts = ["TTThis player needs tp be reported lolz."] * 100
    texts = [randomString(100) for _ in range(1000)]

    test_arr = [texts[0], "a", "b", "c"]
    enc = sbert_model.encode(test_arr)
    print("test encode:", [e[:5] for e in enc])
    print("token encode:", [sbert_model.tokenize(s) for s in test_arr])

    print("Encoding", texts[0], "...")
    for b in [2**i for i in range(10)]:
        t = 0
        n = 10
        for i in range(n):
            t1 = time()
            output = sbert_model.encode(texts, b)
            t2 = time()
            t += t2 - t1
        t /= n
        print("Elapsed time", t, "s for batch size", b)
    print(output[0][:5])

    return output


test_sbert()
