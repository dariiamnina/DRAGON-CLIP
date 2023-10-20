import clip
import pickle 

with open('data/test/test_CLIP_list_of_textprompts.pickle', 'rb') as fi:
    list_of_textprompts = pickle.load(fi)

tokens = clip.tokenize(list_of_textprompts, truncate=True)
with open('data/test/test_CLIP_tokens.pickle', 'wb') as fo:
    pickle.dump(tokens, fo)