# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


from collections import namedtuple
Concept_Ngram = namedtuple('Concept_Ngram', ['ngram','title','id','embeddings'])


class Conceptualizer:

    import csv
    import sys
    import nltk

    def __init__(self):
        self.model = None
        self.titles = None
        self.redirects = None
        self.all_titles = None
        self.vocabulary = None
        self.vector_size = 0

    def unicode_dict_reader(self,utf8_data, **kwargs):
        import numpy as np
        import os
        csv_reader = csv.DictReader(utf8_data, **kwargs)
        for row in csv_reader:
            yield {unicode(key): unicode(value, 'utf-8') for key, value in row.iteritems()}

    def get_extra(self, keys):
        single_keys = set()
        keys = keys.replace('\,','\t')
        for key in keys.split(','):
            single_keys.add(key.replace('\t',','))
        return single_keys

    def get_all_titles(self,orig_titles=True, lower=True):
        """
        return a map of all wikipedia titles and redirects existing in the model 
        as keys and article id as values
        """
        all_pairs = []
        self.all_titles = {}
        for i,j in self.titles.items():
            all_pairs.append((i,'id'+j+'di',i))
        for i,j in self.redirects.items():
            all_pairs.append((i,'id'+self.titles[j]+'di',j))
        for i,id,j in all_pairs:
            if self.model is None or id in self.model:
                if lower==True:
                    i = i.lower()
                if orig_titles==True:
                    oldval = self.all_titles.setdefault(i,(id,j))
                    if oldval!= (id,j):
                        #print('unexpected duplicate title ({0}) for orginal title ({1}) where old title ({2})'.format(i,j,oldval[1]))
                        pass
                else:
                    oldval = self.all_titles.setdefault(i,(id,))
                    if oldval!= (id,):
                        #print('unexpected duplicate title ({0}) for orginal title ({1})'.format(i,j))
                        pass

    def load_titles(self,inpath, to_keep=set(), load_redirects=True,
                    redirects_value='id', load_seealso=False, log=True):
        titles = {}
        redirects = {}
        seealso = {}
        if log==True:
            print("loading titles and redirects and seealso")
        if sys.version_info[0]==3:
            records = csv.DictReader(open(inpath))
        else:
            records = self.unicode_dict_reader(open(inpath))
        count = 0
        for pair in records:
            count = count + 1
            if len(to_keep)==0 or pair['title'] in to_keep:
                titles[pair['title']] = pair['id']
                if load_redirects==True:
                    extra = pair['redirect']
                    if len(extra)>0:
                        keys = self.get_extra(extra)
                        for key in keys:
                            redirects[key] = pair[redirects_value]
                if load_seealso==True:
                    extra = pair['seealso']
                    if len(extra)>0:
                        keys = self.get_extra(extra)
                        if len(keys)>0:
                            seealso[pair['title']] = keys
        if log==True:
            print("loaded "+str(count)+" titles")
        return (titles,redirects,seealso)

    def load(self, model_path=None, first=0, log_every=0):
        """
        load word2vec vocabulary vectors from pickle file
        """
        vectors_dic = {}
        import pickle
        # open input file
        with open(model_path, 'rb') as output:
            if log_every>0:
                print('start loading!')
            # read titles meta
            titles_num = pickle.load(output)
            if titles_num>0:
                self.titles = {}
            for _ in range(titles_num):
                title, id = pickle.load(output)
                self.titles[title] = id
            if log_every>0:
                print('loaded ({0}) titles'.format(titles_num))
            # read redirects meta
            redirects_num = pickle.load(output)
            if redirects_num>0:
                self.redirects = {}
            for _ in range(redirects_num):
                redirect, title = pickle.load(output)
                self.redirects[redirect] = title
            if log_every>0:
                print('loaded ({0}) redirects'.format(titles_num))
            # read vector size
            self.vector_size = pickle.load(output)
            # read number of vectors
            num = pickle.load(output)
            if log_every>0:
                print('loading ({0}) vectors'.format(num))
            for i in range(num):
                term, vec = pickle.load(output)
                vectors_dic[term] = vec
                if first>0 and i>=first-1:
                    break
                if log_every>0 and i>0 and i%log_every==0:
                    print('loaded ({0}) vectors'.format(i))
        if log_every>0:
            print('done loading!')
        self.model = vectors_dic
        self.get_all_titles()
        l = list(self.all_titles.keys())
        l.extend(list(self.model.keys()))
        self.vocabulary = set(l)

    def save(self, model=None, model_path=None, titles_meta=None, titles_path=None, outpath=None, vocab=None, concepts_only=False, concepts_prefix='id', concepts_postfix='di', protocol=2, log_every=0):
        """
        writes word2vec vocabulary vectors to pickle file
        if vocab is not none, only elements in it will be flushed
        """
        import gensim
        import pickle
        # open the model if necessary
        if model==None:
            model = gensim.models.Word2Vec.load(model_path)
            model.init_sims(replace=True)
        # load the titles if necessary
        titles = {}
        redirects = {}
        if titles_meta is None:
            if titles_path is not None:
                titles, redirects, _ = self.load_titles(titles_path, redirects_value='title')
        else:
            titles = titles_meta['titles']
            redirects = titles_meta['redirects']
        # open output file
        with open(outpath, 'wb') as output:
            if log_every>0:
                print('start saving!')
            if vocab is None:
                vocab = model.vocab
            # write titles meta
            print('dumping ({0}) titles'.format(len(titles)))
            pickle.dump(len(titles), output, protocol)
            for title, id in titles.items():
                pickle.dump((title,id), output, protocol)
            # write redirects meta
            print('dumping ({0}) redirects'.format(len(redirects)))
            pickle.dump(len(redirects), output, protocol)
            for redirect, title in redirects.items():
                pickle.dump((redirect,title), output, protocol)
            # write vector size
            pickle.dump(model.vector_size, output, protocol)
            # calculate number of vectors to be written
            num = 0
            for term in model.vocab:
                if term in vocab:
                    if concepts_only==False or (term[0:2]==concepts_prefix and term[-2:]==concepts_postfix):
                        num += 1
            # write number of vectors
            #num = 1011
            pickle.dump(num, output, protocol)
            # write vectors
            print('dumping ({0}) vectors'.format(num))
            count = 0
            for term in model.vocab:
                if term in vocab:
                    if concepts_only==False or (term[0:2]==concepts_prefix and term[-2:]==concepts_postfix):
                        pickle.dump((term,model[term]), output, protocol)
                        count += 1
                        #if count>num:
                        #    break
                        if log_every>0 and count>0 and count%log_every==0:
                            print('dumped ({0}) vectors'.format(count))
        if log_every>0:
            print('done saving!')

    def conceptualize(self, input_text='', ngram_range=(1,1),stop_words=set(nltk.corpus.stopwords.words('english')), token_pattern=r"(?u)\b\w\w+\b",lowercase=True):
        """
        given a text snippet, parse the text and generate ngrams of Wikipedia concepts 
        aggressively. tokens not resolved to concepts will be tokenized as well 
        embeddings for both raw and concept tokens will be returned along with meta data 
        about concepts such as article title and its id
        if token is not in the trained model, its embeddings will be <UNK> 
        TODO: resolve collisions where titles and redirects of different case map to 
        different articles when thier case is lowered: 
        e.g., Rail simulator --> Train simulator vs. Rail Simulator --> Rail Simulator 
        Also Toyota Runx --> Toyota Corolla vs. Toyota RunX --> Toyota Corolla (E120)
        """
        from nltk import ngrams
        import re
        import numpy as np
        if lowercase==True:
            input_text = input_text.lower()
        input_text = ' '.join(re.findall(token_pattern, input_text))
        all_vecs = []
        if stop_words is not None:
            stop_words_set = set(stop_words)
            # replace each stopword with space
            new_input_text = []
            for token in input_text.split(' '):
                if token in stop_words_set:
                    new_input_text.append(' ')
                else:
                    new_input_text.append(token)
            input_text = ' '.join(new_input_text)
        input_text = ' ' + input_text + ' '    # replace('or','<tok>1<tok>') would result in more --> m<tok>1<tok>
        input_text_tokenized = input_text
        for ngram_len in range(ngram_range[1],ngram_range[0]-1,-1):
            ngram_list = ngrams(input_text.split(' '), ngram_len)
            for ngram_token in ngram_list:
                if '' in ngram_token: # this ngram is not correct, original string had a stopword
                    continue
                ngram_token = ' '.join(ngram_token)
                title = self.all_titles.get(ngram_token) # check if a title, redirect
                if title is None: # not a title, redirect, may be a word
                    title = ngram_token
                    org_title = ''
                    id = ''
                else:
                    org_title = title[1]
                    title = title[0] # retrieve original title
                    id = title.replace('id','').replace('di','')
                vec = self.model.get(title,'<UNK>')
                if isinstance(vec,np.ndarray) or (vec=='<UNK>' and ngram_len==1): # append vector or <unk> if unigrams
                    indx = ' '+'<TOK>' + str(len(all_vecs)) + '<TOK>'+' '
                    all_vecs.append(Concept_Ngram(ngram_token,org_title,id,vec))
                    input_text_tokenized = input_text_tokenized.replace(' '+ngram_token+' ',indx)
        final_tokens = [int(i) for i in re.findall(r"<TOK>([0-9]+)<TOK>",input_text_tokenized)]
        tokens_num = len(final_tokens)
        out_vecs = [0]*tokens_num
        for token_indx in range(tokens_num):
            out_vecs[token_indx] = all_vecs[final_tokens[token_indx]]
        return out_vecs
