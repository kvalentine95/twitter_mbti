import bisect
from collections import Counter, defaultdict
import sys
import nltk

class Featurizer:

    PREFIX_WORD_NGRAM="N:"
    PREFIX_CHAR_NGRAM="C:"
    PREFIX_FRAMENET="FN:"

    def __init__(self,binary=True,lowercase=False,remove_stopwords=False):
        # delimiter to split tweets into tokens
        self.DELIM=" "
        self.d = {} if binary else defaultdict(int) #dictionary that holds features
        self.lowercase=lowercase
        self.binary=binary
        self.remove_stopwords=remove_stopwords
        self.wiktionary=None

    def init_wiktionary(self,wiktionaryfile):
        """
        initialize wiktionary: token to taglist
        >>> f=Featurizer()
        >>> f.init_wiktionary("wiktionary/en.tags.li")
        >>> f.wiktionary['awesome']
        ['ADJ']
        """
        self.wiktionary=defaultdict(list)
        wikfile=[x.strip() for x in open(wiktionaryfile).readlines()]
        for l in wikfile:
            fields=l.split("\t")
            token,tag=fields[1],fields[2]
            self.wiktionary[token].append(tag)

    def get_gender(self,gender):
        self.d["gender"]=gender

    def add_feature(self,key,value):
        self.d[key]=value


    def get_meta_features(self, line):
        breakpoints = {
            'followers_count':[10, 50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000],
            'statuses_count':[50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000],
            'favorites_count':[5, 10, 50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000],
            'listed_count':[5, 10, 50, 100, 200, 500, 1000, 5000, 10000]
        }

        feature_input = [tuple(feature.split('=')) for feature in line.split('\t')]

        for (feature_name, value) in feature_input:
            if feature_name in breakpoints:
                number = bisect.bisect(breakpoints[feature_name], int(value))
                if number < len(breakpoints[feature_name]):
                    fname = "%s<%s" % (feature_name, breakpoints[feature_name][number])
                else:
                    fname = "%s>%s" % (feature_name, breakpoints[feature_name][-1])
                self.d[fname] = 1
            else:
                self.d[feature_name] = '%s' % (value)


    def word_ngrams(self,tweet,ngram="1-2-3"):
        """
        extracts word n-grams

        >>> f=Featurizer()
        >>> f.word_ngrams("this is a test",ngram="1-3")
        >>> f.getNumFeatures()
        6
        """
        if self.lowercase:
            tweet = tweet.lower()
        words=tweet.split(self.DELIM)
        if self.remove_stopwords:
            words = [w for w in words if w not in ENGLISH_STOP_WORDS]

        for n in ngram.split("-"):
            for gram in nltk.ngrams(words, int(n)):
                if self.binary:
                    self.d[gram] = 1 #binary
                else:
                    self.d[gram] += 1

                # # ngram features (G)
                # for i in range(len(words)) :
                #     # up to N n-grams
                #         gram = self.PREFIX_WORD_NGRAM #"N:"
                #         N=int(n)
                #         for j in range(i,min(i+N, len(words))) :
                #             gram += words[j] +  " "
                #             if len(gram.split(" "))==N+1: #because of prefix
                #                 if self.binary:
                #                     self.d[gram]=1 #binary
                #                 else:
                #                     self.d[gram]=self.d.get(gram,0)+1

    def character_ngrams(self,tweet,ngram="1-2-3"):
        """
        extracts character n-grams

        >>> f=Featurizer()
        >>> f.character_ngrams("yess !",ngram="1")
        >>> f.getNumFeatures()
        5
        >>> f=Featurizer()
        >>> f.character_ngrams("yess !",ngram="1-2-3")
        >>> f.getNumFeatures()
        14
        """
        words=tweet
        if self.lowercase:
            words=words.lower()
        if self.remove_stopwords:
            words = self.DELIM.join([w for w in words.split(self.DELIM) if w not in ENGLISH_STOP_WORDS])

        # ngram features (G)
        for i in range(len(words)) :
            # up to N n-grams
            for n in ngram.split("-"):
                gram = self.PREFIX_CHAR_NGRAM #"C:"
                N=int(n)
                for j in range(i,min(i+N, len(words))) :
                    gram += words[j]
                    if len(gram)==N+2: #because of prefix
                        if self.binary:
                            self.d[gram]=1 #binary
                        else:
                            self.d[gram]=self.d.get(gram,0)+1

    def framenet_frames_all(self,tweet):
        """
        count the total number of invoced frames in framenet
        for every token and every possible tag according to wiktionary, looks up frames and counts them
        
        >>> f=Featurizer()
        >>> f.init_wiktionary("wiktionary/en.tags.li")
        >>> f.framenet_frames_all("the little guy")
        >>> f.printFeatures()
        """
        from nltk.corpus import framenet as fn
        if not self.wiktionary:
            print>>sys.stderr, "call init_wiktionary before using framenet"
            raise Error("init_wiktionary needed before calling this method")
        words=tweet
        if self.lowercase:
            words=words.lower()
        if self.remove_stopwords:
            words = self.DELIM.join([w for w in words.split(self.DELIM) if w not in ENGLISH_STOP_WORDS])
        for token in words.split(self.DELIM):
            for tag in self.wiktionary[token]:
                token_tag=token+"."+tag[0].lower() #initial tag of UPOS
                print token_tag
                frames=fn.frames_by_lemma(r'(?i)\b{}\b'.format(token_tag))
                for frame in frames:
                    f=self.PREFIX_FRAMENET+frame['name']
                    self.d[f]=self.d.get(f,0)+1

    def printFeatures(self):
        for f in self.d.keys():
            print f, self.d[f]

    def getNumFeatures(self):
        return len(self.d.keys())

    def getDict(self):
        return self.d

# from sklearn
# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
