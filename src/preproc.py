###### library for preprocessing tweets

from happierfuntokenizing import Tokenizer

class Preprocessor:
    
    def __init__(self,delimiter=" "):
        # delimiter to split tweets into tokens
        self.DELIM=delimiter
        self.tokenizer=Tokenizer()

    def tokenize(self,tweet):
        return " ".join(self.tokenizer.tokenize(tweet))

    def replace_user_tags(self,tweet,remove=False):
        """
        Replace mentions to usernames with "@USER"
        if remove=True removes the user mentions
        
        >>> p=Preprocessor()
        >>> p.replace_user_tags("@maya yes this is cool1@ did b@ @augyyz")
        '@USER yes this is cool1@ did b@ @USER'
        >>> p.replace_user_tags("@maya yes this is cool1@ did b@ @augyyz",remove=True)
        'yes this is cool1@ did b@'
        """
        if remove:
            return self.DELIM.join([w for w in tweet.split(self.DELIM) if not w.startswith("@")])
        else:
            return self.DELIM.join(["@USER" if w.startswith("@") else w for w in tweet.split(self.DELIM)])

    def replace_urls(self,tweet,remove=False):
        """
        Replace urls with @URL
        if remove=True removes them
    
        >>> p=Preprocessor()
        >>> p.replace_urls("@maya yes this is cool1@ did b@ @augyyz http://www.bitly")
        '@maya yes this is cool1@ did b@ @augyyz @URL'
        >>> p.replace_urls("@maya yes this is cool1@ did b@ @augyyz http://www.bitly",remove=True)
        '@maya yes this is cool1@ did b@ @augyyz'
        
        """
        if remove:
            return self.DELIM.join([w for w in tweet.split(self.DELIM) if not w.startswith("http")])
        else:
            return self.DELIM.join(["@URL" if w.startswith("http") else w for w in tweet.split(self.DELIM)])

    def replace_hashtags(self,tweet,remove=False):
        """
        Replace hashtags with @HASHTAG
        if remove=True removes them (any number of # at token start)

        >>> p=Preprocessor()
        >>> p.replace_hashtags("yes #cool we are in #miami ###yes")
        'yes @HASHTAG we are in @HASHTAG @HASHTAG'
        >>> p.replace_hashtags("yes #cool we# are in #miami ###yes",remove=True)
        'yes we# are in'
        >>> p.replace_hashtags("yes #cool we# are in #miami ###yes bar . #wishiwere in italy .")
        'yes @HASHTAG we# are in @HASHTAG @HASHTAG bar . @HASHTAG in italy .'
        """
        if remove:
            return self.DELIM.join([w for w in tweet.split(self.DELIM) if not w.startswith("#")])
        else:
            return self.DELIM.join(["@HASHTAG" if w.startswith("#") else w for w in tweet.split(self.DELIM)])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
