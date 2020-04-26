import re, ssl
from collections import Counter
import snowballstemmer
from nltk import download
from nltk.corpus import stopwords
import jpype
import os
import yaml

## KULLANIMI ##
###############
# 1) Örnek corpusun cümlelerini parcalara ayırır fonksiyonu ile parçalanır kelime-kelime haline getirilir
#    ve gereksiz kelimeler atılır
# 2) Parçalara ayrılmış olan haber cümlenin öğelerine ayrılır
# 3) Cümlenin öğelerine ayrılmış olan kelimelerin kökleri bulunur bir listeye konulur


def _find_libjvm():
    java_home = os.environ.get('JAVA_HOME', None)
    jre_home = os.environ.get('JRE_HOME', None)
    if java_home is not None:
        return _find_libjvm_in_java_home(java_home)
    elif jre_home is not None:
        return _find_libjvm_in_jre_home(jre_home)
    else:
        raise ValueError('Either set one of JAVA_HOME and JRE_HOME environment variables, or pass a path value to libjvmpath argument.')

def _find_libjvm_in_java_home(path):
    if os.name == 'nt': # windows
        path = os.path.join(path, 'jre', 'bin', 'server', 'jvm.dll')
    else:
        path = os.path.join(path, 'jre', 'lib', 'amd64', 'server', 'libjvm.so')
    if os.path.exists(path):
        return path
    else:
        raise IOError('Could not find libjvm in {}. Please make sure that you set JAVA_HOME environment variable correctly, or pass a value to libjvmpath argument'.format(path))

def _find_libjvm_in_jre_home(path):
    if os.name == 'nt': # windows
        path = os.path.join(path, 'bin', 'server', 'jvm.dll')
    else:
        path = os.path.join(path, 'lib', 'amd64', 'server', 'libjvm.so')
    if os.path.exists(path):
        return path
    else:
        raise IOError('Could not find libjvm in {}. Please make sure that you set JRE_HOME environment variable correctly, or pass a value to libjvmpath argument'.format(path))

class zemberek_api:
    def __init__(self,libjvmpath=None,zemberekJarpath=os.path.join(os.path.dirname(__file__), 'zemberek-full.jar')):
        if libjvmpath is not None:
            self.libjvmpath = libjvmpath
        else:
            self.libjvmpath = _find_libjvm()
        self.zemberekJarpath = zemberekJarpath
        jpype.startJVM(self.libjvmpath, "-Djava.class.path=" + self.zemberekJarpath, "-ea")

    def getTurkishTokenizer(self):
            Token = jpype.JClass("zemberek.tokenization.Token")
            turkishTokenizer = jpype.JClass("zemberek.tokenization.TurkishTokenizer").builder().ignoreTypes(Token.Type.Punctuation, Token.Type.NewLine, Token.Type.SpaceTab).build();
            return turkishTokenizer

    def getTurkishPOSTagger(self):
            turkishPOSTagger = jpype.JClass("zemberek.morphology.TurkishMorphology").createWithDefaults();
            return turkishPOSTagger

class TokenizerTool:
    def __init__(self,tokenizer):
        self.turkishTokenizer = tokenizer

    def tokenize(self,text):
        """
        input format: a paragraph of text
        output format: a list of sentences as lists of words.
            e.g.: [['Bu', 'bir', 'cümle'], ['Bu', 'da', 'diğeri']]
        """
        sentences = text.split('.')
        tokenized_sentences = [self.turkishTokenizer.tokenizeToStrings(sentence) for sentence in sentences]
        return tokenized_sentences

class POSTaggerTool:
    def __init__(self, tagger):
        self.turkishPOSTagger = tagger

    def analyze_and_disambiguate(self,sentence):
        return self.turkishPOSTagger.analyzeAndDisambiguate(sentence)

    def pos_tag(self, sentence_analysis):
        """
        input format: list of words
        output format: a word form, a word lemma, and a list of associated tags
        """
        pos_tagged_sentence = []
        for sentence_word_analysis in sentence_analysis:
            word_analysis = sentence_word_analysis.getWordAnalysis()
            best_word_analysis = sentence_word_analysis.getBestAnalysis()
            best_lemma = self.get_best_lemma(best_word_analysis)
            primary_pos = best_word_analysis.getPos()
            tagged_word_tuple = (word_analysis.getInput(),best_lemma,[primary_pos.getStringForm()])
            pos_tagged_sentence.append(tagged_word_tuple)
        return pos_tagged_sentence

    def get_best_lemma(self, best):
        return best.getLemmas()[0]

class nltk_download:
    def __init__(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        download()

class DictionaryTagger(object):
    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        for file in files:
            file.close()
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

class Reviewer(object):

    def value_of(self, sentiment):
         if sentiment == 'positive': return 1
         if sentiment == 'negative': return -1
         return 0

    def sentiment_score(self, dict_tagged_sentences):
         return sum ([self.value_of(tag) for sentence in dict_tagged_sentences for token in sentence for tag in token[2]])
