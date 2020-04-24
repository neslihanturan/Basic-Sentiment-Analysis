import re, ssl
from collections import Counter
import snowballstemmer
from nltk import download
from nltk.corpus import stopwords
import jpype
import os

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
            print(1)
            primary_pos = best_word_analysis.getPos()
            #print(word_analysis.getInput(),best_lemma,primary_pos);
            tagged_word_tuple = (word_analysis.getInput(),best_lemma,primary_pos.getStringForm())
            pos_tagged_sentence.append(tagged_word_tuple)
            print(tagged_word_tuple)
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
