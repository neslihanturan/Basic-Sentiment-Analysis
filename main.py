import jpype
from zemberek_python import main_libs as lib

zemberek_api =  lib.zemberek_api(libjvmpath="/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so", zemberekJarpath="./zemberek_python/zemberek-full.jar")
turkishTokenizer = zemberek_api.getTurkishTokenizer()
turkishPOSTagger = zemberek_api.getTurkishPOSTagger()

corpus = "merhaba bu bir python zemberek denemesidir. bu denemeden garip yazılar. Gizli tanik ifade verdi"

sentences = lib.TokenizerTool(turkishTokenizer).tokenize(corpus)
for sentence in sentences:
    print (sentence)

pos_tagged_sentences = []
for sentence in sentences:
    if not sentence:
        continue
    sentence_analysis = lib.POSTaggerTool(turkishPOSTagger).analyze_and_disambiguate(" ".join(sentence))
    pos_tagged_sentence = lib.POSTaggerTool(sentence_analysis).pos_tag(sentence_analysis)
    pos_tagged_sentences.append(pos_tagged_sentence)

print (pos_tagged_sentences)


##print(a)
