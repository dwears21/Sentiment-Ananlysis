# Sentiment-Ananlysis
Groups twitter tweets in to neutral,positive,or negative 

PTDF = open("project_twitter_data.csv","r")
RDF = open("resulting_data.csv","w")

punctuation_chars = ["'", '"', ",", ".", "!", ":", ";", '#', '@']
# lists of words to use
positive_words = []
with open("positive_words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            positive_words.append(lin.strip())

negative_words = []
with open("negative_words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            negative_words.append(lin.strip())
            
def get_pos(pos_sentences):
    PosSentences = strip_punctuation(pos_sentences)
    PosLst= PosSentences.split()
    
    cnt=0
    
    for wrd in PosLst:
        for PosWord in positive_words:
            if wrd == PosWord:
                cnt+=1
    return cnt

negative_words = []
with open("negative_words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            negative_words.append(lin.strip())

            
def get_neg(pos_sentences):
    PosSentences = strip_punctuation(pos_sentences)
    PosLst = PosSentences.split()
    
    cnt=0
    
    for wrd in PosLst:
        for NegWrd in negative_words:
            if wrd == NegWrd:
                cnt+=1
    return cnt

    
def strip_punctuation(strx):
    for char in punctuation_chars:
        strx = strx.replace(char, "")
    return strx


def WriteOnData(RDF):
    RDF.write("Number of Retweets, Number of Replies, Positive Score, Negative Score, Net Score")
    RDF.write("\n")

    linesPTDF =  PTDF.readlines()
    NoHeader= linesPTDF.pop(0)
    for linePTD in linesPTDF:
        lineTD = linePTD.strip().split(',')
        RDF.write("{}, {}, {}, {}, {}".format(lineTD[1], lineTD[2], get_pos(lineTD[0]), get_neg(lineTD[0]), (get_pos(lineTD[0])-get_neg(lineTD[0]))))    
        RDF.write("\n")

        

WriteOnData(RDF)
PTDF.close()
RDF.close()

