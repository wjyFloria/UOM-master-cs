from spacyNER import extract_affiliation, possible_location
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
stanfordNERnlp = StanfordCoreNLP(r'/Users/yuan/Desktop/stanford-corenlp-4.2.0', lang="en")

def extract_plainText_StanfordNER(stanfordNERnlp, plaintext):
    IBO = stanfordNERnlp.ner(plaintext)
    # print("IBO",IBO)
    ### ORGANIZATION, CITY, COUNTRY, LOCATION
    i = 0
    current = []
    while i < len(IBO):
        word, label = IBO[i]
        if label == "COUNTRY":
        # if label == "ORGANIZATION" or label == "CITY" or label == "COUNTRY" or label == "LOCATION":
            continuous = "".join(word)
            j = i + 1
            # print(word, label)
            while j < len(IBO):
                word1, label1 = IBO[j]
                if label1 == label:
                    continuous = continuous + " " + word1
                    # print(continuous, label)
                    j += 1
                    continue
                else:
                    i = j
                    current.append(continuous)
                    break
        else:
            i += 1
    return current

def extract_organization_StanfordNER(plaintext):
    IBO = stanfordNERnlp.ner(plaintext)
    # print("IBO",IBO)
    ### ORGANIZATION, CITY, COUNTRY, LOCATION
    i = 0
    current = []
    while i < len(IBO):
        word, label = IBO[i]
        if label == "ORGANIZATION":
        # if label == "ORGANIZATION" or label == "CITY" or label == "COUNTRY" or label == "LOCATION":
            continuous = "".join(word)
            j = i + 1
            # print(word, label)
            while j < len(IBO):
                word1, label1 = IBO[j]
                if label1 == label:
                    continuous = continuous + " " + word1
                    # print(continuous, label)
                    j += 1
                    continue
                else:
                    i = j
                    current.append(continuous)
                    break
        else:
            i += 1
    return current

def extract_locations_stanfordNER(stanfordNERnlp, value, title_weight, abstract_weight, affilation_weight):
    articleTitle, abstractTexts, affiliations = value
    if articleTitle != "":
        title_locations = extract_plainText_StanfordNER(stanfordNERnlp, articleTitle)
    else:
        title_locations = []

    if abstractTexts != "":
        abstract_locations = []
        sentenses = sent_tokenize(abstractTexts.replace("%","percent"))
        for sentense in sentenses:
            abstract_locations += extract_plainText_StanfordNER(stanfordNERnlp, sentense)
    else:
        abstract_locations = []

    if affiliations != "":
        # affiliations_locations = extract_affiliation(affiliations)
        affiliations_locations = extract_plainText_StanfordNER(stanfordNERnlp, affiliations)
    else:
        affiliations_locations = []
    # print("title_locations", title_locations)
    # print("abstract_locations", abstract_locations)
    # print("affiliations_locations", affiliations_locations)
    # print("affiliations",affiliations)
    possible = possible_location(title_locations, abstract_locations, affiliations_locations, title_weight, abstract_weight, affilation_weight)
    return possible, title_locations, abstract_locations, affiliations_locations

if __name__ == "__main__":
    # data = "Determinants of public phobia about infectious diseases in the ROC: effect of health communication and gender difference."
    # data = "Sequence analysis of hemagglutinin and nucleoprotein genes of measles viruses isolated in Korea during the 2000 epidemic."
    # data = "Time-dependent receiver operating characteristic (ROC) and Kaplan–Meier (KM) survival analysis were used to assess its prognostic power."
    # data = "Further, ROC analysis of each gene was performed and the results showed that the sensitivity and specificity of single parameter was poorer than that of five-gene prognostic signature, which suggested that the predictive power of multi variables would perform much better."
    data = "gene expression analysis of Mx mRNA transcripts"





    a = extract_plainText_StanfordNER(stanfordNERnlp, data)
    print(a)