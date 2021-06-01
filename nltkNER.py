from nltk import word_tokenize, pos_tag, ne_chunk, Tree, sent_tokenize
import nltk
from spacyNER import extract_affiliation, possible_location

def extract_plainText_nltk(plainText):
    tokenized = nltk.word_tokenize(plainText)
    pos_tag = nltk.pos_tag(tokenized)
    chunked = nltk.ne_chunk(pos_tag)
    continuous_chunk = []
    current_chunk = []
    # print(chunked)
    for subtree in chunked:
        # print("0")
        if type(subtree) == Tree and subtree.label() == "GPE":
        # if type(subtree) == Tree and (subtree.label() == "GPE" or subtree.label() == "ORGANIZATION" or subtree.label() == "LOCATION"):
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            # print("subtree.leaves()", subtree.leaves())
            # print("1")
            # print(subtree.leaves, subtree.label)

        # elif current_chunk:
        #     name_entity = " ".join(current_chunk)
        #     # print("2")
        #     if name_entity not in continuous_chunk:
        #         continuous_chunk.append(name_entity)
        #         current_chunk = []
        # else:
            # print("00")
            # continue
    return current_chunk

def extract_locations_nltk(value, title_weight, abstract_weight, affilation_weight):
    articleTitle, abstractTexts, affiliations = value
    if articleTitle != "":
        title_locations = extract_plainText_nltk(articleTitle)
    else:
        title_locations = []

    if abstractTexts != "":
        abstract_locations = []
        sentenses = sent_tokenize(abstractTexts)
        for sentense in sentenses:
            abstract_locations += extract_plainText_nltk(sentense)
    else:
        abstract_locations = []

    if affiliations != "":
        # affiliations_locations = extract_affiliation(affiliations)
        affiliations_locations = extract_plainText_nltk(affiliations)
    else:
        affiliations_locations = []
    # print("title_locations", title_locations)
    # print("abstract_locations", abstract_locations)
    # print("affiliations_locations", affiliations_locations)
    possible = possible_location(title_locations, abstract_locations, affiliations_locations, title_weight, abstract_weight, affilation_weight)
    return possible, title_locations, abstract_locations, affiliations_locations

if __name__ == "__main__":
    data = "Determinants of public phobia about infectious diseases in South Korea: effect of health communication and gender difference."
    a = extract_plainText_nltk(data)
    print(a)