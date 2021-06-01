from functools import reduce
import re

import spacy
from nltk.tokenize import sent_tokenize
Spacynlp = spacy.load("en_core_web_sm")

# import rpy2
# import os
#
# os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
#
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
#
# # for row in rpy2.situation.iter_info():
# #     print(row)
#
# rangeBuilder = importr("rangeBuilder")

def extract_affiliation(affiliation):
    list = affiliation.split(",")
    return list

def extract_plainText_Spacy(Spacynlp, plainText):
    ### ORG, GPE, LOC
    locations = []
    entities = Spacynlp(plainText)
    for ent in entities.ents:
        if ent.label_ == "GPE" and delete_Spacy(ent.text, ent.label_):
        # if (ent.label_ == "ORG" or ent.label_ == "GPE" or ent.label_ == "LOC") and delete_Spacy(ent.text, ent.label_): # and delete_Spacy(ent.text)
            locations.append(ent.text)
            # print("test", ent.text, ent.label_)
    return locations

def delete_Spacy(ent, label):
    if ent.islower():
        return False
    if ent.isupper() and label != "GPE":
        return False
    # if "/" in ent or "=" in ent or "-" or "(" in ent:
    #     return False
    if "influenza" in ent or "Influenza" in ent:
        return False
    if len(ent.split()) > 1:
        for e in ent.split():
            if not e.istitle() and e != "the" and e != "of":
                return False
    if re.findall("H\dN\d", ent)!=[]:
        return False
    return True

def combination(dict1, dict2):
    for i, j in dict2.items():
        if i in dict1.keys():
            dict1[i] += j
        else:
            dict1.update({f"{i}": dict2[i]})
    return dict1

# def possible_location(title_locations, abstract_locations, affiliations_locations, affiliation):
#     count1 = {}
#     count2 = {}
#     count3 = {}
#     possible = []
#     # print(title_locations)
#     # print(affiliation)
#     if title_locations == [] and abstract_locations == [] and affiliations_locations == []:
#         return None
#     if affiliation == "" and title_locations != []:
#         return title_locations
#     if affiliation == "" and abstract_locations != []:
#         return abstract_locations
#     if title_locations != [] and affiliation != []:
#         for t in title_locations:
#             count1[t.replace(".", "").strip("")] = affiliation.count(t.replace(".", "").strip(""))
#     print("count1", count1)
#     if abstract_locations != [] and affiliation != "":
#         for ab in abstract_locations:
#             count2[ab.replace(".", "").strip("")] = affiliation.count(ab.replace(".", "").strip(""))
#     print("count2", count2)
#     if affiliations_locations != [] and affiliation != "":
#         for af in affiliations_locations:
#             count3[af.replace(".", "").strip("")] = affiliation.count(af.replace(".", "").strip(""))
#     print("count3", count3)
#     count_sum = reduce(combination, [count1, count2, count3])
#     print("count_sum", count_sum)
#     max_value = max(count_sum.values())
#     print("max_value", max_value)
#     for keys, values in count_sum.items():
#         if values == max_value and max_value != 0:
#             possible.append(keys)
#     # print(count_sum["University"])
#     return possible

# def possible_location(title_locations, abstract_locations, affiliations_locations, affiliation):
#     dic1 = {}
#     if title_locations != []:
#         for t in title_locations:
#             dic1[t.strip().replace(".","")] = dic1.setdefault(t,0) + 0.5
#     if abstract_locations != []:
#         for a in abstract_locations:
#             dic1[a.strip().replace(".","")] = dic1.setdefault(a,0) + 0.6
#     if affiliations_locations != []:
#         for af in affiliations_locations:
#             dic1[af.strip().replace(".","")] = dic1.setdefault(af, 0) + 0.4
#     max = 0
#     max_key = []
#     for key, value in dic1.items():
#         if value > max:
#             max = value
#             max_key.append(key)
#     print(dic1)
#     return max_key

def possible_location(title_locations, abstract_locations, affiliations_locations, title_weight, abstract_weight, affilation_weight):
    dic1 = {}
    if title_locations != []:
        for t in title_locations:
            # t1 = rangeBuilder.standardizeCountry(t)[0]
            # if t1 != "":
            #     t = t1
            # dic1[t] = dic1.setdefault(t, 0) + title_weight
            dic1[t.strip().replace(".","")] = dic1.setdefault(t,0) + title_weight
    if abstract_locations != []:
        for a in abstract_locations:
            # a1 = rangeBuilder.standardizeCountry(a)[0]
            # if a1 != "":
            #     a = a1
            # dic1[a] = dic1.setdefault(a, 0) + abstract_weight
            dic1[a.strip().replace(".","")] = dic1.setdefault(a,0) + abstract_weight
    if affiliations_locations != []:
        for af in affiliations_locations:
            # af1 = rangeBuilder.standardizeCountry(af)[0]
            # if af1 != "":
            #     af = af1
            # dic1[af] = dic1.setdefault(af, 0) + affilation_weight
            dic1[af.strip().replace(".","")] = dic1.setdefault(af, 0) + affilation_weight
    max = 0
    max_key = []
    for key, value in dic1.items():
        if value > max:
            max = value
            max_key.append(key)
    return max_key

def extract_locations_spacy(Spacynlp, value, title_weight, abstract_weight, affilation_weight):
    articleTitle, abstractTexts, affiliations = value
    if articleTitle != "":
        title_locations = extract_plainText_Spacy(Spacynlp, articleTitle)
    else:
        title_locations = []

    if abstractTexts != "":
        abstract_locations = []
        sentenses = sent_tokenize(abstractTexts)
        for sentense in sentenses:
            abstract_locations += extract_plainText_Spacy(Spacynlp, sentense)
    else:
        abstract_locations = []

    if affiliations != "":
        # affiliations_locations = extract_affiliation(affiliations)
        affiliations_locations = extract_plainText_Spacy(Spacynlp, affiliations)
    else:
        affiliations_locations = []
    # print("title_locations", title_locations)
    # print("abstract_locations", abstract_locations)
    # print("affiliations_locations", affiliations_locations)

    possible = possible_location(title_locations, abstract_locations, affiliations_locations, title_weight, abstract_weight, affilation_weight)
    # possible = possible_location(title_locations, abstract_locations, affiliations_locations, affiliations, title_weight, abstract_weight, affilation_weights)
     #
     # for tw in range (1,10):
     #     for aw in range(1,10):
     #         for afw in range(1,10):
     #             if (tw+aw+afw == 10):
     #                 possible = possible_location(title_locations, abstract_locations, affiliations_locations,
     #                                              affiliations, tw/10.0, aw / 10.0, afw / 10.0);

    return possible, title_locations, abstract_locations, affiliations_locations

if __name__ == "__main__":
    data = "Determinants of public phobia about infectious diseases in South Korea: effect of health communication and gender difference."
    a = extract_plainText_Spacy(Spacynlp, data)
    print(a)