import csv
import math
from xml.dom import minidom

import Bio
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from Bio import Entrez, Medline, SeqIO
Entrez.email = 'jiayuanw1@student.unimelb.edu.au'
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from urllib.request import urlopen
from stanfordcorenlp import StanfordCoreNLP
from stanfordNER import extract_plainText_StanfordNER, extract_locations_stanfordNER, extract_organization_StanfordNER
from spacyNER import extract_locations_spacy
from nltkNER import extract_locations_nltk
import spacy
import time
import pandas as pd
import json
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

stanfordNERnlp = StanfordCoreNLP(r'/Users/yuan/Desktop/stanford-corenlp-4.2.0', lang="en")
Spacynlp = spacy.load("en_core_web_sm")

import rpy2
import os

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# for row in rpy2.situation.iter_info():
#     print(row)

rangeBuilder = importr("rangeBuilder")

import MySQLdb

import requests
import eventlet
import time
# get_geocoder_for_service("nominatim")
import re
from geotext import GeoText

from nltk.tokenize import sent_tokenize

import Levenshtein

######## use GeoPy
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.geocoders import get_geocoder_for_service
geolocator = Nominatim(user_agent="project")

config = dict(user_agent="project1")
def geocode(geocoder, config, query):
    cls = get_geocoder_for_service(geocoder)
    geolocator = cls(**config)
    location = geolocator.geocode(query, timeout=5, language="en")
#     print(location.address)
    if len(str(location).split(",")) > 1:
        return ""
    return location
    # geolocator = Nominatim(user_agent="project")
    # location = geolocator.geocode("PARIS", language="en")
    # print("#########", location)

########

##############################################
################ Get Full Text ###############
##############################################
"""
E-utility can help to get full text with PMICD.
Full text is stored in "full_text.json", eg:{31411: full text}
Extractions of country and organisation based on StandfordNER are stored in the "entity_fullTextCountry_fullTextOrganisation.json", eg:{"31411": [["Sweden"], ["Collection Institut Pasteur"}
"""

def get_fullText_json(PMCID):
    # https: // eutils.ncbi.nlm.nih.gov / entrez / eutils / efetch.fcgi?db = pmc & id = 4304705
    data1 = {}
    num = 0
    for i in list(PMCID):
        print(i)
        print(num)
        num += 1
        if math.isnan(i):
            data = {num:[]}
            data1.update(data)
        else:
            efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pmc&id=%s" % (int(i))
            handle = urlopen(efetch)
            data_xml = handle.read()
            text = bytes.decode(data_xml)
            paras = re.findall(r"<p>.*</p>", text)
            pa = []
            for para in paras:
                para = re.sub(r"<p>|</p>", "", para)
                para = re.sub(r"<.*>|\[\]|\(\)", "", para)
                if para:
                    pa += [para]
            para = " ".join(pa)
            para = re.sub(r"[A-Z\.]{2,3}|;]|[\.,]]", "", para)
            data = {int(i): para}
            data1.update(data)
    # with open("full_text.json", "w") as fw:
    #     json.dump(data1, fw)
    #     print("finish json.")

def parse_fullText():
    # https: // eutils.ncbi.nlm.nih.gov / entrez / eutils / efetch.fcgi?db = pmc & id = 4304705
    dic = {}
    #         s = {'sup', 'xref', 'italic', 'bold'}
    with open("full_text.json", "r") as fr:
        paras = json.load(fr)
        num = 1
        for key, value in paras.items():
            print(key)
            print(num)
            num += 1
            entity_country = []
            entity_organisation = []
            sentences = sent_tokenize(str(value))
            for sentence in sentences:
                sentence = sentence.replace("%", "percent")
                entity_country += extract_plainText_StanfordNER(stanfordNERnlp, sentence)
                entity_organisation += extract_organization_StanfordNER(sentence)
            dic[key] = [entity_country, entity_organisation]
    # with open("entity_fullTextCountry_fullTextOrganisation.json", "w") as fw:
    #     json.dump(dic, fw)
    #     print("finish json.")

def get_fullText_country():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")
    with open("entity_fullTextCountry_fullTextOrganisation.json", "r") as fr:
        dic_country = {}
        data = json.load(fr)
        countries = [v[0] for v in list(data.values())]
        print(countries)
        for i in range(len(PMID)):
            print(i)
            dic = {}
            if len(countries[i]) != 0:
                for c in countries[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_country[str(PMID[i])] = dic
    # with open("entity_count_fullText_country.json", "w") as fw:
    #     json.dump(dic_country, fw)
    #     print("finish")

def get_fullText_organisation():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")
    with open("entity_fullTextCountry_fullTextOrganisation.json", "r") as fr:
        dic_org = {}
        data = json.load(fr)
        organsations = [v[1] for v in list(data.values())]
        for i in range(len(PMID)):
            print(i)
            dic = {}
            if len(organsations[i]) != 0:
                for c in organsations[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_org[str(PMID[i])] = dic
    with open("entity_count_fullText_organisation.json", "w") as fw:
        json.dump(dic_org, fw)
        print("finish")

def get_fullText_organisation_disambiguation():
    with open("entity_count_fullText_organisation.json", "r") as fr:
        data = json.load(fr)
        dict1 = {}
        for key, value in data.items():
            # print("1", key)
            # print("2", value)
            dict2 = {}
            if value != {}:
                for k, v in value.items():
                    if re.search("Laboratory|Department|Faculty|Programme|Program|\.|-|&|\(|\)|;|,|/", k):
                        continue
                    if re.search("University", k) and len(k.split()) > 1:
                        dict2[k] = v
                        continue
                    if len(k.split()) <= 2:
                        continue
                    else:
                        dict2[k] = v
            dict1[key] = dict2
            # print("3", dict2)

    dict3 = {}
    for key, value in dict1.items():
        dict4 = {}
        judge = False
        if value != {}:
            for k, v in value.items():
                try:
                    # location = geolocator.geocode("National Institute of Health Stroke", timeout=5, language="en")
                    location = geolocator.geocode(k, language="en")
                    address = location.address
                    if address:
                        judge = True
                except:
                    continue
                if judge:
                    dict4[k] = v
        dict3[key] = dict4

    with open("entity_count_fullText_organisation2.json", "w") as fw:
        json.dump(dict3, fw)

def cal_fullText_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_fullText_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k,v in value.items() if v==max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    y_pred = np.array(country)
    y_true = np.array(list(label_country))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision1", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def cal_fullText_organisation_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_fullText_organisation2.json", "r")
    data = json.load(fr)
    organisation = []
    for value in data.values():
        if value != {}:
            c = [k for k,v in value.items() if v==max(value.values())]
            organisation.append(c[0])
        else:
            organisation.append("")
    print(organisation)

    y_pred = np.array(organisation)
    y_true = np.array(list(label_org))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

##############################################
###################  Finish  #################
##############################################


###########################################################
## Get title, abstract and affiliation Text and entities ##
###########################################################
"""
 This step is to get the title, abstract and affiliation plaintext based on the PMID and store these message in a json file "dataset_json". There is a csv file "dataset.csv" storing PMID and Data-Location-country annotated by ourselves.
"""

def get_data(data):
    dic = {}
    root = ET.fromstring(data)
    for medlineCitation in root.iter("MedlineCitation"):
        affiliations_list = []
        abstractText_list = []
        PMID = medlineCitation.find("PMID").text
        for article in medlineCitation.iter("Article"):
            articleTitle = article.find("ArticleTitle").text
            for abstract in article.iter("Abstract"):
                for abstractText in abstract.iter("AbstractText"):
                    abstractText_list.append(abstractText.text)
            for affiliationInfo in article.iter("AffiliationInfo"):
                affiliation = affiliationInfo.find("Affiliation").text
                affiliations_list.append(affiliation)
        affiliations = "".join(affiliations_list)
        if abstractText_list != [None]:
            abstractTexts = "".join(abstractText_list)
        else:
            abstractTexts = ""
        dic[PMID] = [articleTitle, abstractTexts, affiliations]
    return dic

def get_title_abstract_affiliation_json(PMID):
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = get_data(data_xml)
        data1.update(data)
    # print(len(data1))
    # with open("dataset.json", "w") as f:
    #     json.dump(data1, f)
    #     print("finish json.")

def parse_title_abstract_affiliation():
    with open("dataset_json", "r") as fr:
        data = json.load(fr)
        dic = {}
        j = 0
        for i in data.keys():
            print(i)
            j += 1
            print(j)
            abstract_entity_country = []
            abstract_entity_org = []
            title = re.sub("%", "percent", data[i][0])
            title_entity_country = extract_plainText_StanfordNER(stanfordNERnlp, title)
            title_entity_org = extract_organization_StanfordNER(title)
            abstract = sent_tokenize(data[i][1])
            for a in abstract:
                a = re.sub("%", "percent", a)
                country = extract_plainText_StanfordNER(stanfordNERnlp, a)
                org = extract_organization_StanfordNER(a)
                abstract_entity_country += country
                abstract_entity_org += org
            dic[i] = [[title_entity_country, title_entity_org], [abstract_entity_country, abstract_entity_org]]
    # with open("entity_titleCountry_abstractCountry.json", "w") as fw:
    #     json.dump(dic, fw)
    #     print("finish")

def get_title_country():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    with open("entity_titleCountry_abstractCountry.json", "r") as fr:
        data = json.load(fr)
        countries = [value[0][0] for key, value in data.items()]
        dic_country = {}
        for i in range(len(PMID)):
            print(i)
            dic = {}
            if len(countries[i]) != 0:
                for c in countries[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_country[str(PMID[i])] = dic
    # with open("entity_count_title_country.json", "w") as fw:
    #     json.dump(dic_country, fw)
    #     print("finish")

def get_title_organisation():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    with open("entity_titleCountry_abstractCountry.json", "r") as fr:
        data = json.load(fr)
        orgainsations = [value[0][1] for key, value in data.items()]
        print(orgainsations)
        dic_org = {}
        for i in range(len(PMID)):
            print(i)
            dic = {}
            if len(orgainsations[i]) != 0:
                for c in orgainsations[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_org[str(PMID[i])] = dic
    with open("entity_count_title_organisation.json", "w") as fw:
        json.dump(dic_org, fw)
        print("finish")

def get_abstract_country():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    with open("entity_titleCountry_abstractCountry.json", "r") as fr:
        data = json.load(fr)
        countries = [value[1][0] for key, value in data.items()]
        dic_country = {}
        for i in range(len(PMID)):
            print(i)
            dic = {}
            if len(countries[i]) != 0:
                for c in countries[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_country[str(PMID[i])] = dic
    with open("entity_count_abstract_country.json", "w") as fw:
        json.dump(dic_country, fw)
        print("finish")

def get_abstract_organisation():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    with open("entity_titleCountry_abstractCountry.json", "r") as fr:
        data = json.load(fr)
        orgainsations = [value[1][1] for key, value in data.items()]
        dic_org = {}
        for i in range(len(PMID)):
            print(i)
            dic = {}
            if len(orgainsations[i]) != 0:
                for c in orgainsations[i]:
                    dic[c] = dic.get(c, 0) + 1
            dic_org[str(PMID[i])] = dic
    with open("entity_count_abstract_organisation.json", "w") as fw:
        json.dump(dic_org, fw)
        print("finish")

def get_affiliation_country():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    with open("affiliation_country_city_org1.json", "r") as fr:
        data = json.load(fr)
        countries = [value[0] for key, value in data.items()]
        dic_country = {}
        for i in range(len(PMID)):
            print(i)
            if len(countries) != 0:
                dic_country[str(PMID[i])] = countries[i]
            else:
                dic_country[str(PMID[i])] = {}
    with open("entity_count_affiliation_country.json", "w") as fw:
        json.dump(dic_country, fw)
        print("finish")

def get_affiliation_organisation():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    # with open("affiliation_country_city_org1.json", "r") as fr:
    #     data = json.load(fr)
    #     affiliations = [value[2] for key, value in data.items()]
    #     dic_org = {}
    #     for i in range(len(PMID)):
    #         print(i)
    #         if len(affiliations[i]) != 0:
    #             dic_org[str(PMID[i])] = affiliations[i]
    #         else:
    #             dic_org[str(PMID[i])] = {}
    # with open("entity_count_affiliation_organisation.json", "w") as fw:
    #     json.dump(dic_org, fw)
    #     print("finish")

    with open("affiliation_country_city_org2.json", "r") as fr:
        data = json.load(fr)
        affiliations = [value[2] for key, value in data.items()]
        dic_org = {}
        for i in range(len(PMID)):
            print(i)
            if len(affiliations[i]) != 0:
                dic_org[str(PMID[i])] = affiliations[i]
            else:
                dic_org[str(PMID[i])] = {}
    with open("entity_count_affiliation_organisation2.json", "w") as fw:
        json.dump(dic_org, fw)
        print("finish")

##############################################
###################  Finish  #################
##############################################

###########################################################
###### Cal title, abstract and affiliation baseline #######
###########################################################

def cal_title_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_title_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v==max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    print(country)
    print(list(label_country))
    # test = [i for i in country if i !='']
    # print(len(test))

    y_pred = np.array(country)
    y_true = np.array(list(label_country))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def cal_title_organisation_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_title_organisation.json", "r")
    data = json.load(fr)
    organisation = []
    for value in data.values():
        if value != {}:
            c = [k for k,v in value.items() if v==max(value.values())]
            organisation.append(c[0])
        else:
            organisation.append("")
    # print(organisation)

    y_pred = np.array(organisation)
    y_true = np.array(list(label_org))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def cal_abstract_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_abstract_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v==max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    print(country)
    print(list(label_country))
    # test = [i for i in country if i !='']
    # print(len(test))

    y_pred = np.array(country)
    y_true = np.array(list(label_country))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def cal_abstract_organisation_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_abstract_organisation.json", "r")
    data = json.load(fr)
    organisation = []
    for value in data.values():
        if value != {}:
            c = [k for k,v in value.items() if v==max(value.values())]
            organisation.append(c[0])
        else:
            organisation.append("")
    print(organisation)

    y_pred = np.array(organisation)
    y_true = np.array(list(label_org))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def cal_affiliation_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_affiliation_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v==max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    print(country)
    print(list(label_country))
    # test = [i for i in country if i !='']
    # print(len(test))

    y_pred = np.array(country)
    y_true = np.array(list(label_country))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def cal_affiliation_organisation_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_affiliation_organisation2.json", "r")
    data = json.load(fr)
    organisation = []
    for value in data.values():
        if value != {}:
            c = [k for k,v in value.items() if v==max(value.values())]
            organisation.append(c[0])
        else:
            organisation.append("")
    print(organisation)

    y_pred = np.array(organisation)
    y_true = np.array(list(label_org))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()
#####

##############################################
###################  Finish  #################
##############################################

###########################################################
##### Cal title, abstract and affiliation improvement #####
###########################################################
"""
input: list
output: list
get_standard_countries(["U.S.","America","Republic of China","Korea","South Korea"])
['UNITED STATES', 'America', 'Republic of China', 'Korea', 'SOUTH KOREA']
"""
def get_standard_countries(a):
    b = []
    for i in a:
        i = i.replace("the ", "")
        # i = "China" if i == "Republic of China" else i
        # i = "South Korea" if i == "Korea" else i
        s = rangeBuilder.standardizeCountry(i)[0]
        if s != "":
            b.append(s)
        else:
            if i == "America":
                b.append("UNITED STATES")
            elif i == "Republic of China":
                b.append("CHINA")
            elif i == "Korea":
                b.append("SOUTH KOREA")
            else:
                b.append(i)
    return b

"""
input: list
output: list
"""
# def get_geocode(a):
# #     print("a",a)
#     b = []
#     for i in a:
# #         print("i",i)
#         geo = []
#         for ii in i.split(","):
# #             print("ii",ii)
#             geo.append(str(geocode("nominatim", config, ii)))
#         geo = ",".join(geo)
#         b.append(geo)
#     return b

def get_geocode(a):
#     print("a",a)
    b = []
    for i in a:
#         print("i",i)
        geo = []
        for ii in i.split(","):
#             print("ii",ii)
            geo.append(str(geocode("nominatim", config, ii)))
        geo = ",".join(geo)
        b.append(geo)
    return b

def calIm_fulltext_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_fullText_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v == max(value.values())]
            country.append(c[0])
        else:
            country.append("")
    country_std = get_standard_countries(country)
    label_std = get_standard_countries(label_country)
    y_pred = np.array(country_std)
    y_true = np.array(list(label_std))
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision1", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def calIm_title_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_title_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v == max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    country_std = get_standard_countries(country)
    label_std = get_standard_countries(label_country)

    # country_geo = []
    # for i in country_std:
    #     if i != "" and not i.isupper():
    #         country_geo.append(str(geocode("nominatim", config, i)))
    #         print("test", str(geocode("nominatim", config, i)))
    #     else:
    #         country_geo.append(i)
    # print(country_geo)

    y_pred = np.array(country_std)
    y_true = np.array(list(label_std))

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def calIm_abstract_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_abstract_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v == max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    country_std = get_standard_countries(country)
    label_std = get_standard_countries(label_country)
    y_pred = np.array(country_std)
    y_true = np.array(list(label_std))

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)
    fr.close()

def calIm_affiliation_country_pre_recall():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr = open("entity_count_affiliation_country.json", "r")
    data = json.load(fr)
    country = []
    for value in data.values():
        if value != {}:
            c = [k for k, v in value.items() if v == max(value.values())]
            country.append(c[0])
        else:
            country.append("")

    country_std = get_standard_countries(country)
    label_std = get_standard_countries(label_country)
    y_pred = np.array(country_std)
    y_true = np.array(list(label_std))

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("precision", precision)
    print("recall", recall)
    print("f1-score", f1)
    fr.close()

##############################################
###################  Finish  #################
##############################################


###########################################################
################ Cal conbination weights ##################
###########################################################

def cal_combination_weights_country():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr_fullText = open("entity_count_fullText_country.json", "r")
    fr_title = open("entity_count_title_country.json", "r")
    fr_abstract = open("entity_count_abstract_country.json", "r")
    fr_affiliation = open("entity_count_affiliation_country.json", "r")

    dataset1 = json.load(fr_fullText)
    dataset2 = json.load(fr_title)
    dataset3 = json.load(fr_abstract)
    dataset4 = json.load(fr_affiliation)

    fullText = list(dataset1.values())
    title = list(dataset2.values())
    abstract = list(dataset3.values())
    affiliation = list(dataset4.values())

    # weights_combination = [[0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0],
    #                        [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]]
    #
    # file = open("country_weights.csv", "w")
    # writer = csv.writer(file)
    # writer.writerow(["title weight", "abstract weight", "affiliation weight", "full text weight", "precision", "recall"])
    # for tw, aw, afw, fw in weights_combination:
    #     country = cal_combination_weights(PMID, title, abstract, affiliation, fullText, tw, aw, afw, fw)
    #
    #     country_std = get_standard_countries(country)
    #     label_std = get_standard_countries(label_country)
    #     y_pred = np.array(country_std)
    #     y_true = np.array(list(label_std))
    #
    #     precision = precision_score(y_true, y_pred, average="weighted")
    #     recall = recall_score(y_true, y_pred, average="weighted")
    #     print(precision)
    #     print(recall)
    #     writer.writerow([tw, aw, afw, fw, format(precision, ".3f"), format(recall, ".3f")])

    ###
    ###
    # weights_combination = [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    #
    # file = open("country_weights3.csv", "w")
    # writer = csv.writer(file)
    # writer.writerow(
    #     ["title weight", "abstract weight", "affiliation weight", "full text weight", "precision", "recall"])
    # for tw, aw, afw in weights_combination:
    #     country = cal_combination_weights(PMID, title, abstract, affiliation, tw, aw, afw)
    #
    #     country_std = get_standard_countries(country)
    #     label_std = get_standard_countries(label_country)
    #     y_pred = np.array(country_std)
    #     y_true = np.array(list(label_std))
    #
    #     precision = precision_score(y_true, y_pred, average="weighted")
    #     recall = recall_score(y_true, y_pred, average="weighted")
    #     f1 = f1_score(y_true, y_pred, average="weighted")
    #     print(precision)
    #     print(recall)
    #     print(f1)
    #     writer.writerow([tw, aw, afw, format(precision, ".3f"), format(recall, ".3f"), format(f1, ".3f")])
    ###
    ###

    # file1 = open("country_weights1.csv", "w")
    # writer1 = csv.writer(file1)
    # writer1.writerow(
    #     ["title weight", "abstract weight", "affiliation weight", "full text weight", "precision", "recall"])
    # for tw in range(1, 10):
    #     for aw in range(1, 10):
    #         for afw in range(1, 10):
    #             for fw in range(1, 10):
    #                 if tw + aw + afw + fw == 10:
    #                     country = cal_combination_weights(PMID, title, abstract, affiliation, fullText, tw, aw, afw, fw)
    #
    #                     country_std = get_standard_countries(country)
    #                     label_std = get_standard_countries(label_country)
    #                     y_pred = np.array(country_std)
    #                     y_true = np.array(list(label_std))
    #
    #                     precision = precision_score(y_true, y_pred, average="weighted")
    #                     recall = recall_score(y_true, y_pred, average="weighted")
    #                     print(precision)
    #                     print(recall)
    #                     writer1.writerow([tw/10.0, aw/10.0, afw/10.0, fw/10.0, format(precision, ".3f"), format(recall, ".3f")])

    ###
    ###
    # file1 = open("country_weights2.csv", "w")
    # writer1 = csv.writer(file1)
    # writer1.writerow(
    #     ["title weight", "abstract weight", "affiliation weight", "full text weight", "precision", "recall"])
    # for tw in range(1, 10):
    #     for aw in range(1, 10):
    #         for afw in range(1, 10):
    #             if tw + aw + afw == 10:
    #                 country = cal_combination_weights(PMID, title, abstract, affiliation, tw, aw, afw)
    #                 country_std = get_standard_countries(country)
    #                 label_std = get_standard_countries(label_country)
    #                 y_pred = np.array(country_std)
    #                 y_true = np.array(list(label_std))
    #
    #                 precision = precision_score(y_true, y_pred, average="weighted")
    #                 recall = recall_score(y_true, y_pred, average="weighted")
    #                 F1_score = f1_score(y_true, y_pred, average="weighted")
    #                 print(precision)
    #                 print(recall)
    #                 writer1.writerow([tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".3f"),
    #                                   format(recall, ".3f"), format(F1_score, ".3f")])
    ###
    ###
    file1 = open("country_weights4.csv", "w")
    writer1 = csv.writer(file1)
    writer1.writerow(
        ["affiliation weight", "title weight", "abstract weight", "full text weight", "precision", "recall"])
    for afw in range(1, 10):
        for tw in range(1, 10):
            for aw in range(1, 10):
                if tw + aw + afw == 10:
                    country = cal_combination_weights(PMID, title, abstract, affiliation, tw, aw, afw)
                    country_std = get_standard_countries(country)
                    label_std = get_standard_countries(label_country)
                    y_pred = np.array(country_std)
                    y_true = np.array(list(label_std))

                    precision = precision_score(y_true, y_pred, average="weighted")
                    recall = recall_score(y_true, y_pred, average="weighted")
                    F1_score = f1_score(y_true, y_pred, average="weighted")
                    print(precision)
                    print(recall)
                    writer1.writerow([afw / 10.0, tw / 10.0, aw / 10.0, format(precision, ".3f"),
                                      format(recall, ".3f"), format(F1_score, ".3f")])


# def cal_combination_weights(PMID, title, abstract, affiliation, fullText, t_weight, a_weight, af_weight, f_weight):
#     result = []
#     for i in range(len(PMID)):
#         dic = {}
#         if not title and not abstract and not affiliation and not fullText:
#             result.append("")
#             continue
#         if title:
#             # print("title", title)
#             for key, value in title[i].items():
#                 dic[key] = int(dic.get(key, 0)) + int(value) * t_weight
#         if abstract:
#             for key1, value1 in abstract[i].items():
#                 dic[key1] = int(dic.get(key1, 0)) + int(value1) * a_weight
#         if affiliation:
#             for key2, value2 in affiliation[i].items():
#                 if key2:
#                     dic[key2] = int(dic.get(key2, 0)) + int(value2) * af_weight
#         if fullText:
#             for key3, value3 in fullText[i].items():
#                 dic[key3] = int(dic.get(key3, 0)) + int(value3) * f_weight
#         # r = ",".join([key4 for key4, value4 in dic.items() if value4 == max(dic.values()) and value4 != 0])
#         r = [key4 for key4, value4 in dic.items() if value4 == max(dic.values()) and value4 != 0]
#         if r:
#             result.append(r[0])
#         else:
#             result.append("")
#     return result

def cal_combination_weights(PMID, title, abstract, affiliation, t_weight, a_weight, af_weight):
    result = []
    for i in range(len(PMID)):
        dic = {}
        if not title and not abstract and not affiliation:
            result.append("")
            continue
        if title:
            # print("title", title)
            for key, value in title[i].items():
                dic[key] = int(dic.get(key, 0)) + int(value) * t_weight
        if abstract:
            for key1, value1 in abstract[i].items():
                dic[key1] = int(dic.get(key1, 0)) + int(value1) * a_weight
        if affiliation:
            for key2, value2 in affiliation[i].items():
                if key2:
                    dic[key2] = int(dic.get(key2, 0)) + int(value2) * af_weight
        # r = ",".join([key4 for key4, value4 in dic.items() if value4 == max(dic.values()) and value4 != 0])
        r = [key4 for key4, value4 in dic.items() if value4 == max(dic.values()) and value4 != 0]
        if r:
            result.append(r[0])
        else:
            result.append("")
    return result

def cal_combination_weights_organisation():
    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    fr_fullText = open("entity_count_fullText_organisation.json", "r")
    fr_title = open("entity_count_title_organisation.json", "r")
    fr_abstract = open("entity_count_abstract_organisation.json", "r")
    fr_affiliation = open("entity_count_affiliation_organisation.json", "r")

    dataset1 = json.load(fr_fullText)
    dataset2 = json.load(fr_title)
    dataset3 = json.load(fr_abstract)
    dataset4 = json.load(fr_affiliation)

    fullText = list(dataset1.values())
    title = list(dataset2.values())
    abstract = list(dataset3.values())
    affiliation = list(dataset4.values())

    weights_combination = [[0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0],
                           [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]]

    # file = open("organisation_weights.csv", "w")
    # writer = csv.writer(file)
    # writer.writerow(["title weight", "abstract weight", "affiliation weight", "full text weight", "precision", "recall"])
    # for tw, aw, afw, fw in weights_combination:
    #     organisation = cal_combination_weights(PMID, title, abstract, affiliation, fullText, tw, aw, afw, fw)
    #
    #     organisation_std = get_standard_countries(organisation)
    #     label_std = get_standard_countries(label_org)
    #     y_pred = np.array(organisation_std)
    #     y_true = np.array(list(label_std))
    #
    #     precision = precision_score(y_true, y_pred, average="weighted")
    #     recall = recall_score(y_true, y_pred, average="weighted")
    #     print(precision)
    #     print(recall)
    #     writer.writerow([tw, aw, afw, fw, format(precision, ".3f"), format(recall, ".3f")])

    # file1 = open("organisation_weights1.csv", "w")
    # writer1 = csv.writer(file1)
    # writer1.writerow(
    #     ["title weight", "abstract weight", "affiliation weight", "full text weight", "precision", "recall"])
    # for tw in range(1, 10):
    #     for aw in range(1, 10):
    #         for afw in range(1, 10):
    #             for fw in range(1, 10):
    #                 if tw + aw + afw + fw == 10:
    #                     organisation = cal_combination_weights(PMID, title, abstract, affiliation, fullText, tw, aw, afw, fw)
    #
    #                     organisation_std = get_standard_countries(organisation)
    #                     label_std = get_standard_countries(label_org)
    #                     y_pred = np.array(organisation_std)
    #                     y_true = np.array(list(label_std))
    #
    #                     precision = precision_score(y_true, y_pred, average="weighted")
    #                     recall = recall_score(y_true, y_pred, average="weighted")
    #                     print(precision)
    #                     print(recall)
    #                     writer1.writerow([tw / 10.0, aw / 10.0, afw / 10.0, fw / 10.0, format(precision, ".3f"),
    #                                       format(recall, ".3f")])

    file1 = open("organisation_weights2.csv", "w")
    writer1 = csv.writer(file1)
    writer1.writerow(
        ["title weight", "abstract weight", "affiliation weight", "precision", "recall", "F1-score"])
    for tw in range(1, 10):
        for aw in range(1, 10):
            for afw in range(1, 10):
                if tw + aw + afw == 10:
                    organisation = cal_combination_weights(PMID, title, abstract, affiliation, tw, aw,
                                                           afw)

                    organisation_std = get_standard_countries(organisation)
                    label_std = get_standard_countries(label_org)
                    y_pred = np.array(organisation_std)
                    y_true = np.array(list(label_std))

                    precision = precision_score(y_true, y_pred, average="weighted")
                    recall = recall_score(y_true, y_pred, average="weighted")
                    F1_score = f1_score(y_true, y_pred, average="weighted")
                    print(precision)
                    print(recall)
                    writer1.writerow([tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".3f"),
                                      format(recall, ".3f"), format(F1_score, ".3f")])



##############################################
###################  Finish  #################
##############################################

"""
get the standard location entities of 
"""
def get_highestNumber_entity(entities_li):
    li = []
    for entities in entities_li:
        if len(entities) > 1:
            dic = {}
            for entity in entities:
                dic[entity] = dic.get(entity, 0) + 1
            li.append(",".join([key+","+str(value) for key, value in dic.items() if value == max(dic.values())]))
        elif len(entities) == 1:
            li.append(entities[0]+","+str(1))
        else:
            li.append("")
    return li

# def get_entity_title_abstract_standard():
#     with open("entity_titleCountry_abstractCountry.json", "r") as fr:
#         data = json.load(fr)
#         dic_country_title = {key: value[0][0] for key, value in data.items()}
#         dic_country_title1 = {key: value for key, value in zip(dic_country_title.keys(), get_highestNumber_entity(dic_country_title.values()))}
#         dic_country_abstract = {key: value[1][0] for key, value in data.items()}
#         dic_country_abstract1 = {key: value for key, value in
#                               zip(dic_country_abstract.keys(), get_highestNumber_entity(dic_country_abstract.values()))}
#
#         standard_title = get_standard_country(dic_country_title1.values())
#         standard_abstract = get_standard_country(dic_country_abstract1.values())
#         geo_title = []
#         geo_abstract = []
#         for s in standard_title:
#             geo = get_geocode(s) if not s.isupper() and not s else s
#             geo_title.append(geo)
#         for g in standard_abstract:
#             geo = get_geocode(g) if not s.isupper() and not g else g
#             geo_abstract.append(geo)
#
#         dic_country = {}
#         keys = list(data.keys())
#         for i in range(len(keys)):
#             dic_country[keys[i]] = [geo_title[i], geo_abstract[i]]
#         print(dic_country)
#
#     with open("entity_titleCountry_abstractCountry_standard1.json", "w") as fw:
#         json.dump(dic_country, fw)
#         print("finish.")

"""
split the country, city and organisation of the affiliation, storing in "affiliation_country_city_org.json"
"""
def get_country_city_organization():
    #### get json file, {PMID: [{country}, {city}, {organization}]}
    file = open("author_affiliation.csv", "r")
    dataframe = pd.read_csv(file)
    affiliation = dataframe.pop("Affiliation")

    i = 2
    num = 0
    dict1 = {}
    for af in affiliation:
        print(i)
        countries = []
        cities = []
        organisations = []
        if str(af) != "None":
            dic_coun = {}
            dic_city = {}
            dic_org = {}
            for a in str(af).split("$"):
                result = re.sub(r"[\w,-]+([\.|,][\w,-]+){0,3}@(([\w,-]+\.)|([\w,-]+)){1,3}", "", str(a))
                result = re.sub(r"\;[\s]", ",", result)
                result = re.sub(r"\([\w,-,\.,\,,\s]+\)", "", result)
                result = result.rstrip(".").rstrip(". ").split(",")
                country = GeoText(result[-1]).countries
                # print("before", result[-1])
                # print("after", country)
                if country == []:
                    country = re.sub(r"[0-9,\-,_]+", "", result[-1]).strip()
                    country = re.sub(r"\w{1,2}\s", "", country).strip()
                    country = re.sub(r"\.\s.+", "", country).strip()
                    # country = str(geolocator.geocode(country, language="en"))
                    # print("after2", country)
                countries.append("".join(country))
                if re.search("^[0-9,\s,A-Z]+$", result[-2]):
                    city = GeoText(result[-3]).cities
                    if city == []:
                        city = re.sub(r"[0-9,\-,_]+", "", result[-3]).strip()
                    # organisations.extend(result[0:-3])
                    o1 = []
                    for o in result[0:-3]:
                        if re.search("[0-9]|\.", o) or len(o.strip().split()) == 1:
                            continue
                        else:
                            o1.append(o)

                    try:
                        organisations.append(o1[-1])
                    except:
                        organisations.append(result[0])

                else:
                    city = GeoText(result[-2]).cities
                    if city == []:
                        city = re.sub(r"[0-9,\-,_]+", "", result[-2]).strip()
                    # organisations.extend(result[0:-2])
                    o1 = []
                    for o in result[0:-2]:
                        if re.search("[0-9]|\.", o) or len(o.strip().split()) == 1:
                            # print("1", o)
                            continue
                        else:
                            o1.append(o)
                    # print("2", o1)

                    try:
                        organisations.append(o1[-1])
                    except:
                        organisations.append(result[0])

                cities.append("".join(city))
                # print(result)
                # print("country", country)
                # print("city", city)
            for coun in countries:
                if coun:
                    dic_coun[coun.strip()] = dic_coun.get(coun.strip(), 0) + 1
                else:
                    dic_coun[coun.strip()] = "None"
            # print("countries", countries)
            # print("dic_coun", dic_coun)
            for c in cities:
                if c:
                    dic_city[c.strip()] = dic_city.get(c.strip(), 0) + 1
                else:
                    dic_city[c.strip()] = "None"
            # print("cities", cities)
            # print("dic_city", dic_city)

            for org in organisations:
                org = re.sub("^(\s|[0-9])+", "", org)
                org = re.sub("^[\s,A-Z,0-9]+$", "", org)
                if org:
                    dic_org[org.strip()] = dic_org.get(org.strip(), 0) + 1
                else:
                    # dic_org[org.strip()] = "None"
                    continue

            # dic_org[organisations.strip()] = dic_org.get(organisations.strip(), 0) + 1

            print("organisations", organisations)
            print("dict2", dic_org)
            dict1[str(PMID[num])] = [dic_coun, dic_city, dic_org]
        else:
            dict1[str(PMID[num])] = [{}, {}, {}]
        i += 1
        num += 1
    print(len(dict1))

    fw = open("affiliation_country_city_org2.json", "w")
    json.dump(dict1, fw)
    fw.close()


##############################################
#########  result of country level  ##########
##############################################

def improvement1(li):
    result = []
    for l in li:
        if l == [] or l.split(",")[0] == "":
            result.append("")
            continue
        if len(l.split(",")) == 2:
            l1, l2 = l.split(",")
        if len(l.split(",")) >= 4:
            l1, l2 = l.split(",")[0], l.split(",")[1]

        if l1.lower() == "republic of china":
            result.append("china"+","+l2)
        elif l1.lower() == "korea":
            result.append("south korea"+","+l2)
        elif l1.lower() == "america":
            result.append("united states"+","+l2)
        else:
            result.append(l1.lower()+","+l2)
    # print("finish")
    return result

def improvement2(li):
    result = []
    for l in li:
        if l == []:
            result.append("")
            continue
        if l.lower() == "republic of china":
            result.append("china")
        elif l.lower() == "korea":
            result.append("south korea")
        elif l.lower() == "america" or l.lower() == "u.s.":
            result.append("united states")
        else:
            result.append(l.lower())
    # print("2 finish")
    return result

def cal_country_level(label):
    with open("entity_titleCountry_abstractCountry_standard1.json", "r") as fr:
        data = json.load(fr)
        # title_country = [value[0] for key, value in data.items()]
        title_country = improvement1([value[0] for key, value in data.items()])
        # abstract_country = [value[1] for key, value in data.items()]
        abstract_country = improvement1([value[1] for key, value in data.items()])

    geo_label = []
    label_country = get_standard(label)
    for s in label_country:
        geo = get_geocode(s) if not s.isupper() and not s else s
        geo_label.append(geo)
    geo_label = improvement2(geo_label)

    with open("affiliation_country_city_org.json", "r") as fr1:
        data1 = json.load(fr1)
        affiliation_country = []
        for key, value in data1.items():
            a = [k+","+str(v) for k, v in value[0].items() if value[0][k] == max(value[0].values())]
            affiliation_country.append(",".join(a))
    affiliation_country = improvement1(affiliation_country)

    # precision, recall = different_weights_combination(geo_label, title_country, abstract_country, affiliation_country,
    #                                                   1, 0, 0)
    # precision, recall = different_weights_combination(geo_label, title_country, abstract_country, affiliation_country,
    #                                                   0, 1, 0)
    # precision, recall = different_weights_combination(geo_label, title_country, abstract_country, affiliation_country,
    #                                                   0, 0, 1)
    # precision, recall = different_weights_combination(geo_label, title_country, abstract_country, affiliation_country, 0.5, 0.5, 0)
    # precision, recall = different_weights_combination(geo_label, title_country, abstract_country, affiliation_country,
    #                                                   0.5, 0, 0.5)
    # precision, recall = different_weights_combination(geo_label, title_country, abstract_country, affiliation_country,
    #                                                   0, 0.5, 0.5)
    with open("combination_weights.csv", "w") as fw:
        csv_write = csv.writer(fw)
        csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision", "recall"])
        for tw in range(1, 10):
            for aw in range(1, 10):
                for afw in range(1, 10):
                    if tw + aw + afw == 10:
                        precision, recall = different_weights_combination(geo_label, title_country, abstract_country,
                                                                          affiliation_country, tw, aw, afw)
                        csv_write.writerow(
                            [tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".4f"), format(recall, ".4f")])

def different_weights_combination(label_country, title_country, abstract_country, affiliation_country, title_weight,
                                  abstract_weight, affiliation_weight):
    countries = []
    for i in range(len(label_country)):
        dic = {}
        if title_country[i] != "" and title_country[i]:
            a1, a2 = title_country[i].split(",")
            # print("a1, a2", a1, a2)
            dic[a1] = dic.get(a1, 0) + title_weight * int(a2)
        if abstract_country[i] != "":
            a3, a4 = abstract_country[i].split(",")
            # print("a3, a4", a3, a4)
            dic[a3] = dic.get(a3, 0) + abstract_weight * int(a4)
        if affiliation_country[i] != "":
            a5, a6 = affiliation_country[i].split(",")
            dic[a5] = dic.get(a5, 0) + affiliation_weight * int(a6)
            # print("a5, a6", a5, a6)
        c = ",".join([key for key, value in dic.items() if value == max(dic.values()) and value != 0])
        if title_country[i] == "" and abstract_country[i] == "" and affiliation_country[i] == "":
            c = ""
        countries.append(c)
        # print("c", c)
    print("countries", countries)

    y_true = np.array(label_country)
    y_pred = np.array(countries)

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    print("precision", precision)
    print("recall", recall)
    return precision, recall

# def different_weights_combination(label):
#     spacy_result = []
#     nltk_result = []
#     fw = open("test.csv", "w", newline="")
#     csv_write = csv.writer(fw)
#     with open("dataset_json", "r") as fr:
#         data = json.load(fr)
#         # print(data)
#         csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision", "recall"])
#         for tw in range(1, 10):
#             for aw in range(1, 10):
#                 for afw in range(1, 10):
#                     if tw + aw + afw == 10:
#                         stanford_result = []
#                         for key, value in data.items():
#                             possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_stanfordNER(
#                                 stanfordNERnlp, value, tw/10.0, aw/10.0, afw/10.0)
#                             # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_spacy(
#                             #     Spacynlp, value, tw / 10.0, aw / 10.0, afw / 10.0)
#                             # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_nltk(
#                             #     value, tw / 10.0, aw / 10.0, afw / 10.0)
#                             stanford_result = stanford_result + [" ".join(possible_stanfordNER)]
#                             # print(possible_stanfordNER)
#                         # stanford_pre = get_precision(get_standard_country(stanford_result), list(label))
#                         y_pred = np.array(get_standard_country(stanford_result))
#                         y_pred = get_geocode(y_pred)
#                         y_true = np.array(get_standard_country(list(label)))
#                         y_true = get_geocode(y_true)
#
#                         precision = precision_score(y_true, y_pred, average="weighted")
#                         recall = recall_score(y_true, y_pred, average="weighted")
#                         csv_write.writerow([tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".4f"), format(recall, ".4f")])
#                         print(
#                             "title weight: {}, abstract weight: {}, affiliation weight: {}, stanford precision: {:.4f}, stanford recall: {:.4f}".format(
#                                 tw / 10.0, aw / 10.0, afw / 10.0, precision, recall))
#                         print("1", y_true)
#                         print("2", y_pred)
#                     else:
#                         continue
#     fw.close()

# def different_weights_combination(label):
#     fw = open("test.csv", "w", newline="")
#     csv_write = csv.writer(fw)
#     with open("dataset_json", "r") as fr:
#         data = json.load(fr)
#         # print(data)
#         csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision", "recall"])
#         for tw in range(1, 10):
#             for aw in range(1, 10):
#                 for afw in range(1, 10):
#                     if tw + aw + afw == 10:
#                         stanford_result = []
#                         for key, value in data.items():
#                             possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_stanfordNER(
#                                 stanfordNERnlp, value, tw/10.0, aw/10.0, afw/10.0)
#                             stanford_result = stanford_result + [" ".join(possible_stanfordNER)]
#                         y_pred = np.array(get_standard_country(stanford_result))
#                         y_pred = get_geocode(y_pred)
#                         y_true = np.array(get_standard_country(list(label)))
#                         y_true = get_geocode(y_true)
#
#                         precision = precision_score(y_true, y_pred, average="weighted")
#                         recall = recall_score(y_true, y_pred, average="weighted")
#                         csv_write.writerow([tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".4f"), format(recall, ".4f")])
#                         print(
#                             "title weight: {}, abstract weight: {}, affiliation weight: {}, stanford precision: {:.4f}, stanford recall: {:.4f}".format(
#                                 tw / 10.0, aw / 10.0, afw / 10.0, precision, recall))
#                         print("1", y_true)
#                         print("2", y_pred)
#                     else:
#                         continue
#     fw.close()

##############################################
###################  Finish  #################
##############################################



##############################################
#######  result of organisation level  #######
##############################################

##############################################
###################  Finish  #################
##############################################

def parse_data_author(data):
    judge_same = True
    dic = {}
    root = ET.fromstring(data)
    for medlineCitation in root.iter("MedlineCitation"):
        affiliations_list = []
        abstractText_list = []
        author_affiliation = []
        author_list = []
        PMID = medlineCitation.find("PMID").text
        print(PMID)
        for article in medlineCitation.iter("Article"):
            for AuthorList in article.iter("AuthorList"):
                for author in AuthorList.iter("Author"):
                    last_name = author.find("LastName").text
                    fore_name = author.find("ForeName").text
                    full_name = last_name + " " + fore_name
                    author_list.append(full_name)
                    if author.iter("AffiliationInfo") != None:
                        for affiliationInfo in author.iter("AffiliationInfo"):
                            affiliation = affiliationInfo.find("Affiliation").text
                            affiliations_list.append(affiliation)
                            author_affiliation.append([full_name, affiliation])
            # print("1", author_list)
            # print("11", len(author_list))
            # print("2", affiliations_list)
            # print("22", len(affiliations_list))
            # print("3", author_affiliation)
            # print("33", len(author_affiliation))
            affiliations = "$".join(affiliations_list)
            if affiliations == "":
                affiliations = "None"
            authors = ",".join(author_list)
            if len(author_list) != len(affiliations_list):
                judge_same = False
            # print("judge", judge_same)
            dic[PMID] = [authors, affiliations, len(author_list), len(affiliations_list), judge_same]
    return dic

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('dataset.xlsx', index_col=0)
    data_xls.to_csv('dataset.csv', encoding='utf-8')

def get_precision(a, b):
    num = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            num += 1
    return num/len(a)

def get_json(PMID):
    data1 = {}
    for i in list(PMID):
        # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=4304705
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = parse_data_author(data_xml)
        data1.update(data)
    # print(len(data1))
    # with open("dataset_author.json", "w") as f:
    #     json.dump(data1, f)
    #     print("finish json.")

def get_csv(PMID):
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = parse_data_author(data_xml)
        data1.update(data)
    # with open("author_affiliation_json.json", "w") as f:
    #     json.dump(data1, f)
    #     print("finish json.")

    fw = open("author_affiliation.csv", "w", newline="")
    csv_write = csv.writer(fw)
    csv_write.writerow(["PMID", "Author", "Affiliation", "length(Author)", "length(Affiliation)", "equal"])
    for i in list(data1.keys()):
        csv_write.writerow([i, data1[i][0], data1[i][1], data1[i][2], data1[i][3], data1[i][4]])
    print("Finish csv.")

def get_standard_improvement(a):
    b = []
    for i in a:
#         print(type(i))
        if i != "None":
            i = i.replace("the ","")
            s = rangeBuilder.standardizeCountry(i)[0]
            if s != "":
                b.append(s)
            else:
                b.append(i)
        else:
            b.append(i)
    return b

# def get_standard(a):
#     b = []
#     for i in a:
# #         print(type(i))
#         i = i.replace("the ","")
#         s = rangeBuilder.standardizeCountry(i)[0]
#         b.append(s)
#     return b

def get_standard(a):
    b = []
    for i in a:
#         print("i",i)
#         print("i",i.split(","))
#         print(type(i))
        ss = []
        for a in i.split(","):
            # print("a",a)
            # print(len(i.split(",")))
            a = a.replace("the ","")
            s = rangeBuilder.standardizeCountry(a)[0]
#             print("s",s)
            ss.append(s)
#         print("ss",ss)
        b.append(",".join(ss))
    return b

def delete_none(a, b):
    true = []
    pred = []
    for i in range(len(a)):
        if a[i] == "None":
            continue
        else:
            true.append(a[i])
            pred.append(b[i])
    return np.array(true), np.array(pred)

def compare(li1, li2):
    li1 = str(li1).replace("nan", "").split(",")
    li2 = str(li2).replace("nan", "").split(",")
    if li1 == [] or li2 == []:
        return ""
    for i in li1:
        sim_list = [Levenshtein.ratio(i, m) for m in li2]
        sim_max = max(sim_list) if sim_list != [] else 0
        if sim_max > 0.8:
            return i

# def different_weights_combination(label):
#     spacy_result = []
#     nltk_result = []
#     fw = open("test.csv", "w", newline="")
#     csv_write = csv.writer(fw)
#     with open("dataset_json", "r") as fr:
#         data = json.load(fr)
#         # print(data)
#         csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision", "recall"])
#         for tw in range(1, 10):
#             for aw in range(1, 10):
#                 for afw in range(1, 10):
#                     if tw + aw + afw == 10:
#                         stanford_result = []
#                         for key, value in data.items():
#                             possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_stanfordNER(
#                                 stanfordNERnlp, value, tw/10.0, aw/10.0, afw/10.0)
#                             # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_spacy(
#                             #     Spacynlp, value, tw / 10.0, aw / 10.0, afw / 10.0)
#                             # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_nltk(
#                             #     value, tw / 10.0, aw / 10.0, afw / 10.0)
#                             stanford_result = stanford_result + [" ".join(possible_stanfordNER)]
#                             # print(possible_stanfordNER)
#                         # stanford_pre = get_precision(get_standard_country(stanford_result), list(label))
#                         y_pred = np.array(get_standard_country(stanford_result))
#                         y_pred = get_geocode(y_pred)
#                         y_true = np.array(get_standard_country(list(label)))
#                         y_true = get_geocode(y_true)
#
#                         precision = precision_score(y_true, y_pred, average="weighted")
#                         recall = recall_score(y_true, y_pred, average="weighted")
#                         csv_write.writerow([tw / 10.0, aw / 10.0, afw / 10.0, format(precision, ".4f"), format(recall, ".4f")])
#                         print(
#                             "title weight: {}, abstract weight: {}, affiliation weight: {}, stanford precision: {:.4f}, stanford recall: {:.4f}".format(
#                                 tw / 10.0, aw / 10.0, afw / 10.0, precision, recall))
#                         print("1", y_true)
#                         print("2", y_pred)
#                     else:
#                         continue
#     fw.close()



def get_test1(a):
    b = []
    for i in a:
        geo = []
        for ii in i.split(","):
            location = geolocator.geocode(ii, language="en")
            if len(str(location).split(",")) > 1:
                location = ""
            geo.append(str(location))
        geo = ",".join(geo)
        b.append(geo)
    # location = geolocator.geocode(loc, language="en")
    # if len(str(location).split(",")) > 1:
    #     return ""
    print(b)
    return b

if __name__ == "__main__":
    startime = time.time()

    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label_country = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")
    PMCID = dataset.pop("PMCID")

    # get_country_city_organization()


    ### show
    # with open("entity_count_abstract_country.json", "r") as fr:
    #     data = json.load(fr)
    #     for key, value in data.items():
    #         print(key)
    #         print(value.keys())
    ###

    #### get text and entities
    # get_title_abstract_affiliation_json(PMID)
    # parse_title_abstract_affiliation()
    # get_country_city_organization()
    # get_fullText_json(PMCID)
    # parse_fullText()

    # get_fullText_country()
    # get_title_country()
    # get_abstract_country()
    # get_affiliation_country()

    # get_fullText_organisation()
    # get_fullText_organisation_disambiguation()
    # get_title_organisation()
    # get_abstract_organisation()
    # get_affiliation_organisation()
    ####

    #### calculate country baseline
    # cal_fullText_country_pre_recall()
    # cal_title_country_pre_recall()
    # cal_abstract_country_pre_recall()
    # cal_affiliation_country_pre_recall()


    #### calculate organisation baseline
    # cal_fullText_organisation_pre_recall()
    # cal_title_organisation_pre_recall()
    # cal_abstract_organisation_pre_recall()
    # cal_affiliation_organisation_pre_recall()


    #### calculate country improvement
    # calIm_fulltext_country_pre_recall()
    # calIm_title_country_pre_recall()
    # calIm_abstract_country_pre_recall()
    # calIm_affiliation_country_pre_recall()

    #### calculate combination weights
    # cal_combination_weights_country()
    # cal_combination_weights_organisation()




    # get_entity_title_abstract_standard()
    # cal_country_level(label_country)

    # get_csv(PMID)

    print("time", time.time()-startime)


