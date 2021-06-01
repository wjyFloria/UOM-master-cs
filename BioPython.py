import csv
import Bio
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from Bio import Entrez, Medline, SeqIO
Entrez.email = 'jiayfuanw1@student.unimelb.edu.au'
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from urllib.request import urlopen
from stanfordcorenlp import StanfordCoreNLP
from stanfordNER import extract_locations_stanfordNER, extract_organization_StanfordNER
from spacyNER import extract_locations_spacy
from nltkNER import extract_locations_nltk
import spacy
import time
import pandas as pd
import json
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
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

import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.geocoders import get_geocoder_for_service
import requests
import eventlet
import time
# get_geocoder_for_service("nominatim")
import re
from geotext import GeoText

from nltk.tokenize import sent_tokenize

import Levenshtein

# geolocator = Nominatim(user_agent="project")
config = dict(user_agent="project1")
def geocode(geocoder, config, query):
    cls = get_geocoder_for_service(geocoder)
    geolocator = cls(**config)
    location = geolocator.geocode(query, timeout=5, language="en")
#     print(location.address)
    if len(str(location).split(",")) > 1:
        return ""
    return location

# This function is to get the title, abstract and affiliation based on PMID.
def parse_data(data):
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

# This function is to get the authors and affiliations, because we want to check if the number of authors is same with that of affiliations.
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

# This function is to store the title, abstract and affiliation in json file.
def get_json_title_abstract_affiliation():
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = parse_data(data_xml)
        data1.update(data)
    with open("dataset.json", "w") as f:
        json.dump(data1, f)
        print("finish json.")

# This function is to store the authors and affiliations in json file
def get_json_authors_affiliations(PMID):
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = parse_data_author(data_xml)
        data1.update(data)
    with open("dataset_author_affiliation.json", "w") as f:
        json.dump(data1, f)
        print("finish json.")

def get_csv(PMID):
    data1 = {}
    for i in list(PMID):
        efetch = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?&db=pubmed&retmode=xml&id=%s" % (i)
        handle = urlopen(efetch)
        data_xml = handle.read()
        data = parse_data_author(data_xml)
        data1.update(data)
    fw = open("author_affiliation.csv", "w", newline="")
    csv_write = csv.writer(fw)
    csv_write.writerow(["PMID", "Author", "Affiliation", "length(Author)", "length(Affiliation)", "equal"])
    for i in list(data1.keys()):
        csv_write.writerow([i, data1[i][0], data1[i][1], data1[i][2], data1[i][3], data1[i][4]])
    print("Finish csv.")

# This function is to do normalisation.
def get_standard_country(a):
    b = []
    for i in a:
        i = i.replace("the ","")
        s = rangeBuilder.standardizeCountry(i)[0]
        if s != "":
            b.append(s)
        else:
            b.append(i)
    return b

# This function is to calculate the precision of organisation.
def cal_org_pre():
    file1 = pd.read_csv("dataset_org.csv")
    file2 = open("country_city_org.json", "r")
    org = file1.pop("Data-Location-org")
    data = json.load(file2)
    num = 0
    i = 0
    for PMID in list(data.keys()):
        print(PMID)
        org1 = list(data[PMID][2].keys())
        print("1", org1)
        print("2", org[i])
        o = str(org[i]).split(",")
        o = [re.sub(r"\(.+\)", "", i).strip() for i in o]
        print("2", " ".join(o))
        if len(o) <= 1:
            sim_list = [Levenshtein.ratio(" ".join(o), m) for m in org1]
            print("sim1", sim_list)
            sim_max = max(sim_list) if sim_list != [] else 0
            if " ".join(o) in org1 or sim_max > 0.9:
                num += 1
                print("num", num)
        if len(o) > 1:
            for j in o:
                sim_list = [Levenshtein.ratio(j, m) for m in org1]
                print("sim2", sim_list)
                sim_max = max(sim_list) if sim_list != [] else 0
                if j in org1 or sim_max > 0.9:
                    num += 1
                    print("num", num)
                    break
        i += 1
    print(num)

# compare with the organization and ground truth, calculate the number of the common facility.
def get_country_city_organization(affiliation):
    #### get json file, {PMID: [{country}, {city}, {organization}]}
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
                    organisations.extend(result[0:-3])
                else:
                    city = GeoText(result[-2]).cities
                    if city == []:
                        city = re.sub(r"[0-9,\-,_]+", "", result[-2]).strip()
                    organisations.extend(result[0:-2])
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
            # print("organisations", organisations)
            # print("dict2", dic_org)
            dict1[str(PMID[num])] = [dic_coun, dic_city, dic_org]
        else:
            dict1[str(PMID[num])] = [{}, {}, {}]
        i += 1
        num += 1
    print(len(dict1))
    # fw = open("country_city_org.json", "w")
    # json.dump(dict1, fw)
    # fw.close()
    ####

# calculate the precision and recall based on the country level.
def cal_country_level(label):
    with open("dataset_json") as fr:
        data = json.load(fr)
        stanford_result = []
        for key, value in data.items():
            possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_stanfordNER(
                stanfordNERnlp, value, 0, 1, 0)
            stanford_result = stanford_result + [" ".join(possible_stanfordNER)]
        # y_true = np.array(stanford_result)
        y_true = np.array(get_standard_country(stanford_result))    # the result of normalisation
        # y_pred = np.array(list(label))
        y_pred = np.array(get_standard_country(list(label)))
        # cm = confusion_matrix(y_true, y_pred)
        # print("1", cm)
        precision = precision_score(y_true, y_pred, average="weighted")
        print("precision", precision)
        recall = recall_score(y_true, y_pred, average="weighted")
        print("recall", recall)

        
def cal_organisation_level(label_org):
    with open("dataset_json") as fr:
        file = open("abstract_org.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["PMID", "organisation", "abstract"])

        data = json.load(fr)
        dic = {}  # store the organisations
        dic1 = {}  # store the calculation result
        li1 = []
        i = 0
        number = 0
        for key, value in data.items():
            print(key)
            stanford_result = []
            li = sent_tokenize(value[1])
            # print(li)
            # print(len(li))
            if li != []:
                for l in li:
                    l = re.sub("%", "", l)
                    result = extract_organization_StanfordNER(l)
                    stanford_result = stanford_result + result
                    # print(l)
                    # print(result)
                orgs = ",".join(stanford_result)
                if orgs != "":
                    number += 1
                    print(li)
                    print(number)
                    print(orgs)
                dic[key] = orgs
            else:
                orgs = ""
                dic[key] = orgs
            writer.writerow([key, orgs, label_org[i], li])
            c = compare(orgs, label_org[i])
            li.append(c)
            # print("organisations", stanford_result)
            # print("compare", c)
            i += 1
    # for k in li1:
    #     if k != "":
    #         print("success", k)
    # print(len(li1))
    file.close()

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


if __name__ == "__main__":
    startime = time.time()

    dataset = pd.read_csv("dataset_org.csv")
    PMID = dataset.pop("PMID")
    label = dataset.pop("Data-Location-country")
    label_org = dataset.pop("Data-Location-org")

    # get_json(PMID)
    # get_csv(PMID)

    dataset1 = pd.read_csv("author_affiliation.csv")
    affiliation = dataset1.pop("Affiliation")

    cal_organisation_level(label_org)

    #### judge if have the same organisation
    # cal_org_pre()
    ####

    #### get json file, {PMID:[{country},{city},{organization}]}
    # cal_organization(affiliation)
    ####

    ####
    ####
    ####
    # geolocator = Nominatim(user_agent="project")
    # location = geolocator.geocode("PARIS", language="en")
    # print("#########", location)
    ####
    ####
    ####

    # config = dict(user_agent="project")
    # def geocode(geocoder, config, query):
    #     cls = get_geocoder_for_service(geocoder)
    #     geolocator = cls(**config)
    #     location = geolocator.geocode(query, language="en")
    #     return location.address
    # a = geocode("nominatim", config, "paris")
    # print(a)

    ### test geopy
    # config = dict(user_agent="project")
    # def geocode(geocoder, config, query):
    #     cls = get_geocoder_for_service(geocoder)
    #     geolocator = cls(**config)
    #     location = geolocator.geocode(query, timeout=5, language="en")
    #     print(location.address)
    #     if len(str(location).split(",")) > 1:
    #         return ""
    #     return location
    # a = geocode("nominatim", config, " America")
    # print(a)
    ### geopy


    # config = dict(user_agent="project")
    # def geocode(geocoder, config, query):
    #     cls = get_geocoder_for_service(geocoder)
    #     geolocator = cls(**config)
    #     time_limit = 7
    #     with eventlet.Timeout(time_limit, False):
    #         location = geolocator.geocode(query, language="en")
    #         return "null"
    #     return location.address
    # a = geocode("nominatim", config, "paris")
    # print(a)

    # spacy_result = []
    # nltk_result = []
    # fw = open("nltk_pre.csv", "w", newline="")
    # csv_write = csv.writer(fw)
    # with open("dataset_json", "r") as fr:
    #     data = json.load(fr)
    #     # print(data)
    #     csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision"])
    #     for tw in range(1, 10):
    #         for aw in range(1, 10):
    #             for afw in range(1, 10):
    #                 if tw + aw + afw == 10:
    #                     stanford_result = []
    #                     for key, value in data.items():
    #                         # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_stanfordNER(
    #                         #     stanfordNERnlp, value, tw/10.0, aw/10.0, afw/10.0)
    #                         # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_spacy(
    #                         #     Spacynlp, value, tw / 10.0, aw / 10.0, afw / 10.0)
    #                         possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_nltk(
    #                             value, tw / 10.0, aw / 10.0, afw / 10.0)
    #                         stanford_result = stanford_result + [" ".join(possible_stanfordNER)]
    #                         # print(possible_stanfordNER)
    #                     y_true = np.array(stanford_result)
    #                     y_pred = np.array(list(label))
    #                     # cm = confusion_matrix(y_true, y_pred)
    #                     # print(cm)
    #                     precision = precision_score(y_true, y_pred, average="weighted")
    #                     print("2", precision)
    #                     recall = recall_score(y_true, y_pred, average="weighted")
    #                     print("3", recall)
    #                     # stanford_pre = get_precision(stanford_result, list(label))
    #                     # csv_write.writerow([tw/10.0, aw/10.0, afw/10.0, format(stanford_pre, ".4f")])
    #                     # print("1", stanford_result)
    #                     # print("2", list(label))
    #                     # print("title weight: {}, abstract weight: {}, affiliation weight: {}, stanford precision: {:.4f}".format(tw/10.0, aw/10.0, afw/10.0, stanford_pre))
    #                 else:
    #                     continue
    # fw.close()



    # spacy_result = []
    # nltk_result = []
    # fw = open("nltk_pre.csv", "w", newline="")
    # csv_write = csv.writer(fw)
    # with open("dataset_json1", "r") as fr:
    #     data = json.load(fr)
    #     # print(data)
    #     csv_write.writerow(["title weight", "abstract weight", "affiliation weight", "precision"])
    #     for tw in range(1, 10):
    #         for aw in range(1, 10):
    #             for afw in range(1, 10):
    #                 if tw + aw + afw == 10:
    #                     stanford_result = []
    #                     for key, value in data.items():
    #                         # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_stanfordNER(
    #                         #     stanfordNERnlp, value, tw/10.0, aw/10.0, afw/10.0)
    #                         # possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_spacy(
    #                         #     Spacynlp, value, tw / 10.0, aw / 10.0, afw / 10.0)
    #                         possible_stanfordNER, title_locations, abstract_locations, affiliations_locations = extract_locations_nltk(
    #                             value, tw / 10.0, aw / 10.0, afw / 10.0)
    #                         stanford_result = stanford_result + [" ".join(possible_stanfordNER)]
    #                         # print(possible_stanfordNER)
    #                     stanford_pre = get_precision(stanford_result, list(label))
    #                     csv_write.writerow([tw/10.0, aw/10.0, afw/10.0, format(stanford_pre, ".4f")])
    #                     print("1", stanford_result)
    #                     print("2", list(label))
    #                     print("title weight: {}, abstract weight: {}, affiliation weight: {}, stanford precision: {:.4f}".format(tw/10.0, aw/10.0, afw/10.0, stanford_pre))
    #                 else:
    #                     continue
    # fw.close()


    # print("label", len(list(label)), list(label))
    # print("result", len(stanford_result), stanford_result)
    # stanford_pre = get_precision(stanford_result, list(label))
    # dataset["StanfordNER"] = list(stanford_result)
    # dataset.to_csv("dataset.csv", index=False, mode="a")

    # print(stanford_pre)



    # dataset["SpacyNER"] = list()
    # dataset["NLTK"] = list()


    ######## title, abstract and affiliation of the first author
    # handle = Entrez.efetch(db="pubmed",id="14715089", rettype="medline", retmode="text")
    # record = Medline.parse(handle)
    # record = list(record)
    # print(record)
    # id = record[0].get("PMID","?")
    # title = record[0].get("TI","?")
    # abstract = record[0].get("AB","?")
    # affiliation = record[0].get("AFFL", "?")
    # print(id)
    # print(title)
    # print(abstract)
    # print(affiliation)
    ########

    print("time", time.time()-startime)


