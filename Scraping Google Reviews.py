# -*- coding: utf-8 -*-
import csv
import os
import scrapy
#from bs4 import BeautifulSoup
from scrapy.http import Request
from selenium import webdriver
import time
import json
from unidecode import unidecode
import re 

url_lst = []
s = []

# Input file Path
path = os.path.abspath(r"")
path2 = os.path.abspath(r"")
# Output file Path
if not os.path.exists(path2):
    header_lst = []
    with open(path2, 'ab') as csvf:
        wr = csv.writer(csvf, dialect='excel')
        wr.writerow(header_lst)

class GoogleSpider(scrapy.Spider):

    name = ""   ##crawler name

    key_get = ""   ##API key
    
    with open(path, 'rb') as csvfile:
        thedata = csv.reader(csvfile, delimiter=',')
        get_flag = False        
        for row_sno in thedata:
            if get_flag == True:
                url_details = {}
                row = row_sno
                ###for removing unwanted symbols  
                Phy_line_1=re.sub('[^\-A-Z/a-z,0-9. ]+', '', row[6])                 
                query="%s %s %s %s %s" % (str(row[5]), str(Phy_line_1), str(row[11]), str(row[12]),str(row[13]))                 
                url_details['url'] = "https://maps.googleapis.com/maps/api/place/textsearch/json?query=%s &key=%s" % (query, key_get)  ###url formation to obtain place id
                url_details['ID'] = row[0]
                url_details['mer_no'] = row[1]
                url_details['Geography'] = row[2]
                url_details['Country'] = row[3]
                url_details['MER_LGL_NM'] = row[4]
                url_details['MER_NAME'] = row[5]
                url_details['Phy_ad_line_1'] = row[6]
                url_details['PHYS_AD_LINE_2_TX'] = row[7]
                url_details['PHYS_AD_LINE_3_TX'] = row[8]
                url_details['PHYS_AD_LINE_4_TX'] = row[9]
                url_details['PHYS_AD_LINE_5_TX'] = row[10]
                url_details['Phy_ad_post_town_nm'] = row[11]
                url_details['Phy_ad_rgn_nm'] = row[12]
                url_details['Phy_ad_pstl_area_cd'] = row[13]
                url_details['mcc_inus_cd'] = row[14]
                url_details['Category'] = row[15]
                url_details['Sub_Category'] = row[16]
                url_details['key'] = key_get
                url_lst.append(url_details)
            get_flag = True
    
    ##To send request on 1st URL
	def start_requests(self):
        for data in url_lst:
            try:
                url = data['url']
                yield Request(url, callback=self.parse,dont_filter=True, meta={'ID':data['ID'],'MER_NUMBER':data['mer_no'],'Geography':data['Geography'],'Country':data['Country'],'MER_LGL_NM':data['MER_LGL_NM'],'Mer_name':data['MER_NAME'],'Phy_ad_line_1':data['Phy_ad_line_1'],'PHYS_AD_LINE_2_TX':data['PHYS_AD_LINE_2_TX'],'PHYS_AD_LINE_3_TX':data['PHYS_AD_LINE_3_TX'],'PHYS_AD_LINE_4_TX':data['PHYS_AD_LINE_4_TX'],'PHYS_AD_LINE_5_TX':data['PHYS_AD_LINE_5_TX'], 'Phy_ad_post_town_nm':data['Phy_ad_post_town_nm'], 'Phy_ad_rgn_nm':data['Phy_ad_rgn_nm'], 'Phy_ad_pstl_area_cd':data['Phy_ad_pstl_area_cd'], 'mcc_inus_cd':data['mcc_inus_cd'], 'Category':data['Category'],'Sub_Category':data['Sub_Category'] ,'URL':url, 'key':data['key']})
            except Exception as s:
                
                break

    ###Parse function to tag quota exhausted or No place ID or hits
    def parse(self, response):
        

        Geography = response.meta['Geography']
        Country = response.meta['Country']
        MER_LGL_NM = response.meta['MER_LGL_NM']
        ID = response.meta['ID']
        MER_NUMBER = response.meta['MER_NUMBER']
        MER_NAME = response.meta['Mer_name']
        Phy_ad_line_1 = response.meta['Phy_ad_line_1']
        PHYS_AD_LINE_2_TX = response.meta['PHYS_AD_LINE_2_TX']
        PHYS_AD_LINE_3_TX = response.meta['PHYS_AD_LINE_3_TX']
        PHYS_AD_LINE_4_TX = response.meta['PHYS_AD_LINE_4_TX']
        PHYS_AD_LINE_5_TX = response.meta['PHYS_AD_LINE_5_TX']
        Phy_ad_post_town_nm = response.meta['Phy_ad_post_town_nm']
        Phy_ad_rgn_nm = response.meta['Phy_ad_rgn_nm']
        Phy_ad_pstl_area_cd = response.meta['Phy_ad_pstl_area_cd']
        mcc_inus_cd = response.meta['mcc_inus_cd']
        Category = response.meta['Category']
        Sub_Category = response.meta['Sub_Category']
        url = response.meta['URL']
        get_key = response.meta['key']
        
            
        temp = json.loads(response.body)['status']
       
        ####No place ID    
        if(temp == 'ZERO_RESULTS'):
                        
            lst = []
            lst.append(ID)
            lst.append(MER_NUMBER)
            lst.append(Geography)
            lst.append(Country)
            lst.append(MER_LGL_NM)
            lst.append(MER_NAME)
            lst.append(Phy_ad_line_1)
            lst.append(PHYS_AD_LINE_2_TX)
            lst.append(PHYS_AD_LINE_3_TX)
            lst.append(Phy_ad_post_town_nm)
            lst.append(Phy_ad_rgn_nm)
            lst.append(Phy_ad_pstl_area_cd)
            lst.append(mcc_inus_cd)
            lst.append(Category)
            lst.append(Sub_Category)
            lst.append("No place ID information")
            with open(path2, 'ab') as csvf:
                wr = csv.writer(csvf, dialect='excel')
                wr.writerow(lst)            
            
            return
        ####Quota exhausted
        elif(temp == 'OVER_QUERY_LIMIT'):
            lst = []
            lst.append(ID)
            lst.append(MER_NUMBER)
            lst.append(Geography)
            lst.append(Country)
            lst.append(MER_LGL_NM)
            lst.append(MER_NAME)
            lst.append(Phy_ad_line_1)
            lst.append(PHYS_AD_LINE_2_TX)
            lst.append(PHYS_AD_LINE_3_TX)
            lst.append(Phy_ad_post_town_nm)
            lst.append(Phy_ad_rgn_nm)
            lst.append(Phy_ad_pstl_area_cd)
            lst.append(mcc_inus_cd)
            lst.append(Category)
            lst.append(Sub_Category)
            lst.append('Quota exhausted')
            with open(path2, 'ab') as csvf:
                wr = csv.writer(csvf, dialect='excel')
                wr.writerow(lst)            
                        
            return
        ####else obtains Place ID and forms new URL    
        else:
            place_id = json.loads(response.body)['results'][0]['place_id']
                        
        newurl  = "https://maps.googleapis.com/maps/api/place/details/json?&placeid=%s&key=%s" %(str(place_id), get_key)
        ###Request sent using new URL
        yield Request(url=newurl, callback=self.get_parse,dont_filter=True, meta={'ID':ID, 'MER_NUMBER':MER_NUMBER, 'Geography':Geography, 'Country':Country, 'MER_LGL_NM':MER_LGL_NM, 'MER_NAME':MER_NAME , 'Phy_ad_line_1':Phy_ad_line_1,'PHYS_AD_LINE_2_TX':PHYS_AD_LINE_2_TX, 'PHYS_AD_LINE_3_TX':PHYS_AD_LINE_3_TX, 'PHYS_AD_LINE_4_TX':PHYS_AD_LINE_4_TX, 'PHYS_AD_LINE_5_TX':PHYS_AD_LINE_5_TX, 'Phy_ad_post_town_nm':Phy_ad_post_town_nm, 'Phy_ad_rgn_nm':Phy_ad_rgn_nm, 'Phy_ad_pstl_area_cd':Phy_ad_pstl_area_cd,'mcc_inus_cd':mcc_inus_cd, 'Category':Category, 'Sub_Category':Sub_Category, 'URL':url, 'NEW_URL':newurl })

	####To extract reviews and tags from JSON	
    def get_parse(self, response):
        
        # import pdb;pdb.set_trace()

        lst = []
        Geography = response.meta['Geography']
        Country = response.meta['Country']
        MER_LGL_NM = response.meta['MER_LGL_NM']
        ID = response.meta['ID']
        MER_NUMBER = response.meta['MER_NUMBER']
        MER_NAME = response.meta['MER_NAME']
        Phy_ad_line_1 = response.meta['Phy_ad_line_1']
        PHYS_AD_LINE_2_TX = response.meta['PHYS_AD_LINE_2_TX']
        PHYS_AD_LINE_3_TX = response.meta['PHYS_AD_LINE_3_TX']
        PHYS_AD_LINE_4_TX = response.meta['PHYS_AD_LINE_4_TX']
        PHYS_AD_LINE_5_TX = response.meta['PHYS_AD_LINE_5_TX']
        Phy_ad_post_town_nm = response.meta['Phy_ad_post_town_nm']
        Phy_ad_rgn_nm = response.meta['Phy_ad_rgn_nm']
        Phy_ad_pstl_area_cd = response.meta['Phy_ad_pstl_area_cd']
        mcc_inus_cd = response.meta['mcc_inus_cd']
        Category = response.meta['Category']
        Sub_Category = response.meta['Sub_Category']
        URL = response.meta['URL']
        NEW_URL = response.meta['NEW_URL']

        lst.append(ID)
        lst.append(MER_NUMBER)
        lst.append(Geography)
        lst.append(Country)
        lst.append(MER_LGL_NM)
        lst.append(MER_NAME)
        lst.append(Phy_ad_line_1)
        lst.append(PHYS_AD_LINE_2_TX)
        lst.append(PHYS_AD_LINE_3_TX)
        lst.append(Phy_ad_post_town_nm)
        lst.append(Phy_ad_rgn_nm)
        lst.append(Phy_ad_pstl_area_cd)
        lst.append(mcc_inus_cd)
        lst.append(Category)
        lst.append(Sub_Category)
        lst.append(response.meta['URL'])

		####Reviews extraction
        try:

            reviews_obj = json.loads(response.body)['result']['reviews']
            
            if(len(reviews_obj)<5):
                x=5-len(reviews_obj)
                for i in range(len(reviews_obj)):
                    lst.append(reviews_obj[i]['text'].encode('utf-8'))
                for i in range(x):
                    lst.append("")        
            else:
                for i in range(5):
                    lst.append(reviews_obj[i]['text'].encode('utf-8'))

        except Exception as d:

            lst.append("No_reviews_available")
            lst.append("")
            lst.append("")
            lst.append("")
            lst.append("")
        
		####Tags extraction	
        try:

            types_obj = json.loads(response.body)['result']['types']
            for type in types_obj:
                lst.append(type.encode('utf-8'))

        except Exception as d:

            lst.append("exception_types")
            print(d)
                        
        ###Write merchant information in output file
        with open(path2, 'ab') as csvf:

            wr = csv.writer(csvf, dialect='excel')
            wr.writerow(lst)