# -*- coding: utf-8 -*-

import pandas as pd
import glob

#Total Input path
df=pd.read_csv(r"",dtype=object)

###Reading all the outputs from working directory 
outs = glob.glob(r".csv")

output_all=pd.DataFrame(columns=[])

###Concatenating all outputs
for i in outs:
    outs_x=pd.read_csv("%s"%i,dtype=object) 
    output_all=pd.concat([output_all,outs_x])

output_all=output_all.reset_index()
del output_all["index"]
del i
del outs
del outs_x

####Removing duplicates 
output_all=output_all.sort_values('Address',ascending=True)
output_all=output_all.drop_duplicates(subset=['ID'], keep='first', inplace=False)
output_all.to_csv("combined_output_all.csv",index=None)

###creating copy of output_all
df_op = output_all.copy()

### No place ID information is present
df_no_placeid = df_op[df_op['Address'].str.contains("No place ID information")]

##Hits
df_hits = df_op[~df_op['Address'].str.contains("No place ID information")]
df_hits = df_hits[~df_hits['Address'].str.contains("Quota exhausted")]

#Extracted output=No place id+hits
df_scraped = df_op[~df_op['Address'].str.contains("Quota exhausted")]
df_scraped=df_scraped.reset_index()
del df_scraped["index"]

##Quota Exhausted
df_quota_exhausted = df_op[df_op['Address']=="Quota exhausted"]

#Processing NPI output to get corresponding input for re-dataextraction
npi=pd.merge(df,df_no_placeid[["ID"]],on=['ID'],how="left",indicator=True)
npi=npi[npi["_merge"]=="both"]
del npi["_merge"]
npi.to_csv(r'',index = None)

# Appending new hits to previous hits

df_hits.to_csv(r'.csv',index = None)

#df4=remaining input to be scraped= Total input(df) - scraped output(df_scraped)
df_scraped1=df_scraped[["ID"]]
df_remaining=pd.merge(df,df_scraped1, on = ['ID'] ,how='left',indicator=True)
df_remaining=df_remaining[df_remaining["_merge"]=="left_only"]
del df_remaining["_merge"]
df4.to_csv(r'.csv',index = None)
