import pandas as pd

raw_df = pd.read_excel("dataset\projects.xlsx")

#A priori nos interesan el id de proyecto, acronimo, titulo, summary, topic_title
NSF_df= raw_df[['projectID', 'title', 'summary', 'topic_title', 'euroSciVocCode']].copy()

text = [summary for summary in NSF_df['summary']]



with open("NSF.txt", "w", encoding='utf-8') as fp:
    for element in text:
        fp.write("%s\n" % element)
    print("Done") 
