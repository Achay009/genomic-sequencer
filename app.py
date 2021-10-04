import pandas as pd
from joblib import load
from pathlib import Path 
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


artifacts_path = Path.joinpath(Path.cwd(),'artifacts')
model = load(Path.joinpath(artifacts_path,'MultinomialNB_model.joblib'))
cv = load(Path.joinpath(artifacts_path,'vectorizer.joblib'))


st.write("""
# Genomic biomarkers for prediction of prostate cancer
""")

#read in the fruit image and render with streamlit
image = Image.open('cancer_gene.jpeg')
st.image(image, caption='Cancer Oriented medical lab',use_column_width=True)



gene_expression = st.text_input('Enter Gene Expression', value='GTTCGTTGCAACAAATTGATGAGCAATGCTTTTTTATAATGCCAACTTTGTACAAAAAAGTTGGCATGGA\nTTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCC\nATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGC\nTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAG\nCCTACAAGAAAGTACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTT\nGACACAGGTTTGGAGTATGCAAACAGCTATAATTTTGCAAAAAAGGAAAATAACTCTCCTGAACATCTAA\nAAGATGAAGTTTCTATCATCCAAAGTATGGGCTACAGAAACCGTGCCAAAAGACTTCTACAGAGTGAACC\nCGAAAATCCTTCCTTGCAGGAAACCAGTCTCAGTGTCCAACTCTCTAACCTTGGAACTGTGAGAACTCTG\nAGGACAAAGCAGCGGATACAACCTCAAAAGACGTCTGTCTACATTGAATTGGGATCTGATTCTTCTGAAG\nATACCGTTAATAAGGCAACTTATTGCAGTGTGGGAGATCAAGAATTGTTACAAATCACCCCTCAAGGAAC\nCAGGGATGAAATCAGTTTGGATTCTGCAAAAAAGGCTGCTTGTGAATTTTCTGAGACGGATGTAACAAAT\nACTGAACATCATCAACCCAGTAATAATGATTTGAACACCACTGAGAAGCGTGCAGCTGAGAGGCATCCAG\nAAAAGTATCAGGGTGAAGCAGCATCTGGGTGTGAGAGTGAAACAAGCGTCTCTGAAGACTGCTCAGGGCT\nATCCTCTCAGAGTGACATTTTAACCACTCAGCAGAGGGATACCATGCAACATAACCTGATAAAGCTCCAG\nCAGGAAATGGCTGAACTAGAAGCTGTGTTAGAACAGCATGGGAGCCAGCCTTCTAACAGCTACCCTTCCA\nTCATAAGTGACTCTTCTGCCCTTGAGGACCTGCGAAATCCAGAACAAAGCACATCAGAAAAAGTATTAAC\nTTCACAGAAAAGTAGTGAATACCCTATAAGCCAGAATCCAGAAGGCCTTTCTGCTGACAAGTTTGAGGTG\nTCTGCAGATAGTTCTACCAGTAAAAATAAAGAACCAGGAGTGGAAATGTCATCCCCTTCTAAATGCCCAT\nCATTAGATGATAGGTGGTACATGCACAGTTGCTCTGGGAGTCTTCAGAATAGAAACTACCCATCTCAAGA\nGGAGCTCATTAAGGTTGTTGATGTGGAGGAGCAACAGCTGGAAGAGTCTGGGCCACACGATTTGACGGAA\nACATCTTACTTGCCAAGGCAAGATCTAGAGGGAACCCCTTACCTGGAATCTGGAATCAGCCTCTTCTCTG\nATGACCCTGAATCTGATCCTTCTGAAGACAGAGCCCCAGAGTCAGCTCGTGTTGGCAACATACCATCTTC\nAACCTCTGCATTGAAAGTTCCCCAATTGAAAGTTGCAGAATCTGCCCAGAGTCCAGCTGCTGCTCATACT\nACTGATACTGCTGGGTATAATGCAATGGAAGAAAGTGTGAGCAGGGAGAAGCCAGAATTGACAGCTTCAA\nCAGAAAGGGTCAACAAAAGAATGTCCATGGTGGTGTCTGGCCTGACCCCAGAAGAATTTATGCTCGTGTA\nCAAGTTTGCCAGAAAACACCACATCACTTTAACTAATCTAATTACTGAAGAGACTACTCATGTTGTTATG\nAAAACAGATGCTGAGTTTGTGTGTGAACGGACACTGAAATATTTTCTAGGAATTGCGGGAGGAAAATGGG\nTAGTTAGCTATTTCTGGGTGACCCAGTCTATTAAAGAAAGAAAAATGCTGAATGAGCATGATTTTGAAGT\nCAGAGGAGATGTGGTCAATGGAAGAAACCACCAAGGTCCAAAGCGAGCAAGAGAATCCCAGGACAGAAAG\nATCTTCAGGGGGCTAGAAATCTGTTGCTATGGGCCCTTCACCAACATGCCCACAGGGTGTCCACCCAATT\nGTGGTTGTGCAGCCAGATGCCTGGACAGAGGACAATGGCTTCCATGCAATTGGGCAGATGTGTGCCCAAC\nTTTCTTGTACAAAGTTGGCATTATAAGAAAGCATTGCTTATCAATTTGTTGCAACGAAC', 
max_chars=11143, key=None, type='default', help='Enter Gene expression', autocomplete=None, on_change=None)
features = {'SEQUENCE': gene_expression}
df = pd.DataFrame(features,index=[0])
st.write("""
### Raw Gene Sequence 
""")
st.write(df)

def convert_seq(seq, window_size):
    return [seq[x:x+window_size].lower() for x in range(len(seq)-window_size+1)]

def visualize_predictions(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['BRCA1','HOXB13','BRCA2','MXI1'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Predictions", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Gene Categories", labelpad=10, weight='bold', size=12)
    ax.set_title('Model Prediction', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return


df['words'] = df.apply(lambda x: convert_seq(x['SEQUENCE'],window_size=6),axis=1)
df.drop('SEQUENCE',axis=1,inplace=True)

st.write("""
### k-mer
""")
st.write(df)

texts = list(df.words)
for doc_sec in range(len(texts)):
    texts[doc_sec] = ' '.join(texts[doc_sec])

n_grams = cv.transform(texts)

X = pd.DataFrame(n_grams.todense(), index=df.index, columns=cv.get_feature_names())
st.write(X)
y_pred = model.predict_proba(X)
st.write("""
### Model Prediction
""")
visualize_predictions(y_pred)