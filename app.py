import streamlit as st
st.set_page_config(page_title="Calculator", page_icon="🤖", layout="wide")


st.title("WER / Accuracy Calculator")
st.write("This is a simple calculator to calculate the WER and Accuracy of a ASR system.")
tab1, tab2, tab3 = st.tabs(["Manual","Copypasta", "Under Construction"])

with st.container():
    pass

with tab1:
    st.header("Calculate using manual input")
    st.text('Better way if the hypothesis and reference are too short')
    insCount = st.text_input('Inserted Words')
    delCount = st.text_input('Deleted Words')
    subCount = st.text_input('Substituted Words')
    totCount = st.text_input('Total Words')

    if st.button("Calculate", key='calculate'):
        wer_manual = ((int(insCount) + int(delCount) + int(subCount)) / int(totCount))*100   
        accuracyManual = ((int(totCount) - int(delCount) - int(subCount)) / int(totCount))*100
        st.write("WER:", wer_manual, "%")
        st.write("Accuracy:", accuracyManual, "%")


with tab2:
    st.header("Copypasta !")
    hypothesis = st.text_area('Paste hypothesis text here', key='hypothesis', value='')
    
    refrence = st.text_area('Paste reference text here', key='reference')

    def word_count(refrence):
        return (len(refrence.strip().split(" ")))

    st.write("Total words in reference:", word_count(refrence))

with tab3:
    st.header("Under construction")
    st.text('This is a work in progress')

