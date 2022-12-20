import streamlit as st
import pandas as pd
import numpy as np
import pickle as pickle
import sklearn

## data

feel =['Aroma','Flavor','Aftertaste','Acidity','Body','Balance']

alg = ['Gaussian', 'Multinomial', 'Bernoulli', 'Complementary']

df = pd.read_csv('53_coffee.csv')

df_proc = pd.read_csv('cleandata.csv')

gaussian = pickle.load(open('nb_gaussian.pkl', 'rb'))
multi = pickle.load(open('nb_multinomial.pkl', 'rb'))
bern = pickle.load(open('nb_bernoulli.pkl', 'rb'))
comp = pickle.load(open('nb_complementary.pkl', 'rb'))

##layout

st.write('Bagus Rahma Aulia Chandra - A11.2017.10295')
st.markdown(''' ## Prediksi Jenis Kopi Berdasar Data Penilaian Mutu dari sebuah Biji Kopi ''')
st.markdown(''' Kopi adalah minuman hasil seduhan biji kopi yang telah disangrai dan dihaluskan menjadi bubuk.
                Kopi merupakan salah satu komonitas di dunia yang dibudidayakan lebih dari 50 negara.
                Dua spesies pohon kopi yang dikenal secara umum yaitu Kopi Robusta (Coffea canephora) dan 
                Kopi Arabika (Coffea arabica).
                
                Penilaian Mutu suatu kopi bisa dinilai dari berbagai aspek yang meliputi aroma, flavor, after taste, acidity, body, balance, uniformity, clean cup, sweetness.
                Penilaian dilaksanakan dengan prinsip blind test supaya hasilnya benar-benar objektif. Artinya penguji sama sekali tidak diberi tahu terlebih dahulu tentang
                jenis kopi yang akan dinilai sebelumnya. Penilaian dilakukan menggunakan formulir khusus yang telah disiapkan sebelumnya.
                
                Sesuai penjelasan diatas,aspek penilaian di mutu sebuah kopi meliputi :
                1. Aroma
                2. Flavor (Rasa)
                3. After Taste
                4. Acidity (Keasaman)
                5. Body
                6. Balance
                7. Uniformity
                8. Clean cup
                9. Sweetness
                
                
Namun kali ini, saya hanya akan memakai 6 data penilaian,diantaranya : 
1. Aroma
2. Flavor
3. Aftertaste
4. Acidity
5. Body
6. Balance

*catatan :di tugas yang saya paparkan ini,ini sudah saya jadikan 1:1 dikarenakan,data yang saya dapatkan dari
website kaggle ini,data Robusta hanya 28 data,dan Arabica ada 1312 data,maka terjadi ketimpangan teramat sangat jauh
maka daripada itu,yang Arabica,saya ambil hanya 25 dan Robusta 25 Data''')

st.write(df)

st.markdown(''' Setelah dilakukan proses data cleaning dan data processing antara lain:  
            - Menghapus kolom yang tidak diperlukan    
            - Membersihkan karakter yang tidak dibutuhkan dalam pengetesan mutu sebuah kopi  
            - Mengubah Jenis Kopi menjadi inisialisasi untuk memudahkan dalam melakukan penghitungan dalam algoritma K-NN  
        
Berikut adalah contoh data yang sudah di proses: ''')

st.write(df_proc)

st.markdown(''' ### Masukkan Data Penilaian untuk melakukan prediksi Jenis Kopi : ''')

algorithm = st.selectbox('Pilih Algoritma', alg)
feels = st.multiselect('Masukkan Data Penilaian :', feel, max_selections=6)
    
preds =st.button('Predict')
    
if preds:
    penilaian = []
    for result in feels:
        penilaian.append(feel.index(result))
    
    if algorithm == 'Gaussian':
        prediction = gaussian.predict([penilaian])
        if prediction == 1:
            prediction = 'Win - Accuracy 59%'
        else :
            prediction = 'Lose - Accuracy 59%'
        st.write(prediction)
    elif algorithm == 'Multinomial':
        prediction = multi.predict([penilaian])
        if prediction == 1:
            prediction = 'Win - Accuracy 54%'
        else :
            prediction = 'Lose - Accuracy 54%'
        st.write(prediction)
    elif algorithm == 'Bernoulli':
        prediction = bern.predict([penilaian])
        if prediction == 1:
            prediction = 'Win - Accuracy 59%'
        else :
            prediction = 'Lose - Accuracy 59%'
        st.write(prediction)
    elif algorithm == 'Complementary':
        prediction = comp.predict([penilaian])
        if prediction == 1:
            prediction = 'Win - Accuracy 52%'
        else :
            prediction = 'Lose - Accuracy 52%'
        st.write(prediction)
else:
    st.write('')
