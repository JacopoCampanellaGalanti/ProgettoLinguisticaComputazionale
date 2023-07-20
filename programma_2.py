import sys
import math
import nltk
from nltk import bigrams, trigrams

#Creo delle liste che utilizzo per escludere o includere determinati tokens nelle varie funzioni
listaAggettivi = ['JJ','JJR', 'JJS'] 
listaAvverbi = ['RB', 'RBR', 'RBS', 'WRB']
listaSostantivi = ['NN', 'NNS', 'NNP', 'NNPS']

#Funzione che estrae i 15 nomi propri di persona più frequenti con relativa frequenza e stampa i risultati
def EstrazioneInformazioniQuartoPunto(tokensPOS):

    #Identifico e classifico le Entità Nominate
    analisiNE = nltk.ne_chunk(tokensPOS)
    nomi = []

    #Scorro i nodi dell'albero
    for nodo in analisiNE:
        NE = ''
        #Se sono nodi intermedi controllo che l'etichetta sia PERSON
        if hasattr(nodo, 'label'):
            if nodo.label() in "PERSON":
                #Scorro le foglie e prendo i tokens senza POS che rappresentano nomi propri di persona
                for partNE in nodo.leaves():
                    NE = NE + ' ' + partNE[0]
                nomi.append(NE)

    #Ne calcolo la distribuzione di frequenza e seleziono i primi 15
    Distr_NomiPropri = nltk.FreqDist(nomi)
    primi15_NomiPropri = Distr_NomiPropri.most_common(15)

    #Scorro i nomi, stampo i nomi propri e relative frequenze
    for nome in primi15_NomiPropri:
        print("Nome proprio:", nome[0], " Freq:", nome[1])
    print()

#Funzione che estrae le frasi con almeno 6 tokens e con meno di 25, in cui ogni token ricorre almeno 2 volte nel testo e che calcola ed estrae le frasi con distribuzione di frequenza media massima, con distribuzione di frequenza media minima e con probMarkov di ordine 2 massima con relativi valori    
def EstrazioneInformazioniTerzoPunto(frasi, tokens, bigrammi, trigrammi):

    frasiAggiornate = []
    somma_freq = 0.0
    distr_mediaMAX = 0.0
    distr_mediaMIN = 1.7976931348623157e+308
    prodotto_prob_trigrammi = 1
    probMarkov2MAX = 0.0

    #Scorro le frasi, ricavo le frasi con almeno 6 tokens e con meno di 25, in cui ogni token ricorre almeno due volte nel testo
    for frase in frasi:
        tokensFrase = nltk.word_tokenize(frase)
        i = 0
        if len(tokensFrase) > 5 and len(tokensFrase) < 25:
            for tok in tokensFrase:
                freq_tok = tokens.count(tok)
                if freq_tok > 1:
                    i += 1
            if i == len(tokensFrase):
                frasiAggiornate.append(frase)
    
    #Scorro le nuove frasi, ricavo la distribuzione media di frequenza dei tokens delle nuove frasi 
    for frase in frasiAggiornate:
        tokensFraseAggiornata = nltk.word_tokenize(frase)
        for tok in tokensFraseAggiornata:
            freq_token = tokens.count(tok)
            somma_freq += freq_token
        distr_media = somma_freq/len(tokensFraseAggiornata)
        somma_freq = 0.0
        #Ricavo quella massima con relativa frase
        if distr_media > distr_mediaMAX:
            distr_mediaMAX = distr_media
            frase_distrMAX = frase
        #Ricavo quella minima con relativa frase
        if distr_media < distr_mediaMIN:
            distr_mediaMIN = distr_media
            frase_distrMIN = frase

    #Scorro le nuove frasi, calcolo la probabilità delle singole frasi con Catena di Markov di ordine 2 e ricavo quella con valore maggiore
    for frase in frasiAggiornate:
        #Per prima cosa ricavo il prodotto tra la probabilità del primo token di ogni frase con la probabilità del primo bigramma
        tokensFraseAggiornata = nltk.word_tokenize(frase)
        trigrammi_frasi = list(trigrams(tokensFraseAggiornata))
        bigrammi_frasi = list(bigrams(tokensFraseAggiornata))
        prob_primotoken = tokens.count(bigrammi_frasi[0][0])/len(tokens)
        prob_primobigramma = bigrammi.count(bigrammi_frasi[0])/tokens.count(bigrammi_frasi[0][0])
        probMarkov2 = prob_primotoken*prob_primobigramma
        #Ricavo la frequenza di ogni trigramma e ogni bigramma delle frasi grazie alla quale faccio il prodotto tra le probabilità per ricavare il valore della catena
        for trigramma in trigrammi_frasi:
            prodotto_prob_trigrammi *= trigrammi.count(trigramma)/bigrammi.count((trigramma[0], trigramma[1]))
        probMarkov2 = probMarkov2*prodotto_prob_trigrammi
        #Ricavo la probabilità maggiore con relativa frase
        if probMarkov2 > probMarkov2MAX:
            probMarkov2MAX = probMarkov2
            frase_probMarkov2MAX = frase
                
    return distr_mediaMAX, distr_mediaMIN, frase_distrMAX, frase_distrMIN, probMarkov2MAX, frase_probMarkov2MAX  

#Funzione che calcola e ordina i 20 bigrammi composti da Aggettivo e Sostantivo e dove ogni token ha frequenza maggiore di 3 con frequenza massima, con probabilità condizionata massima e con LMI massima, e che stampa i risultati con relativa frequenza, probabilità e LMI          
def EstrazioneInformazioniSecondoPunto(tokens, tokensPOS):

    bigrammiAggSos = []
    freqtokensPOS = nltk.FreqDist(tokensPOS)
    bigrammiPOS = list(bigrams(tokensPOS))

    #Scorro i bigrammi con POS e ricavo quelli composti da Aggettivo e Sostantivo con relativa POS e con frequenza maggiore di 3 dei singoli tokens
    for bigramma in bigrammiPOS:
        if bigramma[0][1] in listaAggettivi and bigramma[1][1] in listaSostantivi:
            if freqtokensPOS[bigramma[0]] > 3 and freqtokensPOS[bigramma[1]] > 3:
                bigrammiAggSos.append(bigramma)

    #Calcolo la distribuzione di frequenza dei nuovi bigrammi e ne ricavo i primi 20
    freqbigrammiAggSos = nltk.FreqDist(bigrammiAggSos)
    primi20BigrammiAggSosfreq = freqbigrammiAggSos.most_common(20)

    #Scorro i nuovi bigrammi, ricavo e stampo i bigrammi senza POS con relativa frequenza
    for bigramma in primi20BigrammiAggSosfreq:
        aggettivo = bigramma[0][0][0]
        sostantivo = bigramma[0][1][0]
        freq = bigramma[1]
        print("Bigramma:", aggettivo, sostantivo, " Freq:", freq)
    print()

    #Scorro i nuovi bigrammi, calcolo la probabilità condizionata dei bigrammi e creo un Dizionario con chiave i bigrammi e valore la loro probabilità condizionata
    bigrammiProb = {}

    for bigramma in bigrammiAggSos:
        aggettivo = bigramma[0][0]
        freq_aggettivo = tokens.count(aggettivo)
        freq_bigramma = bigrammiPOS.count(bigramma)
        prob_Condbigramma = freq_bigramma/freq_aggettivo
        bigrammiProb[bigramma] = prob_Condbigramma

    #Chiamo la funzione che ordina il Dizionario e ricavo i primi 20 bigrammi con probabilità maggiore
    bigrammiProbOrdinata = OrdinaBigrammi20(bigrammiProb)
    bigrammiProb20 = bigrammiProbOrdinata[:20]

    #Scorro gli elementi del dizionario, stampo i bigrammi senza POS e con relativa probabilità
    for elem in bigrammiProb20:
        print("Bigramma:", elem[0][0][0], elem[0][1][0], " Prob:", elem[1])
    print()

    #Scorro i nuovi bigrammi, calcolo la LMI e creo un Dizionario con chiave i bigrammi e valore la loro LMI
    bigrammiLMI = {}
    
    for bigramma in bigrammiAggSos:
        aggettivo = bigramma[0][0]
        sostantivo = bigramma[1][0]
        freq_aggettivo = tokens.count(aggettivo)
        freq_sostantivo = tokens.count(sostantivo)
        freq_bigramma = bigrammiPOS.count(bigramma)
        prob_aggettivo = freq_aggettivo/len(tokens)
        prob_sostantivo = freq_sostantivo/len(tokens)
        prob_Condbigramma = freq_bigramma/freq_aggettivo
        prob_Congbigramma = prob_Condbigramma*prob_aggettivo
        p = prob_Congbigramma/(prob_aggettivo*prob_sostantivo)
        LMI = freq_bigramma*math.log(p, 2)
        bigrammiLMI[bigramma] = LMI

    #Chiamo la funzione che ordina il dizionario e ricavo i primi 20 bigrammi per LMI
    bigrammiLMIOrdinata = OrdinaBigrammi20(bigrammiLMI)
    bigrammiLMI20 = bigrammiLMIOrdinata[:20]

    #Scorro gli elementi del dizionario, stampo i bigrammi senza POS e con relativa LMI
    for elem in bigrammiLMI20:
        print("Bigramma:", elem[0][0][0], elem[0][1][0], " LMI:", elem[1])

#Funzione che ordina le chiavi del Dizionario sulla base del loro valore
def OrdinaBigrammi20(dict):
    return sorted(dict.items(), key =lambda x: x[1], reverse = True)

#Funzione che calcola e stampa in ordine di frequenza e con relativa frequenza le 10 POS, i 10 bigrammi di POS, i 10 trigrammi di POS, i 20 Aggettivi e i 20 Avverbi più frequenti   
def EstrazioneInformazioniPrimoPunto(tokensPOS, SequenzaPOS):

    #Ricavo la distribuzione di frequenza delle POS e ne ricavo le prime 10
    DistFreqPOS = nltk.FreqDist(SequenzaPOS)
    prime10POS = DistFreqPOS.most_common(10)

    #Scorro le POS e le stampo con relativa frequenza
    for pos in prime10POS:
        POS = pos[0]
        freqPOS = pos[1]
        print("POS:", POS, " Freq:", freqPOS)
    print()

    #Ricavo i bigrammi di POS, la loro distribuzione di frequenza e considero i primi 10
    bigrammi = list(bigrams(SequenzaPOS))
    DistFreqBigrammi = nltk.FreqDist(bigrammi)
    primi10Bigrammi = DistFreqBigrammi.most_common(10)

    #Scorro i bigrammi di POS e li stampo con relativa frequenza
    for bigramma in primi10Bigrammi:
        Bigramma = bigramma[0]
        freqBigramma = bigramma[1]
        print("Bigramma:", Bigramma, " Freq:", freqBigramma)
    print()

    #Faccio la stessa cosa per i trigrammi
    trigrammi = list(trigrams(SequenzaPOS))
    DistFreqTrigrammi = nltk.FreqDist(trigrammi)
    primi10Trigrammi = DistFreqTrigrammi.most_common(10)

    for trigramma in primi10Trigrammi:
        Trigramma = trigramma[0]
        freqTrigramma = trigramma[1]
        print("Trigramma:", Trigramma, " Freq:", freqTrigramma)
    print()
    
    TestoAnalizzato_filtratoAggettivi = []
    TestoAnalizzato_filtratoAvverbi = []

    #Scorro i tokens con POS e ricavo la lista di Aggettivi e Avverbi del testo
    for token in tokensPOS:
        if token[1] in listaAggettivi:
            TestoAnalizzato_filtratoAggettivi.append(token[0])
        elif token[1] in listaAvverbi:
            TestoAnalizzato_filtratoAvverbi.append(token[0])

    #Calcolo la distribuzione di frequenza degli Aggettivi e prendo i primi 20
    DistFreqAggettivi = nltk.FreqDist(TestoAnalizzato_filtratoAggettivi)
    primi20Aggettivi = DistFreqAggettivi.most_common(20)

    #Scorro gli Aggettivi e li stampo con relativa frequenza
    for aggettivo in primi20Aggettivi:
        Aggettivo = aggettivo[0]
        freqAggettivo = aggettivo[1]
        print("Aggettivo:", Aggettivo, " Freq:", freqAggettivo)
    print()

    #Faccio la stessa cosa per gli Avverbi
    DistFreqAvverbi = nltk.FreqDist(TestoAnalizzato_filtratoAvverbi)
    primi20Avverbi = DistFreqAvverbi.most_common(20)

    for avverbio in primi20Avverbi:
        Avverbio = avverbio[0]
        freqAvverbio = avverbio[1]
        print("Avverbio:", Avverbio, " Freq:", freqAvverbio)

#Funzione che restituisce i tokens totali, tokens totali con POS e i POS totali
def AnalisiTesto(frasi):

    tokensTOT = []
    tokensPOStot = []
    SequenzaPOS = []

    #Scorro le frasi, le tokenizzo e ne ricavo i tokens con POS, per poi ricavare i tokens totali e i tokens totali con POS 
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensTOT += tokens
        tokensPOStot += tokensPOS

    #Scorro i tokens con POS, ricavo la sequenza di tutte le POS 
    for tok in tokensPOStot:
        SequenzaPOS.append(tok[1])
        
    return tokensTOT, tokensPOStot, SequenzaPOS

#Funzione principale che chiama le varie funzioni di estrazione d'informazione e stampa i risultati   
def main(file1, file2):

    #Apro i due file di testo
    fileInput1 = open(file1, "r")
    fileInput2 = open(file2, "r")
    #Assegno il contenuto dei due file di testo a due variabili
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    #Carico il modello di tokenizzazione del testo
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #Divido i file di testo in frasi
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    #Chiamo la funzione che restituisce i tokens totali dei file di testo, i tokens totali con relativa POS e le POS totali 
    tokens1, tokensPOS1, SequenzaPOS1 = AnalisiTesto(frasi1)
    tokens2, tokensPOS2, SequenzaPOS2 = AnalisiTesto(frasi2)

    #Ricavo i bigrammi e i trigrammi dei due file di testo
    bigrammi1 = list(bigrams(tokens1))
    trigrammi1 = list(trigrams(tokens1))
    bigrammi2 = list(bigrams(tokens2))
    trigrammi2 = list(trigrams(tokens2))

    #Stampo i risultati del primo punto
    print()
    print()
    print("--PRIMO PUNTO--")
    print()
    print()
    print("Estraggo ed ordino con frequenza decrescente, indicando anche la relativa frequenza, le informazioni richieste al primo punto per i due file di testo:")
    print()
    print("Estrazione delle informazioni per il file", file1, ":")
    print()
    #Chiamo la funzione che estrae e ordina per frequenza le 10 POS, i 10 bigrammi di POS, i 10 trigrammi di POS e i 20 Aggettivi e i 20 Avverbi più frequenti e stampa i risultati con relativa frequenza dei due file di testo
    EstrazioneInformazioniPrimoPunto(tokensPOS1, SequenzaPOS1)
    print()
    print("Estrazione delle informazioni per il file", file2, ":")
    print()
    EstrazioneInformazioniPrimoPunto(tokensPOS2, SequenzaPOS2)

    #Stampo i risultati del secondo punto
    print()
    print()
    print("--SECONDO PUNTO --")
    print()
    print()
    print("Estraggo ed ordino i primi 20 bigrammi, composti da Aggettivo e Sostantivo e dove ogni token ha frequenza maggiore di 3, in base alle informazioni richieste al secondo punto per i due file di testo:")
    print()
    print("Estrazione delle informazioni per il file", file1, ":")
    print()
    #Chiamo la funzione che estrae e ordina i 20 bigrammi composti da Aggettivo e Sostantivo e dove ogni token ha frequenza maggiore di 3 con frequenza massima, con probabilità condizionata massima e con LMI massima, e che stampa i risultati con relativa frequenza, probabilità e LMI dei due file di testo
    EstrazioneInformazioniSecondoPunto(tokens1, tokensPOS1)
    print()
    print("Estrazione delle informazioni per il file", file2, ":")
    print()
    EstrazioneInformazioniSecondoPunto(tokens2, tokensPOS2)

    #Chiamo la funzione che estrae le frasi con almeno 6 tokens e con meno di 25, in cui ogni token ricorre almeno 2 volte nel testo e che calcola ed estrae le frasi con distribuzione media di frequenza massima, con distribuzione media di frequenza minima e con probMArkov di ordine 2 massima con relativi valori dei due file di testo
    distr_mediaMAX1, distr_mediaMIN1, frase_distrMAX1, frase_distrMIN1, probMarkov2MAX1, frase_probMarkov2MAX1 = EstrazioneInformazioniTerzoPunto(frasi1, tokens1, bigrammi1, trigrammi1)
    distr_mediaMAX2, distr_mediaMIN2, frase_distrMAX2, frase_distrMIN2, probMarkov2MAX2, frase_probMarkov2MAX2 = EstrazioneInformazioniTerzoPunto(frasi2, tokens2, bigrammi2, trigrammi2)

    #Stampo i risultati del terzo punto 
    print()
    print()
    print("--TERZO PUNTO--")
    print()
    print()
    print("Estraggo le frasi dei due file di testo con almeno 6 tokens e con meno di 25 tokens e in cui ogni token occorre almeno due volte nel testo, ricavo poi le informazioni richieste:")
    print()
    print(file1, ":")
    print()
    print("La frase con la distribuzione media di frequenza maggiore è '", frase_distrMAX1, "' con distribuzione media pari a", distr_mediaMAX1, "mentre quella con distribuzione più bassa è '", frase_distrMIN1, "' con valore pari a", distr_mediaMIN1, ".", "La frase con probabilità maggiore è '", frase_probMarkov2MAX1, "' con probabilità pari a", probMarkov2MAX1)
    print()
    print(file2, ":")
    print()
    print("La frase con la distribuzione media di frequenza maggiore è '", frase_distrMAX2, "' con distribuzione media pari a", distr_mediaMAX2, "mentre quella con distribuzione più bassa è '", frase_distrMIN2, "' con valore pari a", distr_mediaMIN2, ".", "La frase con probabilità maggiore è '", frase_probMarkov2MAX2, "' con probabilità pari a", probMarkov2MAX2)

    #Stampo i risultati del quarto punto
    print()
    print()
    print("--QUARTO PUNTO--")
    print()
    print()
    print("Estraggo i 15 nomi propri di persona più frequenti, ordinati per frequenza, dei due file di testo:")
    print()
    print(file1, ":")
    print()
    #Chiamo la funzione che estrae i 15 nomi propri di persona più frequenti con relativa frequenza e stampa i risultati per i due file di testo
    EstrazioneInformazioniQuartoPunto(tokensPOS1)
    print(file2, ":")
    print()
    EstrazioneInformazioniQuartoPunto(tokensPOS2)
    
main (sys.argv[1], sys.argv[2])
