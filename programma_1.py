import sys
import nltk

#Creo delle liste che utilizzo per escludere o includere determinati tokens nelle varie funzioni
listaPunteggiatura = ['.', ',', ':', '!', '?', '(', ')']
listaParolePiene = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'WRB']
listaParoleFunzionali = ['CC', 'DT', 'IN', 'PDT', 'POS', 'PRP', 'PRP$', 'WDT', ' WP', 'WP$']

#Funzione che calcola la distribuzione percentuale dell'insieme delle parole piene e dell'insieme di quelle percentuali
def CalcoloStatisticheQuintoPunto(TestoTokenizzato, TestoAnalizzatoPOS):

    insiemeParolePiene = []
    insiemeParoleFunzionali = []
    freq_relativaInsiemeParolePiene = 0.0
    freq_relativaInsiemeParoleFunzionali = 0.0

    #Scorro la lista di tokens con POS e creo due liste, una con tutti i tokens che corrispondono a parole piene, e una con quelli che corrispondono a parole funzionali
    for token in TestoAnalizzatoPOS:
        if token[1] in listaParolePiene:
            insiemeParolePiene.append(token[0])
        elif token[1] in listaParoleFunzionali:
            insiemeParoleFunzionali.append(token[0])

    #Ne ricavo i vocabolari
    vocabolarioParolePiene = list(set(insiemeParolePiene))
    vocabolarioParoleFunzionali = list(set(insiemeParoleFunzionali))

    #Scorro i tokens nella prima lista, quelli che corrispondono a parole piene, calcolo le frequenze assolute, successivamente quelle relative e le sommo per ricavare la frequenza relativa del loro insieme
    for token in vocabolarioParolePiene:
        freq_piene = TestoTokenizzato.count(token)
        freq_relativa_piene = freq_piene/len(TestoTokenizzato)
        freq_relativaInsiemeParolePiene += freq_relativa_piene

    #Faccio la stessa cosa per le parole funzionali
    for token in vocabolarioParoleFunzionali:
        freq_funzionali = TestoTokenizzato.count(token)
        freq_relativa_funzionali = freq_funzionali/len(TestoTokenizzato)
        freq_relativaInsiemeParoleFunzionali += freq_relativa_funzionali

    #Ne ricavo la distribuzione percentuale
    dist_percentualePiene = freq_relativaInsiemeParolePiene*100
    dist_percentualeFunzionali = freq_relativaInsiemeParoleFunzionali*100

    return dist_percentualePiene, dist_percentualeFunzionali

#Funzione che calcola e stampa la grandezza del vocabolario e la TTR all'aumentare dei tokens di 500 in 500
def CalcoloStatisticheQuartoPunto(TestoTokenizzato):

    #Scorro tutto il testo e ricavo ogni volta di 500 in 500 tokens il testo tokenizzato, il vocabolario, la sua grandezza e la TTR
    for i in range(0, len(TestoTokenizzato), 500):
        TestoTokenizzato500 = TestoTokenizzato[0:i+500]
        vocabolario500 = list(set(TestoTokenizzato500))
        grandezzaVocabolario500 = len(vocabolario500)
        TTR500 = grandezzaVocabolario500/len(TestoTokenizzato500)
        
        #Stampo i risultati di 500 in 500
        print(i, '-', i+500, "tokens:")
        print("Dimensioni del vocabolario:", grandezzaVocabolario500)
        print("TTR:", TTR500)
        print()

#Funzione che calcola il numero di hapax considerando solo i primi 1000 tokens
def CalcoloStatisticheTerzoPunto(TestoTokenizzato):

    #Prendo i primi 1000 tokens 
    TestoTokenizzato1000 = TestoTokenizzato[0:1000]
    hapax = []

    #Scorro i primi 1000 tokens e ne calcolo la frequenza
    for token in TestoTokenizzato1000:
        freq_token = TestoTokenizzato.count(token)
        #se la frequenza è 1 sono hapax, li aggiungo alla lista e ne calcolo il numero
        if freq_token == 1:
            hapax.append(token)
    numerodiHapax = len(hapax)

    return numerodiHapax

#Funzione che calcola lunghezza media delle frasi e dei tokens senza punteggiatura 
def CalcoloStatisticheSecondoPunto(frasi, TestoTokenizzato):

    numFrasi = 0.0
    lunghezzaFrasi = 0.0
    numTokens = 0.0
    lunghezzaTokens = 0.0
    TestoAnalizzato_filtrato = []

    #Scorro le frasi, ricavo il numero di frasi e il numero di tokens in ogni frase per calcolare la media della loro lunghezza
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        numFrasi += 1
        lunghezzaFrasi += len(tokens)  
    lunghezzamediaFrasi = lunghezzaFrasi/numFrasi

    #Scorro i tokens, escludo i tokens che sono punteggiatura
    for token in TestoTokenizzato:
        if not token in listaPunteggiatura:
            TestoAnalizzato_filtrato.append(token)

    #Scorro i tokens filtrati, calcolo il numero di tokens presi in considerazione e la loro lunghezza contandone i caratteri per poi ricavare la media della loro lunghezza
    for token in TestoAnalizzato_filtrato:
        numTokens += 1
        lunghezzaTokens += len(token)
    lunghezzamediaTokens = lunghezzaTokens/numTokens
    
    return lunghezzamediaFrasi, lunghezzamediaTokens

#Funzione che calcola il numero di frasi e tokens 
def CalcoloStatistichePrimoPunto(frasi):

    numFrasi = 0.0
    tokensTOT = []
    
    #Scorro le frasi e ricavo i tokens totali e calcolo il numero di frasi 
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        numFrasi += 1
        tokensTOT += tokens
    
    #Ricavo il numero dei tokens totali
    numTokens = len(tokensTOT)

    return numFrasi, numTokens

#Funzione che restituisce i tokens totali e tokens totali con POS dei file di testo
def AnnotazioneLinguistica(frasi):

    tokensTOT = []
    tokensPOStot = []

    #Scorro le frasi, le tokenizzo e ne ricavo i tokens con POS, per poi ricavare i tokens totali e i tokens totali con POS 
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        tokensTOT += tokens
        tokensPOStot += tokensPOS

    return tokensTOT, tokensPOStot

#Funzione principale che chiama le varie funzioni di analisi statistica e stampa i risultati
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

    #Chiamo la funzione che restituisce i tokens totali dei file di testo e i tokens totali con relativa POS 
    TestoTokenizzato1, TestoAnalizzatoPOS1 = AnnotazioneLinguistica(frasi1)
    TestoTokenizzato2, TestoAnalizzatoPOS2 = AnnotazioneLinguistica(frasi2)

    #Chiamo la funzione che restituisce il numero di frasi e di tokens dei due file di testo
    numFrasi1, numTokens1 = CalcoloStatistichePrimoPunto(frasi1)
    numFrasi2, numTokens2 = CalcoloStatistichePrimoPunto(frasi2)

    #Chiamo la funzione che calcola la lunghezza media delle frasi e dei token dei due file di testo, escludendo i tokens che rappresentano punteggiatura
    lunghezzamediaFrasi1, lunghezzamediaTokens1 = CalcoloStatisticheSecondoPunto(frasi1, TestoTokenizzato1)
    lunghezzamediaFrasi2, lunghezzamediaTokens2 = CalcoloStatisticheSecondoPunto(frasi2, TestoTokenizzato2)

    #Chiamo la funzione che calcola il numero di hapax sui primi 1000 tokens dei due file di testo
    numHapax1 = CalcoloStatisticheTerzoPunto(TestoTokenizzato1)
    numHapax2 = CalcoloStatisticheTerzoPunto(TestoTokenizzato2)

    #Stampo i risultati del primo punto
    print()
    print()
    print("--PRIMO PUNTO--")
    print()
    print()
    print("Confronto il numero di frasi e il numero di tokens dei due file di testo:")
    print()
    print("Il file", file1, "ha", round(numFrasi1),"frasi e", numTokens1, "tokens mentre il file", file2, "ha", round(numFrasi2),"frasi e", numTokens2, "tokens")

    #Stampo i risultati del secondo punto
    print()
    print()
    print("--SECONDO PUNTO--")
    print()
    print()
    print("Confronto la lunghezza media delle frasi in termini di tokens e dei tokens in termini di caratteri nei due file di testo, escludendo i tokens che rappresentano punteggiatura:")
    print()
    print("Il file", file1, "ha una lunghezza media delle frasi pari a", lunghezzamediaFrasi1, "e una lunghezza media dei tokens pari a", lunghezzamediaTokens1, "mentre il file", file2,  "ha una lunghezza media delle frasi pari a", lunghezzamediaFrasi2, "e una lunghezza media dei tokens pari a", lunghezzamediaTokens2)

    #Stampo i risultati del terzo punto
    print()
    print()
    print("--TERZO PUNTO--")
    print()
    print()
    print("Confronto il numero di hapax sui primi 1000 tokens dei due file di testo:")
    print()
    print("Il file", file1, "ha", numHapax1, "hapax",  "mentre il file", file2, "ha", numHapax2, "hapax")

    #Stampo i risultati del quarto punto
    print()
    print()
    print("--QUARTO PUNTO--")
    print()
    print()
    print("Confronto la grandezza del vocabolario e la ricchezza lessicale dei due file di testo al crescere del testo di 500 in 500 tokens:")
    print()
    print("Il file", file1, "ha grandezza del vocabolario e ricchezza lessicale ogni 500 tokens pari a:")
    print()
    #Chiamo la funzione che calcola la grandezza del vocabolario e la TTR al crescere dei due file di testo di 500 tokens in maniera incrementale e stampa i risultati 500 tokens alla volta
    CalcoloStatisticheQuartoPunto(TestoTokenizzato1)
    print("Il file", file2, "ha grandezza del vocabolario e ricchezza lessicale ogni 500 tokens pari a:")
    print()
    CalcoloStatisticheQuartoPunto(TestoTokenizzato2)

    #Chiamo la funzione che calcola la distribuzione percentuale dell'insieme delle parole piene e dell'insieme di quelle funzionali dei due file di testo
    dist_ParolePiene1, dist_ParoleFunzionali1 = CalcoloStatisticheQuintoPunto(TestoTokenizzato1, TestoAnalizzatoPOS1)
    dist_ParolePiene2, dist_ParoleFunzionali2 = CalcoloStatisticheQuintoPunto(TestoTokenizzato2, TestoAnalizzatoPOS2)

    #Stampo i risultati del quinto punto
    print()
    print("--QUINTO PUNTO--")
    print()
    print()
    print("Confronto le distribuzioni percentuali dell'insieme delle parole piene e di quelle funzionali dei due file di testo:")
    print()
    print("Il file", file1, "ha una distribuzione percentuale delle parole piene del", dist_ParolePiene1, "% e di quelle funzionali del", dist_ParoleFunzionali1,"% mentre il file", file2, "ha una distribuzione percentuale delle parole piene del", dist_ParolePiene2, "% e di quelle funzionali del", dist_ParoleFunzionali2,"%")
    print()

main(sys.argv[1], sys.argv[2])
    
