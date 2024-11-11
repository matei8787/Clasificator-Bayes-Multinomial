# Clasificator Bayes Multinomial (Naiv)

## Antrenarea Modelului
Practic, am un training data care e separat pe "label"-uri. Aceste label-uri pot fi orice si oricate. Pentru fiecare label, avem {cuvant : nrap}, ceea ce ne ofera posibilitatea sa calculam probabilitatea ca un cuvant sa apara intr-un anumit label:

$P(w_i|label)$ = $$\frac{(freq(w_i) + 1)} {(totalCuvinte + cuvinteDistincte)}$$

- $w_i$ = cuvantul i 


(OBS: numarul de cuvinte total sau separat se calculeaza pentru fiecare label)

## Testarea Modelului
Acum ca avem o probabilitate pentru un cuvant sa apara intr-un label, avem nevoie de un text sa fie un label. Cum facem asta? Tocmai din nume reiese ca suntem naivi si presupunem ca toate evenimentele sunt independente (orice cuvant chit ca e spam sau ham e de sine statator ceea ce nu are sens in viata reala dar are sens acum).

Tocmai datorita independentei evenimentelor, testarea modelului devine triviala deoarece:

$P(D|label)$ = $\prod_{i=1}^n P(w_i|label)$

unde:
- D = documentul de cuvinte
- label = fiecare label
- $w_i$ = cuvantul i din documentul D


Acum, pentru a decide ce label sa punem pe documentul D, doar calculam probabilitatiile de mai sus si alegem labelul cu probabilitatea maxima.

## Functii Principale:
I. Din clasificatorul Bayes
- split_date(self)
    - Ia datele pe care le are si le imparte in training_data si test_data, pastreaza training_data si returneaza test_data
- init_date(self)
    - ia datele de training pe care le are si le initiaza sa le aiba frumos ({label1: {word1 : count1, word2: count2 ...},label2{word1 : count1...}})
- train(self)
    - Calculeaza $P(w_i|label)$ pentru fiecare $w_i$ din fiecare label
- test(self, data)
    - Calculeaza $P(D|label)$

## FOLOSINTA

Vei avea nevoie de scikit-learn, matplotlib, si pandas

Rulezi main-ul si iti va cere sa introduci numele datasetului preinitializat pe care sa il folosesti. Daca nu alegi bine, iti va afisa cum se foloseste si te vei prinde, am incredere in tine! ;)


## Bibliografie
1. https://www.geeksforgeeks.org/multinomial-naive-bayes/
2. Github Copilot Viata mea
3. https://www.kaggle.com/datasets/?fileType=csv&tags=13302-Classification%2C14104-Text%2C16691-Text+Classification&minUsabilityRating=10.00