# To Do List

## Notizen
- property body.force ist immer ein Nullvektor
- property body.moment ist eine Konstante -> kein Informationsgehalt über Zeit
- Bild kleiner gemacht -> weniger leere Fläche
- 9 Features pro Objekt aufgezeichnet
- Normalization der Koordinaten ok?
- Preprocessed Daten 3 * 4 features pro frame (position und orientierung)
- Input shape der Daten besprechen
- Ist das Trainieren des LSTM am Anfang ohne Binding Matrix usw ohne Active Tuning oder mit? -> ohne AT
# Neu
- Impact force in Interaktion C besprechen (solange der Ball am Actor hängt feuert der hohe Werte raus)

## Bis zum nächten Meeting
- [x] Distancen normalisieren, (wie?, z.B. auf [0, 1]) 
- [x] Positionen normalisieren, z.B. zu einer unteren Ecke
- [x] weitere Properties aufzeichnen, mass, moment, velocity, ...
- [x] collision force, siehe arbiter class
- [x] Costom dataset class 
- [x] Fix: All interactions in custom dataset class
- [x] Force ball wenn er los fliegt (motorcode unabhängig auch ins lstm rein)
- [x] Force auch ins lstm
- [x] Impact force und motor force normalisieren? Wie -> auf -1,1 über alle Daten (nur die ungleich 0 normalisieren, zum abs max normalisieren, check ob 0 0 bleibt)
- [x] event codes auch für batchsize > 1
- [ ] teacher forcing flag
- [ ] renderer for prediction
- [ ] Train in kleinere Funktionen aufteilen
- [ ] Dataset class für aufteilung in train, val und test vorbereiten

## Wie gehts danach weiter
- [ ] LSTM definieren
- [ ] MLP für interaction characteristics vector -> concat with other input to LSTM
- [ ] Training and testing classes for LSTM