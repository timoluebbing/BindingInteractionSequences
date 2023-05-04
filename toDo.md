# To Do List

## Notizen
- property body.force ist immer ein Nullvektor
- property body.moment ist eine Konstante -> kein Informationsgehalt über Zeit
- 

## Bis zum nächten Meeting
- [x] Distancen normalisieren, (wie?, z.B. auf [0, 1]) 
- [x] Positionen normalisieren, z.B. zu einer unteren Ecke
- [x] weitere Properties aufzeichnen, mass, moment, velocity, ...
- [x] collision force, siehe arbiter class
- [x] Starte mit costom dataset class 
- [x] Fix: Interaction in custom dataset class

## Wie gehts danach weiter
- [ ] LSTM definieren
- [ ] MLP für interaction characteristics vector -> concat with other input to LSTM
- [ ] Training and testing classes for LSTM