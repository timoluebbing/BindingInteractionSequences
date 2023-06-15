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
- Impact force in Interaktion C besprechen (solange der Ball am Actor hängt feuert der hohe Werte raus)
- Ist der Optimizer.step aufruf an der richtigen Stelle? -> beides ausprobieren

### Neu
- Die Interaction codes haben mit teacher forcing und auch mit dropout derzeit sehr wenig einfluss auf den Loss beim Testen -> Retrospective Inference zur Zeit nicht möglich

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
- [x] renderer for prediction
- [x] Train in kleinere Funktionen aufteilen
- [x] Dataset class für aufteilung in train, val und test vorbereiten
- [x] teacher forcing flag and closed loop training
- [x] torch.util.data.random_split
- [x] evaluate function
- [x] main train in new file
- [x] optimizer step pro batch ausprobieren
- [x] random teacher forcing auf erste x steps umbauen
- [x] Save model with min loss instead of last epoch (wie beim boot projekt)
- [x] embedding layer für 18 input features
- [x] closed loop dropouts progressive 
  - [ ] (auf pretrained model)
- [x] log scale loss plot
- [ ] lr scheduler
- [x] extract eval into tester class
- [x] test loss plotting for each time step
- [x] test loss plotting for each time step by object
- [x] test loss plotting for each time step by type (coords, orientation, force)
### Neu
- [x] Fix batch size impact on test loss...
- [x] Fix sum of losses
- [x] Fix sum of losses type 
  - [x] Für dim=6 ist es gleich aber für dim=4 nicht mehr...
- [x] no forces and no forces out
- [x] Fix renderer for no forces out
- [x] create in out seq auf dim=4 anpassen/ abstrahieren
- [x] timesteps variable
- [ ] train with less timesteps
- [ ] train good model with more dropouts and closed loop training
  - [ ] mit forces und ohne forces (no forces out lass ich erstmal sein)
  - [ ] Interaction label hat auf tf training wenig einfluss...
- [ ] closed loop für no forces out (forces wieder mit rein)
- [ ] interaction module
- [ ] grid search hyperparameter tuning

### Retrospective Inference:
- Wir haben das pretrained model, sind dann alle gradients außer die von den event codes und softmax eingefrohren?
- Statt den event onehot labels wie [0, 1, 0, 0] wird jetzt immer für jede Sequenz [1/4 ...] genommen als input zur event code layer.
- Kein Vergleich der Onehots sondern nur des lstm outputs
  - Also nach ri ist event code z.B. [0.1, 0.2, 0.6, 0.1]. Keinen Vergleich mit loss wie mse zu [0, 0, 1, 0].
  - Nach dem die ganze Sequenz durchgelaufen ist wird der loss zurück auf die event codes layer geführt (plus softmax). Den adaptierten neuen event code 'output' kann man dann wie auslesen?
- Wird das retrospective inference lernen in einem neuen Model gespeichert?
- Wird wieder über den ganzen Datensatz gelooped? 