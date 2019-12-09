## Anleitung für das Xray-Label Tool `ExtractXrayCobb.py`

### Voraussetzungen
-	Python mit den Standardpaketen
-	opencv,Installation über Kommandozeile mit pip install opencv-contrib-python
-  pyarrow für das Lesen der Parquet-Datei

Dokumentation des Workflows als Film:
<https://www.dropbox.com/sh/nkapz9lcdzse99w/AAD1aLnj-yKK3l6CxRf2v75ba?dl=0>

### Workflow
Der Normale Workflow sollte folgendermassen zusammengefasst sein:

-	Markieren der Wirbel -> l -> v -> Einzeichnen der Neutralwirbelkippungen -> v -> s (wenn zufrieden und nicht nochmal angeschaut werden muss) -> n
-	Wenn ihr fertig seid, hätte ich gern alle Dateien als dem Verzeichnis xraydata (außer das .parquet – File) wieder zurück.


### Befehle
Taste | Befehl | Erklärung
----- | ------ | ---------
`n`  | next | Nächstes Bild
`ESC`|  |	Abbruch des Programms (speichert automatisch)
`r`|remove |Entfernt den kompletten Bilddatensatz aus der Labeldatei (z.B. wenn das Bild unbrauchbar ist)
`l`|	lines	|	erzeugt die automatische Spline entlang der mittig markierten Wirbelkörper und misst automatisch den COBB Winkel
`v`|	validate	| hier lassen sich die Eckpunkte für die manuelle COBB-Winkel-Messung einzeichnen. Nochmaliges Drücken von v erzeugt die Winkel und komplettiert den Datensatz.
`d`	|discard	|	löscht evtl. vorhandene Linien und du kannst nochmal versuchen. 
`s`	|save&seal|	fixiert den Datensatz, so dass die Datenarbeit an dem Bild für abgeschlossen erklärt wird. Das Bild wird dann beim Neustart des Tools nicht nochmal aufgerufen

