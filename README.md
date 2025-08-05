# Proximity-Funktion
**Analyse von IOI-Verhältnissen mittels periodizitätsbasierter Proximity-Funktion:**

In diesem Abschnitt werden inter-onset intervals (IOIs) synthetisch generierter rhythmischer Sequenzen mit einer speziell entwickelten, periodizitätsbasierten Proximity-Funktion analysiert. Ziel ist es, das Ausmaß der Periodizität zwischen IOI-Paaren zu quantifizieren und systematisch über unterschiedliche rhythmische Strukturen hinweg zu vergleichen.

**Motivation und Methodik:**
Die zugrunde liegende Annahme ist, dass rhythmische Prototypen durch ganzzahlige oder einfache gebrochene Verhältnisse zwischen IOIs charakterisiert sind – etwa 1:1, 1:2, 2:3 etc. Um diese Verhältnisse als „nahe“ oder „periodisch“ zu bewerten, wurde eine neue Proximity-Funktion definiert, die auf Cosinus-Modulation basiert und frequenzselektiv mit mehreren Harmonikern arbeitet.

*Hauptmerkmale der Funktion:*
Frequenzmultiplikation: Es werden mehrere ganzzahlige Frequenzen bis zu einer einstellbaren Maximalfrequenz berücksichtigt.

*Schärfezunahme mit Frequenz:*
Höhere Frequenzen besitzen schmalere Peaks, gesteuert durch einen exponentiellen Sharpness-Term.

*Normierung:*
Die Funktion ist so skaliert, dass das Maximum immer dem gewünschten Proximity-Wert (z. B. 1.0) entspricht.

*Punktweise Maximumbildung:*
Für jedes Verhältnis wird der Maximalwert aller Frequenzantworten gewählt, um eine klare, nicht additiv überlagerte Bewertung zu erhalten.

Die Funktion erlaubt somit eine differenzierte, frequenzgewichtete und normalisierte Bewertung von IOI-Verhältnissen.
