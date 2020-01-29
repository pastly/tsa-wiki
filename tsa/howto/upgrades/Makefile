PREDICT_OS=predict-os

predict-stretch.png: data.csv
	$(PREDICT_OS) -o predict-stretch.png

data.csv: FORCE
	$(PREDICT_OS) -r -o predict-stretch.png

FORCE:
