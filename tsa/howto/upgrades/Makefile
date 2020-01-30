PREDICT_OS=predict-os

predict-stretch.png: data.csv
	$(PREDICT_OS) predict graph -o

data.csv: FORCE
	$(PREDICT_OS) refresh

FORCE:
