import pandas as pd
import os

def openCSVFile(filename, foldername):
    path = os.path.join(foldername, filename)
    df = pd.read_csv(path, delimiter = "|", encoding = "utf-8")
    return df