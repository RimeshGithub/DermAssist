import pickle
enc = pickle.load(open("models/metadata_encoders.pkl", "rb"))
print(enc["age_median"], enc["age_max"])