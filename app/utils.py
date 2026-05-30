
import joblib


def save_object(obj, path):
    """
    Save Python object as pickle file.
    """
    joblib.dump(obj, path)



def load_object(path):
    """
    Load pickle object.
    """
    return joblib.load(path)
