import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def create_base_model(config):
    return AutoModelForSequenceClassification.from_pretrained(config.tokenizer_class)
    
def create_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.base_class) 
    