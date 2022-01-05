# -*- coding: utf-8 -*-

import tensorflow.keras as keras
import transformers

def main() -> None:
    # const
    labels = [
        'cntt',
        'giai tri',
        'giao duc',
        'kinh doanh',
        'nau an',
        'phap luat',
        'suc khoe',
        'the gioi',
        'the thao',
        'tinh yeu'
    ]
    model_name = "best_model"

    # load PhoBERT from transformers
    PhoBERT_name        = "vinai/phobert-base"
    PhoBERT_tokenizer   = transformers.AutoTokenizer.from_pretrained(PhoBERT_name,
                                                                     use_fast=False)
    # input
    #file_name       = input("Nhap ten file chua van ban ban muon phan lop: ")
    file_name       = "test_input.txt"
    input_ids       = None
    attention_mask  = None
    with open(file_name, 'r', encoding="utf8", errors="ignore") as f:
        encoded_input = PhoBERT_tokenizer(f.read(),
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors="tf")
        input_ids       = encoded_input["input_ids"]
        attention_mask  = encoded_input["attention_mask"]
    print(input_ids)
    print(attention_mask)

    # load best model n predict
    best_model  = keras.models.load_model("{}/model".format(model_name))
    y_pred      = best_model.predict([input_ids, attention_mask])

    print(type(y_pred))
    print("Đoạn văn trên thuộc về lĩnh vực", labels[y_pred.argmax()])

if __name__ == "__main__":
    main()