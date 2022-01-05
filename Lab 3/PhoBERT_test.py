# -*- coding: utf-8 -*-

import transformers

def main() -> None:
    PhoBERT_model_name = "vinai/phobert-large"

    PhoBERT = transformers.TFAutoModel.from_pretrained(PhoBERT_model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(PhoBERT_model_name,
                                                           use_fast=False)

    batch_sentences = ["Chúng_tôi là những nghiên_cứu_viên ."]
    encoded_inputs = tokenizer(batch_sentences,
                               #padding="max_length",
                               truncation=True,
                               return_tensors="tf")
    print(encoded_inputs["input_ids"])
    print(encoded_inputs["attention_mask"])
    print(type(encoded_inputs))

    for ids in encoded_inputs["input_ids"]:
        print(tokenizer.decode(ids))

    tmp = PhoBERT(encoded_inputs)
    print(tmp)

if __name__ == "__main__":
    main()