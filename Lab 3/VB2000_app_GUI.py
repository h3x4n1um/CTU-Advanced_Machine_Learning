# -*- coding: utf-8 -*-
from tkinter import *
from tkinter import filedialog
import tkinter as tk

import tensorflow as tf
import tensorflow.keras as keras
import transformers

class DUBAO(tk.Toplevel):
    def __init__(self, parent):
        self.parent = parent
        Toplevel.__init__(self)

        self.labels = [
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

        # PhoBERT
        self.PhoBERT_name       = "vinai/phobert-base"
        self.PhoBERT_tokenizer  = transformers.AutoTokenizer.from_pretrained(self.PhoBERT_name,
                                                                             use_fast=False)

        # model
        self.model_name         = "best_model"
        self.best_model         = keras.models.load_model("{}/model".format(self.model_name))

        # ui
        self.title("B1812339 B1812282")
        self.geometry('800x600')
        self.lbl = Label(
            self,
            text="CHƯƠNG TRÌNH DEMO",
            font=("Arial Bold", 20)
        )
        self.lbl.grid(column=1, row=1)

        self.lb2 = Label(
            self,
            text="DỰ ĐOÁN VĂN BẢN TIẾNG VIỆT",
            font=("Arial Bold", 15)
        )
        self.lb2.grid(column=1, row=2)

        self.plainlb3 = Label(
            self,
            text="VĂN BẢN CẦN DỰ ĐOÁN",
            font=("Arial", 14)
        )
        self.plainlb3.grid(column=0, row=3)

        self.plaintxt = Text(self, height=7, width=75)
        self.plaintxt.grid(column=1, row=3)

        self.lb6 = Label(
            self,
            text="VĂN BẢN THUỘC LOẠI",
            font=("Arial", 14)
        )
        self.lb6.grid(column=0, row=4)

        self.denctxt = Entry(self, width=100)
        self.denctxt.grid(column=1, row=4)

        self.btn_enc = Button(
            self,
            text="DỰ BÁO",
            command=self.predict
        )
        self.btn_enc.grid(column=1, row=5)

        self.thoat = Button(
            self,
            text="Quay về màn hình chính",
            command=self.destroy
        )
        self.thoat.grid(column=1, row=6)

    def predict(self):
        encoded_input   = self.PhoBERT_tokenizer(self.plaintxt.get("1.0", END),
                                                 padding="max_length",
                                                 truncation=True,
                                                 return_tensors="tf")
        input_ids       = encoded_input["input_ids"]
        attention_mask  = encoded_input["attention_mask"]
        print(input_ids)
        print(attention_mask)
        # load best model n predict
        y_pred          = self.best_model.predict([input_ids, attention_mask])
        print(type(y_pred))
        print(y_pred)
        self.denctxt.delete(0, END)
        self.denctxt.insert(INSERT, self.labels[y_pred.argmax()])

class MainWindow(tk.Frame):
    def __init__(self, parent):
        self.parent = parent
        tk.Frame.__init__(self)

        self.dubao_dulieu = Button(
            text="DỰ BÁO VĂN BẢN TIẾNG VIỆT",
            font=("Times New Roman", 11),
            command=self.dubao
        )
        self.dubao_dulieu.pack()

        self.thoat = Button(
            text="Kết Thúc",
            font=("Times New Roman", 11),
            command=quit
        )
        self.thoat.pack()

    def dubao(self):
        DUBAO(self)
if __name__ == "__main__":
    #main()
    window = tk.Tk()
    window.title("Chương trình chính")
    window.geometry('300x200')
    MainWindow(window)
    window.mainloop()