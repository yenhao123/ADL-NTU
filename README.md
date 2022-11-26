# ADL-NTU

## hw1
Use Bi-RNN/LSTM to solve intent classification and slot tagging problems.

intent classification
![image](https://user-images.githubusercontent.com/46972327/204099723-b65ffa08-9b37-4fe1-83c6-465fd226427c.png)
>預測句子的類別

slot-tagging
![image](https://user-images.githubusercontent.com/46972327/204099764-06d36063-01da-4095-b797-067aa79321a5.png)
>預測字的類別

## hw2
Use Bert to solve mutiple-choice and question answering problems.

Framework
* Huggingface : to offer datasets and models to train mutiple tasks. (e.g, machine translation、summarization) 

mutiple-choice
>Give choices and predict answer, e.g, ending2 is the answer.

```
"ending0": "looks at a mirror in the mirror as he watches someone walk through a door.",
"ending1": "stops, listening to a cup of coffee with the seated woman, who's standing.",
"ending2": "exits the building and rides the motorcycle into a casino where he performs several tricks as people watch.",
"ending3": "pulls the bag out of his pocket and hands it to someone's grandma.",
"label": 2,
```

question answering
>Give you question and predict the answer

## hw3
Use mt5-small to solve summarization problem

Summarization
![image](https://user-images.githubusercontent.com/46972327/204100196-0081467d-265c-43c7-90c0-70ff48f4011f.png)
>Give you maintext and predict the title



