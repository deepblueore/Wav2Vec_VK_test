###
It's important for your device to  have **ffmpeg** and **tensorboard** installed and to have **datasets** library version>=2.11.0
### Alert!
**File predicts.txt contains outdated values!!!**
# Wav2Vec
## A few words about our goals and model.
We are dealing with Wav2Vec2.0. It is a self-supervised learning model, which means it's pre-trained on unlabeled data and we are going to **fine-tune** it on our data which is small enough. The model we are working with consists of **local encoder**, **contextualized encoder** and **quantization module**. A randomly initialized linear projection is added at the output of the contextual encoder and the **CTC-loss** is minimized - that is fine-tuning. Thus, the main advantage of the model is that we can get good results on fairly small data, but on the other hand we have to sample the data just like the ones that were used for training. **For more information, see the attached articles.**
## About dataset
We use RAVDESS dataset to deal with the task. It's convenient enough because this data is gender and emotional balanced. Moreover, there are the same number of recordings of each actor and approximately even distribution of emotions. So So we don't need any additional motivation with the data, except to convert it into a 16kHz. From this uniformity follows **my way of dividing the data**: I divided the dataset into six folds of four actors each, and then sent one of the folds to validation, and the rest to the train. This decision was made **to avoid dataleaks** due to the fact that two recordings of the same emotion from the same actor could fall into both train and validation.
## About Wav2Vec fine-tuning
The Trainer class was used for fine-tuning, since it is the simplest one for these purposes. When choosing, I relied on this [note](https://habr.com/ru/articles/704592/). At the beginning parameters of AdamW were set to [default, learning_rate=3e-4](https://arxiv.org/pdf/1412.6980.pdf). Learning rate was choosen according to the attached article about Wav2Vec2.0 for emotions recognition. It can be seen on the first graph (VK_graph on this repo) that there is convergence, but it is very slow, and possibly (need more iterations to figure out) we are stuck in the neighborhood of the minimum. Since the training turned out to be very slow, I want to make a few assumptions about it:
## About predicting
After the model works, we get predictions that are defined on a set of real numbers. To process the raw file, the **softmax** function is used.
### Conclusions:
1) In trainer class there is parameter max_grad_norm, which is =1 by default. I should have turned it off, but I forgot about it, and when I remembered, it was too late. Obviously, such a gradient clipping makes sense when we are close enough to the solution, or when we are dealing with a heavy tailed problem (this is not it). Thus, cutting the gradient multiplied by a small selected learning-rate slows down the learning process of the model. Unfortunately, I didn't have time to fix it.
2) It can be seen that in the end of training loss fluctuates around a certain value. This may indicate that, on the contrary, too big step was chosen (at the conclusion stage, I found articles where the step about 1e-5 is used)
Thus, as indicated in the attached article, it was worth starting with a sufficiently large step, which will then decrease.
### Final attempt
As it's recommended in the attached article, I started with 10% of warm-up steps and then fix on the constant learning rate 2e-5. By this approach I've got 64% accuracy on validation set. It can be seen  on the final graph (VK_val_FINAL) we have linear convergence, but at the end we rest on the neighborhood of the solution. Thus, after approximately 70% of all the iterations one should increase batch-size or make the learning rate decreasing. I didn't do it because the Trainer doesn't allow to work with decreasing step:(
## About baseline-model
It would be nice to train a non-neural model on our data and look at what results we can get. I chose the **support vector machine** as such a model, because it is one of the most popular machine learning methods after gradient boosting. There should be couple words about training the **SVM**:
### Feature-extracting
After searching the internet I've found the [information](https://daehnhardt.com/blog/2023/03/05/python-audio-signal-processing-with-librosa/), that one could use **Mel-Frequency Cepstral Coefficients** for mood-recognition, so I extracted this vectors with **Librosa lib** and then used it as vector of features for learning.
### Hyperparameters
As hyperparameters of **SVM** I had: 1) kernel type, 2) regulirization coef. and 3) learning rate. To select hyperparameters I used the easiest method - grid-search. For more information, see the attached notebook.
### Results
I've got 32.5% accuracy on this baseline model (random classifier gives 12.5%).
## Conclusion
Wav2Vec2.0 model was trained on a much smaller number of iterations than the SVM, but still received loss whic is twice better. To get better accuracy, we can simply increase the number of iterations - but this way we will get stuck in the neighborhood of the optimum. By increasing the size of the training batch or starting to reduce the step after the 70% of all iterations, we will get significantly better values of the accuracy of the Wav2Vec model.
# What I would like to do
1) **The most essential**. Implement training using PyTorch (because using Trainer I can't make step decreasing after 70% of all the iterations). 
3) Use bayessian approach to the selection of hyperparameters (AdamW).
4) Do kFold approach on my six folds.
