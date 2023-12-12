# CSI-BERT
data link: https://drive.google.com/file/d/1Y9q9Y7nphcZn8mRc-M_8pm8uM9Bb01wu/view?usp=sharing



**Article:** CSI-BERT



## 1.Dataset

### 1.1 Dataset Description

TODO



### 1.2 Data Preparation

The data structure is as follows:

Amplitude, Phase: batch_size * length * (receiver_num * carrier_dim)
TimeStamp: batch_size * length
Label: batch_size

In our current code version, we do not utilize the phase information as it has been observed to negatively impact the performance of downstream tasks. However, it is worth noting that our model has the capability to successfully recover the lost phase data. If you wish to utilize the phase information, you can concatenate it in the last dimension to the amplitude data.

Furthermore, our dataloader function is specifically designed for our action and people classification task. If you intend to use CSI-BERT for your own task, you will need to create your own dataloader function tailored to your specific requirements.

What’s more, if you wish to use CSI-BERT to recover your own dataset, but your data does not contain timestamp information, you can disable the time embedding (instructions for doing so will be provided in the next section). Additionally, if you intend to use CSI-BERT for downstream tasks but are unsure about the position of the lost CSI, you can also disable the position embedding and solely rely on the time embedding.



## 2.Train

You can refer to our code to see the parameters that can be easily modified through the command line. Here, we will highlight some important parameters.

### 2.1 Pretrain

```bash
python pretrain_adversarial.py --normal --time_embedding
```

If you do not wish to use adversarial learning, you can run `pretrain.py` in the same manner.

As mentioned in the previous section, if you want to disable the time embedding, you can run the model using the following command:

```bash
python pretrain_adversarial.py --normal
```

However, in this case, you may need to make some changes to the data loader. Regardless of the timestamp in your data, it will be ignored by the model.

If you want to disable the position embedding, you can use the following command:

```bash
python pretrain_adversarial.py --normal --position_embedding_type None
```



###2.2 Recover & Finetune

During the recovery phase, you need to maintain the same settings as in the pretraining phase.

```bash
python recover.py --normal --time_embedding --path <full pretrain model path>
```



###2.3 Finetune

During the finetuning phase, you need to maintain the same settings as in the pretraining phase.

```bash
python recover.py --normal --time_embedding --path <bottom pretrain model path> --class_num <class num> --task <task name>
```

As mentioned earlier, if you want to use CSI-BERT for your own tasks, you will need to make some changes to the dataloader.



