#worklog
##2016-11-15
* Read Torch tutorial

   (https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)

   (https://github.com/soumith/cvpr2015/blob/master/Char-RNN.ipynb)

* Read train.lua

* Learn Lua in 15 minutes
   (http://tylerneylon.com/a/learn-lua/)
```lua
num = 42  -- All numbers are doubles.

t = nil -- Undefines t; Lua has garbage collection.

while num ## 50 do
  num = num + 1  -- No ++ or += type operators.
end

if num ## 40 then
  print('over 40')
elseif s ~= 'walternate' then  -- ~= is not equals.
  -- Equality check is == like Python; ok for strs.
  io.write('not over 40\n')  -- Defaults to stdout.
else

-- How to make a variable local:
  local line = io.read()  -- Reads next stdin line.
```

* Install Cuda on the new Server:

   problem: fail to log into gui. /dev/nvidia*  has nothing.

   solved by not install opengl

##2016-11-16
* Continue read torch tutorial

(https://github.com/soumith/cvpr2015/blob/master/NNGraph%20Tutorial.ipynb)
```lua
function get_rnn(input_size, rnn_size)
  
    -- there are n+1 inputs (hiddens on each layer and x)
    local input = nn.Identity()()
    local prev_h = nn.Identity()()

    -- RNN tick
    local i2h = nn.Linear(input_size, rnn_size)(input)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local added_h = nn.CAddTable()({i2h, h2h}) --performs an element-wise addition
    local next_h = nn.Tanh()(added_h)
    nngraph.annotateNodes()
    return nn.gModule({input, prev_h}, {next_h})
end
```

   (https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf)

   (https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical6.pdf)

   (https://github.com/oxford-cs-ml-2015/practical6)

```lua
--the model can be found at https://en.wikipedia.org/wiki/Long_short-term_memory
function LSTM.lstm(opt)
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end
return LSTM
```

* Prepare data for news generator task

1. description:this task uses news text to produce train data. The source sentences are cleaned sentences from news documents and the target sentences are the next sentence after the source sentence.

2. delete sentences of length <5 or >25

3. 对话系统实验

   path:/home/han/Documents/text-gen/exp1

   result: ppl:46.69

   observation : many instances are translated into "我不知道"

* Install lua hdf5 for Dr.Gong
```
sudo pip install cython
sudo apt-get install libhdf5-dev
sudo pip install h5py
```
##2016-11-17
###Install torch on Telsa server
follow instructions on torch.ch
###Learn torch
(https://github.com/torch/nn/blob/master/doc/table.md)

(https://github.com/torch/torch7/blob/master/doc/tensor.md)

(https://github.com/torch/torch7/blob/master/doc/maths.md)

* 对话系统实验 2:

   path:/home/han/seq2seq-attn/text-gen/exp2

   data

   use entertainment news

   dropped sentences of length >20

   set up

```python
python preprocess.py --srcfile text-gen/data2/src-train.txt --targetfile text-gen/data2/targ-train.txt --srcvalfile text-gen/data2/src-val.txt --targetvalfile text-gen/data2/targ-val.txt --outputfile text-gen/exp2/data/demo --batchsize 256 --seqlength 20
th train.lua -data_file text-gen/exp2/data/demo-train.hdf5 -val_data_file text-gen/exp2/data/demo-val.hdf5 -savefile text-gen/exp2/demo-model -word_vec_size 500 -rnn_size 512 -epochs 20 -start_decay_at 11 -attn 0 -gpuid 1 -gpuid2 1 -print_every 1000 -num_layers 4 -dropout 0.2 ## text-gen/exp2/train.log
#experiment result:
#ppl:39.2
```

##2016-11-18
(https://github.com/torch/torch7/wiki/Cheatsheet)

(http://hunch.net/~nyoml/torch7.pdf)

##2016-11-21
* write lab report

* conduct experiments to tune parameters.

##2016-11-22
(http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

* installed my new desktop however failed 

   after diable nouveau driver and reboot the system, cannot enter into text mode.

##2016-11-23
* Neural Turing Machines:https://arxiv.org/pdf/1410.5401v2.pdf

* 多层lstm前向传播顺序:先纵后横

* understand ResNet

* understand multi-attn

##2016-11-24
###Word error rate
minimum number of editing steps to transform output to reference.

* match: words match, no cost

* substitution: replace one word with another

* insertion: add word

* deletion: drop word

* WER=(substitutions+insertions+deletions)/reference-length

* bleu:n-gram overlap between machine translation output and reference translation

   (http://www.statmt.org/book/slides/08-evaluation.pdf)

##2016-11-28
###Strategies for paraphrasing:
(https://arxiv.org/pdf/cs/0112005v1.pdf)

1. Synonyms:

   Original: 65 is the traditional age for workers to retire in the U.S.

   Paraphrase: 65 is the traditional age for employees to retire in the U.S.

2. Condensation:

   Original: 65 is the traditional age for workers to retire in the U.S.

   Paraphrase: 65 is the traditional retirement age in the U.S.

3. Circumlocution

   Original: 65 is the traditional age for worker to retire in the U.S.

   Paraphrase: 65 is the traditional age for workers to end their professional career in the U.S.

4. Phrase Reversal

   Original: 65 is the traditional age for workers to retire in the U.S.

   Paraphrase: In the U.S., the traditional age for workers to retire is 65.

5. Active-Passive Voice

   Original: The company fired 15 workers.

   Paraphrase: 15 workers were fired by the company.

6. Alternate Word Form

   Original: A manager’s success is often due to perseverance.

   Paraphrase: A manager often succeeds because of perseverance. Managers’ success is often because they persevere.



##2016-12-01
* read train.lua

   when attn=0 use the hidden state of the last rnn unit as the context vector.

##2016-12-02
* read Semantic Parsing via Paraphrasing

##2016-12-12

* read [Tagger: Deep Unsupervised Perceptual Grouping](https://arxiv.org/pdf/1606.06724v2.pdf)

* read [GAN tutorial](http://www.jiqizhixin.com/article/1969)

##2016-12-13
* read [Generating Sentences From a Continuous Space](https://arxiv.org/pdf/1511.06349v2.pdf)

* read [Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf#cite.Graupmann)

* read [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869v3.pdf)

##2016-12-14
###Seven possible relations between phrases/sentences.

(http://web.stanford.edu/class/cs224u/materials/cs224u-2016-bowman.pdf) slide:23

1. equivalence

2. forward entailment

3. reverse entailment 

4. negation

5. alternation

6. cover

7. independence

###Readings

* read [Generating Natural Language Inference Chains](https://arxiv.org/pdf/1606.01404v1.pdf)

* read [Paraphrase-Driven Learning for Open Question Answering](http://knowitall.cs.washington.edu/paralex/acl2013-paralex.pdf)

* read [A Roadmap towards Machine Intelligence](https://arxiv.org/abs/1511.08130)

* read [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473v5.pdf)

##2016-12-16
* read [Data Generation as Sequential Decision Making](https://arxiv.org/pdf/1506.03504v3.pdf)

* read [Generating Chinese Classical Poems with RNN](https://arxiv.org/pdf/1604.01537.pdf)

* read [Tracking The World State With Recurrent Entity Networks](https://arxiv.org/pdf/1612.03969v1.pdf)

##2016-12-19
* read [Learning Distributed Word Representations for Natural Logic Reasoning](http://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10221/10027)

##2016-12-20
* read [A Neural Attention Model for Abstractive Summarization](https://arxiv.org/pdf/1509.00685v2.pdf)

* read [Monolingual Machine Translation for Paraphrase Generation](https://www.microsoft.com/en-us/research/publication/monolingual-machine-translation-for-paraphrase-generation/)

##2016-12-21
* read [DIRT – Discovery of Inference Rules from Text](https://pdfs.semanticscholar.org/511c/439c59f9bbfeb3be135d85ee75bef5594ad2.pdf)

