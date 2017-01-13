#worklog
##2016-11-15
#### Read Torch tutorial

   (https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)

   (https://github.com/soumith/cvpr2015/blob/master/Char-RNN.ipynb)

#### Read train.lua

#### Learn Lua in 15 minutes
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

#### Install Cuda on the new Server:

   problem: fail to log into gui. /dev/nvidia####  has nothing.

   solved by not install opengl
---
##2016-11-16
#### Continue read torch tutorial

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

#### Prepare data for news generator task

1.description:this task uses news text to produce train data. The source sentences are cleaned sentences from news documents and the target sentences are the next sentence after the source sentence.

2.delete sentences of length <5 or >25
#### Install lua hdf5 for Dr.Gong
```
sudo pip install cython
sudo apt-get install libhdf5-dev
sudo pip install h5py
```
---
##2016-11-17
####Install torch on Telsa server
follow instructions on torch.ch
####Learn torch
(https://github.com/torch/nn/blob/master/doc/table.md)

(https://github.com/torch/torch7/blob/master/doc/tensor.md)

(https://github.com/torch/torch7/blob/master/doc/maths.md)

---
##2016-11-18
(https://github.com/torch/torch7/wiki/Cheatsheet)

(http://hunch.net/~nyoml/torch7.pdf)
---
##2016-11-21
#### write lab report

#### conduct experiments to tune parameters.
---
##2016-11-22
(http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

#### installed my new desktop however failed 

   after diable nouveau driver and reboot the system, cannot enter into text mode.
---
##2016-11-23
#### Neural Turing Machines:https://arxiv.org/pdf/1410.5401v2.pdf

#### 多层lstm前向传播顺序:先纵后横

#### understand ResNet

#### understand multi-attn
---
##2016-11-24
####Word error rate
minimum number of editing steps to transform output to reference.

1. match: words match, no cost

2. substitution: replace one word with another

3. insertion: add word

4. deletion: drop word

5. WER=(substitutions+insertions+deletions)/reference-length

6. bleu:n-gram overlap between machine translation output and reference translation

   (http://www.statmt.org/book/slides/08-evaluation.pdf)
---
##2016-11-28
####Strategies for paraphrasing:
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
#### read train.lua

   when attn=0 use the hidden state of the last rnn unit as the context vector.

##2016-12-02
#### read Semantic Parsing via Paraphrasing

##2016-12-12

#### read [Tagger: Deep Unsupervised Perceptual Grouping](https://arxiv.org/pdf/1606.06724v2.pdf)

#### read [GAN tutorial](http://www.jiqizhixin.com/article/1969)
---
##2016-12-13
#### read [Generating Sentences From a Continuous Space](https://arxiv.org/pdf/1511.06349v2.pdf)

#### read [Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf#cite.Graupmann)

#### read [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869v3.pdf)
---
##2016-12-14
####Seven possible relations between phrases/sentences.

(http://web.stanford.edu/class/cs224u/materials/cs224u-2016-bowman.pdf) slide:23

1. equivalence

2. forward entailment

3. reverse entailment 

4. negation

5. alternation

6. cover

7. independence

####Readings

#### read [Generating Natural Language Inference Chains](https://arxiv.org/pdf/1606.01404v1.pdf)

#### read [Paraphrase-Driven Learning for Open Question Answering](http://knowitall.cs.washington.edu/paralex/acl2013-paralex.pdf)

#### read [A Roadmap towards Machine Intelligence](https://arxiv.org/abs/1511.08130)

#### read [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473v5.pdf)
---
##2016-12-16
#### read [Data Generation as Sequential Decision Making](https://arxiv.org/pdf/1506.03504v3.pdf)

#### read [Generating Chinese Classical Poems with RNN](https://arxiv.org/pdf/1604.01537.pdf)

#### read [Tracking The World State With Recurrent Entity Networks](https://arxiv.org/pdf/1612.03969v1.pdf)
---
##2016-12-19
#### read [Learning Distributed Word Representations for Natural Logic Reasoning](http://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10221/10027)
---
##2016-12-20
#### read [A Neural Attention Model for Abstractive Summarization](https://arxiv.org/pdf/1509.00685v2.pdf)

#### read [Monolingual Machine Translation for Paraphrase Generation](https://www.microsoft.com/en-us/research/publication/monolingual-machine-translation-for-paraphrase-generation/)
---
##2016-12-22
#### read [DIRT – Discovery of Inference Rules from Text](https://pdfs.semanticscholar.org/511c/439c59f9bbfeb3be135d85ee75bef5594ad2.pdf)

Ideas: words that tend to occur in the same contexts tend  to  have  similar  meanings.

#### read [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615v4.pdf)

Ideas: each filter of the CharCNN is essentially learning to detect particular character n-grams.
---
##2016-12-23
#### read [Dual Learning for Machine Translation](https://arxiv.org/pdf/1611.00179.pdf)

#### read [中文信息处理发展报告](http://cips-upload.bj.bcebos.com/cips2016.pdf)

#### read [Connectionist Temporal Classification: Labelling Unsegmented
Sequence Data with Recurrent Neural Networks](http://www.cs.toronto.edu/%7Egraves/icml_2006.pdf)
---
##2016-12-26
#繁简转换
opencc -i wiki.zh.text -o wiki.zh.text.jian -c zht2zhs.ini
---
##2016-12-27
#### read [Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](https://arxiv.org/pdf/1610.09513v1.pdf)

#### read [Chinese sentence segmentation as comma classification](http://www.aclweb.org/anthology/P11-2111)
The punctuation comma in Chinese sometimes functions as a period causing the diffuculty of segmenting sentences in a paragraph.
---
##2016-12-30
#### read [Data Programming:Creating Large Training Sets, Quickly](https://arxiv.org/pdf/1605.07723v2.pdf)

#### read [Sentence Boundary Detection for Social Media Text](http://amitavadas.com/Pub/SBD_ICON_2015.pdf)
---
##2016-01-02
#### read [text segmentation](https://en.wikipedia.org/wiki/Text_segmentation)

#### read [Elephant:Sequence Labelling for Word and Sentence Segmentation](http://www.aclweb.org/anthology/D13-1146)

IOB tokenization

Label tokens of a sentence.

#### install ruby 2.3.3

```
sudo apt-get update
sudo apt-get install git-core curl zlib1g-dev build-essential libssl-dev libreadline-dev libyaml-dev libsqlite3-dev sqlite3 libxml2-dev libxslt1-dev libcurl4-openssl-dev python-software-properties libffi-dev

cd
git clone https://github.com/rbenv/rbenv.git ~/.rbenv
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
exec $SHELL

git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build
echo 'export PATH="$HOME/.rbenv/plugins/ruby-build/bin:$PATH"' >> ~/.bashrc
exec $SHELL

rbenv install 2.3.3
rbenv global 2.3.3
ruby -v
#optional:
echo "gem: --no-ri --no-rdoc" > ~/.gemrc
```
---
##2017-01-05

#### read [Semantic Parsing: The Task, the State-of-the-Art and the Future](http://www.aclweb.org/anthology/P10-5006)

#### read [Unsupervised Semantic Parsing](http://research.microsoft.com/en-us/um/people/hoifung/papers/poon09.pdf)
---
##2017-01-06

#### read [The Ubuntu Dialogue Corpus: A Large Dataset for Research in
Unstructured Multi-Turn Dialogue Systems](https://arxiv.org/pdf/1506.08909v3.pdf)

Retrieve-based chatbot. Use dual encoders to predict whether a context and a response is a match.
---
##2017-01-09

#### read [Show and Tell](https://arxiv.org/pdf/1609.06647v1.pdf)

#### read [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
---
##2017-01-10

#### find a repo which is able to convert pdf into html (https://github.com/coolwanglu/pdf2htmlEX)

magic installation:(https://gist.github.com/rajeevkannav/d07f822e209a22d07176)

###Install Docker

(https://docs.docker.com/engine/installation/linux/ubuntulinux/)

```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates
sudo apt-key adv \
               --keyserver hkp://ha.pool.sks-keyservers.net:80 \
               --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual
sudo apt-get update
sudo apt-get install docker-engine
sudo service docker start
sudo docker run hello-world
```
---
##2017-01-13
####install tomcat
```
tar -zxvf apache-tomcat-7.0.73.tar.gz
sudo mv apache-tomcat-7.0.73 /opt/
cd /opt/apache-tomcat-7.0.73/bin/

sudo gedit setclasspath.sh
//add the following text to setclasspath.sh in the beginning
//export CATALINA_HOME=/opt/apache-tomcat-7.0.73
//export JAVA_HOME=/usr/local/java/jdk1.7.0_80

sudo ./startup.sh
//Using CATALINA_BASE:   /opt/apache-tomcat-7.0.73
//Using CATALINA_HOME:   /opt/apache-tomcat-7.0.73
//Using CATALINA_TMPDIR: /opt/apache-tomcat-7.0.73/temp
//Using JRE_HOME:        /usr/lib/jdk/jdk1.7.0_80
//Using CLASSPATH:       /opt/apache-tomcat-//7.0.73/bin/bootstrap.jar:/opt/apache-tomcat-7.0.73/bin/tomcat-juli.jar
//Tomcat started.

sudo ./shutdown.sh
//Using CATALINA_BASE:   /opt/apache-tomcat-7.0.73
//Using CATALINA_HOME:   /opt/apache-tomcat-7.0.73
//Using CATALINA_TMPDIR: /opt/apache-tomcat-7.0.73/temp
//Using JRE_HOME:        /usr/lib/jdk/jdk1.7.0_80
//Using CLASSPATH:       /opt/apache-tomcat-//7.0.73/bin/bootstrap.jar:/opt/apache-tomcat-7.0.73/bin/tomcat-juli.jar
```
