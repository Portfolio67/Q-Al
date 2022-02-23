# Q-Al
Artificial intelligence machine learning with the queue algorithm and Ethics.

<!doctype html>
<html>
<head>
	<meta charset="UTF-8">
	<title>Shane Flaten_AI</title>
</head>
<body>
<p><meta charset="utf-8"/></p><div class="TOC">


<ul>
<li><a href="#about-the-project-deep-q-algorithm-with-keras-python">About the Project /Deep Q-algorithm with Keras Python</a>

<ul>
<li><a href="#what-computer-scientist-do">What computer scientist do.</a></li>
<li><a href="#the-ethics-of-artificial-intelligence">The ethics of artificial intelligence</a></li>
<li><a href="#about">About</a></li>
<li><a href="#motivation">Motivation</a>

<ul>
<li><a href="#the-library-specifications-are-as-follows">The Library specifications are as follows:</a></li>
<li><a href="#the-hardware-specifications-are-as-follows">The hardware specifications are as follows:</a></li>
</ul></li>
<li><a href="#setting-up-the-m1-for-ai">Setting up the M1 for Ai.</a></li>
</ul></li>
<li><a href="#install-instructions">Install Instructions</a>

<ul>
<li><a href="#if-you-have-issues">If you have issues</a></li>
<li><a href="#old-file">Old File</a></li>
</ul></li>
<li>[!! Contents within this block are managed by 'conda init' !!][contents-within-this-block-are-managed-by-conda-init]</li>
<li>[&lt;&lt;&lt; conda initialize &lt;&lt;&lt;][<<<-conda-initialize-<<<]

<ul>
<li><a href="#newcode">NEWCODE</a></li>
</ul></li>
<li>[!! Contents within this block are managed by ‘conda init’ !!][contents-within-this-block-are-managed-by-conda-init]</li>
<li>[&lt;&lt;&lt; conda initialize &lt;&lt;&lt;][<<<-conda-initialize-<<<]

<ul>
<li><a href="#terminal-code">TERMINAL CODE</a></li>
</ul></li>
<li><a href="#check-what-version-of-python-you-have">Check what version of Python you have</a></li>
<li><a href="#steps-taken-to-complete-the-project">Steps taken to complete the project</a></li>
<li><a href="#challenges">Challenges.</a></li>
<li><a href="#code-example">Code Example</a></li>
<li><a href="#q-algoritium">Q algoritium:</a></li>
<li><a href="#test">Test</a></li>
<li><a href="#notes-on-deep-learning-ai">Notes on Deep Learning ai</a></li>
<li><a href="#q-algorithm-background">Q - Algorithm background</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#references">REFERENCES</a></li>
</ul>
</div>

<h1 id="about-the-project-deep-q-algorithm-with-keras-python">About the Project /Deep Q-algorithm with Keras Python</h1>

<p>This project was the pirate maze problem that required implementation of the Deep Q - algorithm to solve it. I was given <em>TreasureMaze.py</em> that sets up the environment with an 8 x 8 matrix that is the maze.
<em>GameExperience.py</em> which stores the episodes. I was given the initial pseudo code for the Q Algorithm.</p>

<h2 id="what-computer-scientist-do">What computer scientist do.</h2>

<p>Computer scientist work on the science of data within computers. They don't work on the hardware itself but they work in the theoretical field of how to use computers to solve problems with data. This is important as the world is becoming more reliant on computers and computers have a great deal of power to transform our lives. As a computer scientist it is important to approach a problem one step at a time. Taking in iterative approach can limit mistakes and grow your understanding on the problem and future problems at the solution can create.</p>

<h2 id="the-ethics-of-artificial-intelligence">The ethics of artificial intelligence</h2>

<p>The ethics behind using this algorithm depends on the application. Overall privacy must be maintained. One way is to limit the amount of actual identifying information on a person and mixing nonessential data to further ensure privacy is not violated. Another thing is to use the data stored in the cloud of the user to update the model so the data never has a chance of being leaked from the company. After testing and running simulations it is a good idea to come up with an impact statement of the use of the algorithm.</p>

<h2 id="about">About</h2>

<p>This project uses a python library called Keras to utilize some of the artificial intelligence methods. Gym was imported which has the pathfinding problem. </p>

<p>If you are a data scientist with experience in machine learning or an AI programmer with some exposure to neural networks, you will find this book a useful entry point to deep learning with Keras. Knowledge of Python is required for this book.</p>

<p><em>Deep Learning with Keras</em>
Antonio Gulli &amp; Sujit Pal</p>

<h2 id="motivation">Motivation</h2>

<p>This project exists for learning and testing purposes. It’s a relatively simple project with powerful usages. Being able to do artificial Ai is powerful tool for any application. The Q- algorithm could be very effective for many different things. The main goal was to get the model trained as soon as possible. Please message me if you would like to do pair programming to help me set this up with a Deep Q - Algorithm and (HER) hindsight experience reinforcement. </p>

<p>Required functionality
The Application of artificial intelligence requires a great deal of computing power, for this project I utilize the Mac M1 GPU, that requires Tensorflow. This can be done without Tensorflow but he won’t be using the GPU so the code itself with the import statements and how the layers are called are different than without the use of TensorFlow. </p>

<hr />

<h3 id="the-library-specifications-are-as-follows">The Library specifications are as follows:</h3>

<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• TensorFlow 1.0.0 or higher</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• Keras 2.0.2 or higher</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• Matplotlib 1.5.3 or higher</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• Scikit-learn 0.18.1 or higher</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• NumPy 1.12.1 or higher</li>
</ul>

<h3 id="the-hardware-specifications-are-as-follows">The hardware specifications are as follows:</h3>

<ul>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• Preferably the Mac M1 or I will show you how to set it up based off another GitHub repository</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• Either 32-bit or 64-bit architecture</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• 2+ GHz CPU</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• At least 4 GB RAM</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>• 1 GB of hard disk space available</li>
</ul>

<h2 id="setting-up-the-m1-for-ai">Setting up the M1 for Ai.</h2>

<p>Please refer to the following GitHub for detailed instructions on how to set this up all credit goes to the teacher <mark> <a href="https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup.ipynb">Jeff Heaton</a> </mark> for posting this on his GitHub<sub>Install</sub> I will add Instructions based off his and side-notes to his instructions with links. He provided a <strong><em>tensorflow apple metal_yml</em></strong> file that makes this easy to install rather than downloading each thing you need for a Project. But I installed what I needed one at a time.</p>

<p>A quick overview of what you’re doing is using home brew to install miniforge. From there you use conda to install everything else. This makes a new form of python you use in your IDE.</p>

<p>Links below to mini forge if you want to read more. I recommend checking out the other repositories.</p>

<ul>
<li><a href="https://docs.conda.io/en/latest/miniconda.html">mini conda</a></li>
<li><a href="https://github.com/conda-forge/miniforge">mini forge</a></li>
<li><a href="https://towardsdatascience.com/using-conda-on-an-m1-mac-b2df5608a141">Using conda on the m1</a></li>
</ul>

<hr />

<h1 id="install-instructions">Install Instructions</h1>

<ol>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>Install home-brew and Xcode command line tools</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>https://brew.sh. and follow the instruction in the command line prompts while downloading.</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled></li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>Use Brew to install miniforge, it will get the right version automatically. TERMINAL CODE Brew install miniforge conda. This is where it is installed /opt/homebrew/Caskroom/miniforge/base https://docs.conda.io/en/latest/miniconda.html</li>
<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" disabled>TERMINAL which python “You need to be running the python out of”. <strong><em>user/YOUR_USER_NAME/miniforge3/bin/python</em></strong> . If you don't see that you're running mini forge; close out your terminal restart your Mac in and try again. If that doesn't work I have instructions below.</li>
</ol>

<hr />

<h2 id="if-you-have-issues">If you have issues</h2>

<p>open ~/.zshrc</p>

<p>Make changes in the .zshrc file window that opens</p>

<p>CHANGE THIS FILE TO THE CODE BELOW IT</p>

<h2 id="old-file">Old File</h2>

<p><code># &gt;&gt;&gt; conda initialize &gt;&gt;&gt;
# !! Contents within this block are managed by 'conda init' !!
__conda_setup=&quot;$('/Users/YOUR_USRE_NAME_HERE/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2&gt; /dev/null)&quot;
if [ $? -eq 0 ]; then
    eval &quot;$__conda_setup&quot;
else
    if [ -f &quot;/Users/YOUR_USRE_NAME_HERE/opt/anaconda3/etc/profile.d/conda.sh&quot; ]; then
        . &quot;/Users/YOUR_USRE_NAME_HERE/opt/anaconda3/etc/profile.d/conda.sh&quot;
    else
        export PATH=&quot;/Users/YOUR_USRE_NAME_HERE/opt/anaconda3/bin:$PATH&quot;
    fi
fi
unset __conda_setup
# &lt;&lt;&lt; conda initialize &lt;&lt;&lt;
</code></p>

<h2 id="newcode">NEWCODE</h2>

<p><code># &gt;&gt;&gt; conda initialize &gt;&gt;&gt;
# !! Contents within this block are managed by ‘conda init’ !!
__conda_setup=”$(‘/Users/YOUR_USRE_NAME_HERE/miniforge3/bin/conda’ ‘shell.zsh’ ‘hook’ 2&gt; /dev/null)”
if [ $? -eq 0 ]; then
 eval “$__conda_setup”
else
 if [ -f “/Users/YOUR_USRE_NAME_HERE/miniforge3/etc/profile.d/conda.sh” ]; then
 . “/Users/YOUR_USRE_NAME_HERE/miniforge3/etc/profile.d/conda.sh”
 else
 export PATH=”/Users/YOUR_USRE_NAME_HERE/miniforge3/bin:$PATH”
 fi
fi
unset __conda_setup
# &lt;&lt;&lt; conda initialize &lt;&lt;&lt;
</code>
 Close and reopen the shell</p>

<h2 id="terminal-code">TERMINAL CODE</h2>

<p><code>
conda init zsh
conda activate
which python
conda install -y jupyter
</code></p>

<p>Install the libraries use conda to install what you need. Or you can follow the link provided and take his instructions (easier)
<code>
conda deactivate
jupyter notebook
</code></p>

<h1 id="check-what-version-of-python-you-have">Check what version of Python you have</h1>

<p><code>import sys</code></p>

<p><code>import tensorflow.keras</code>
<code>import pandas as pd</code>
<code>import sklearn as sk</code>
<code>import tensorflow as tf</code></p>

<p><code>print(f&quot;Tensor Flow Version: {tf.**version**}&quot;)</code>
<code>print(f&quot;Keras Version: {tensorflow.keras.**version**}&quot;)</code>
<code>print()</code>
<code>print(f&quot;Python {sys.version}&quot;)</code>
<code>print(f&quot;Pandas {pd.**version**}&quot;)</code>
<code>print(f&quot;Scikit-Learn {sk.**version**}&quot;)</code>
<code>gpu = len(tf.config.list_physical_devices('GPU'))&gt;0</code>
<code>print(&quot;GPU is&quot;, &quot;available&quot; if gpu else &quot;NOT AVAILABLE&quot;)</code></p>

<p><code>Init Plugin</code>
<code>Init Graph Optimizer</code>
<code>Init Kernel</code>
<code>Tensor Flow Version: 2.5.0</code>
<code>Keras Version: 2.5.0</code></p>

<p><code>Python 3.9.7 | packaged by conda-forge | (default, Sep 29 2021, 19:24:02)</code>
<code>[Clang 11.1.0 ]</code>
<code>Pandas 1.3.4</code>
<code>Scikit-Learn 1.0.1</code>
<code>GPU is available</code></p>

<p><img src="1.jpg" alt="1" />
<img src="2.jpg" alt="2" /></p>

<h1 id="steps-taken-to-complete-the-project">Steps taken to complete the project</h1>

<p>This was an iterative project required to first develop algorithm from the very helpful pseudocode. Most algorithms I have found are based off of discrete math which is possible but takes much longer without easy to read pseudocode. I used Jupiter notebook and dataSpell. Dataspell is all around great and can finding errors. </p>

<h1 id="challenges">Challenges.</h1>

<p>PyCharm does not work on the M1. There are ways of getting it running but it has many problems. Is a tradeoff between exploration and exploitation. In my project I used a Decaying epsilon. That means at first you’re using the maximum amount of exploration through each loop a decays see you exploit more what the agent has learned. Setting the agent to learn from random points is better than starting it at the top left because it’ll take a long time with a Q Algorithm.</p>

<p>You can get errors in your terminal with Jupiter notebook, if set up but they're not detailed and in this project that did not work. Use DataSpell</p>

<p><strong><a href="https://towardsdatascience.com/fixing-the-keyerror-acc-and-keyerror-val-acc-errors-in-keras-2-3-x-or-newer-b29b52609af9">ERROR</a> ‘KeyError: ‘acc’ or KeyError: ‘val_acc’</strong> this comes up if you're using an older code you have to replace acc to accuracy</p>

<p><strong><a href="https://towardsdatascience.com/fixing-the-keyerror-acc-and-keyerror-val-acc-errors-in-keras-2-3-x-or-newer-b29b52609af9">Compile a model with Optimizers(Adam, Relu ect..)in m1 Chip </a> ‘KeyError: ‘acc’ or KeyError: ‘val_acc’</strong></p>

<hr />

<h1 id="code-example">Code Example</h1>

<p>`‌</p>

<h1 id="q-algoritium">Q algoritium:</h1>

<pre><code>for epoch in range(n_epoch):  #---- epochs are the elements

    pirate_cell = (0, 0) #set priate cell to 0 col, 0 row. randomly select a free cell, this should be the default

    #Agent_cell = qmaze.reset(pirate_cell)  # or qmaze.reset(pirate_cell) see def play_game(model, qmaze, pirate_cell):

    pirate_cell = qmaze.reset(pirate_cell)   #UnboundLocalError  if line 34 is not there

    #pirate_cell = random.choice(qmaze.free_cells)


    #Reset the maze with agent set to above position
    #See: Review the reset method in the TreasureMaze.py class.
    n_episodes = 0



    #---envstate = Environment.current_state---  
    envstate = qmaze.observe() #environment is visualized with show(qmaze) # This method calls returns the current environment state.
    #See: Review the observe method in the TreasureMaze.py class.



            #----While state is not game over:
    game_status = 'not_over' #keeps updating -step function     see  def game_status(self): in Treasure maze
    while game_status == 'not_over': # function game_status   #GameExperience.py, stores  all the states

        previous_envstate = envstate
        # randomly choose action (left, right, up, down) either by exploration or by exploitation

        if np.random.rand() &lt; epsilon: #------ DECAYING EPSILON &quot;if np.random.rand() &lt; epsilon:&quot;     --citation:(Habith)
            # -----normally start at 1.0 and decay,   either explore or exploit.
            # epsilon is probability of exploring (random action)
            # #-------------I will decay this for faster convergence either explore or exploit.
            action = np.random.choice([LEFT, RIGHT, UP, DOWN]) # this gets the dic for the random actions
        else:
            action = np.argmax(experience.predict(envstate))

        #print(&quot;action&quot;, action) #----for testing purposes
        envstate, reward, game_status = qmaze.act(action) #------ after the action we pass it to the act function in treasure maze, it return three things
        #    SEE: Review the act method in the TreasureMaze.py class.
        episode = [previous_envstate, action, reward, envstate, game_status]
        #        Store episode in Experience replay object
        #    SEE: Review the remember() method in the GameExperience.py class.
        experience.remember(episode)
        #        Train neural network model and evaluate loss
        #Calling GameExperience.get_data() to retrieve training data (input and target) and pass to model.fit method
        #          TO TRAIN THE MODEL.
        inputs, targets = experience.get_data()
        model.fit(inputs, targets)
        n_episodes += 1

    if game_status == 'win':  #SEE  def game_status(self):
        win_history.append(1)
    else:
        win_history.append(0)

    win_rate = sum(win_history[-hsize:]) / hsize
    # You can call model.evaluate to determine loss.
    loss = model.evaluate(inputs, targets)
    epsilon = epsilon - max_epsilon / n_epoch  #-----  the Updating epsilon value   another way is
    #“epsilon_decay = 0.99 #decay factor ”
    #“ epsilon = epsilon * epsilon_decay #decay step” (Habith)




#Print the epoch, loss, episodes, win count, and win rate for each epoch
    dt = datetime.datetime.now() - start_time
    t = format_time(dt.total_seconds())
    template = &quot;Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}&quot;
    print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
    # We simply check if training has exhausted all free cells and if in all
    # cases the agent won.
    if win_rate &gt; 0.9 : epsilon = 0.05
    if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
        print(&quot;Reached 100%% win rate at epoch: %d&quot; % (epoch,))
        break
</code></pre>

<p>`</p>

<h1 id="test">Test</h1>

<p>Without having a Decaying epsilon this took a long time. Using random placement of the pirate agent increased the model training right 3 to 7 times faster. </p>

<hr />

<h1 id="notes-on-deep-learning-ai">Notes on Deep Learning ai</h1>

<h3 id="integer-encoding">Integer encoding</h3>

<p><a href="https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/">Why One-Hot Encode Data in Machine Learning?</a></p>

<p>“red” is 1, “green” is 2, and “blue” is 3.
This is called a label encoding or an integer encoding and is easily reversible.</p>

<h3 id="layers">Layers</h3>

<p><a href="https://www.tutorialspoint.com/keras/keras_dense_layer.htm">KERAS DENSE LAYER</a></p>

<p>Dense layer
, is a deeply connected neural network layer, it connects each note to every other note in the next layer.
<code>output = activation(dot(input, kernel) + bias)</code></p>

<p>Input - the input data.</p>

<p>Kernel - the weight data.</p>

<p>Numpy dot product of all input and there weights.</p>

<p>bias - is to optimize the model</p>

<p>activation- is the activation function.
<a href="https://en.wikipedia.org/wiki/Activation_function">Activation LAYER</a> this is part of the neural network that takes input from one layer and sends and output to another layer</p>

<p><a href="https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/">Rectified Linear Activation</a>
 This means that if the input value (x) is negative, then a value 0.0 is returned, otherwise, the value is returned. This helps with unstable learning. But sometimes you want a negative value in the case of Q learning those values tell the agent not to go there again.</p>

<h3 id="input-shape">Input shape</h3>

<p>The input shape refers to the fact that all data ass be converted into an array of numbers and then fed into the algorithm. They can be one or two dimensional arrays or a matrix.</p>

<p><a href="https://keras.io/api/models/model_training_apis/">Model training APIs</a></p>

<h3 id="losses">Losses</h3>

<p>The loss function computes the quantity that a model should try to minimize during training.</p>

<h3 id="activation">Activation</h3>

<p><a href="https://www.example.com">title</a>
Activation functions are a key part of neural network design.
The modern default activation function for hidden layers is the ReLU function.
The activation function for output layers depends on the type of prediction problem
https://keras.io/api/layers/activations/</p>

<hr />

<h1 id="q-algorithm-background">Q - Algorithm background</h1>

<p>Reinforcement learning uses a Markov decision process aka MDP. The MDP learns from feedback it contains the following: states, models, possible actions, reward function, policy solution. (Geeks) Q learning is also reinforcement learning an is commonly used for the maze problem. In the maze problem there is an Ai agent that has 4 options: up, down, left, right. The Ai will try all combinations in the maze, all the while updating the Q-Values. The Q-value is two values: the state and the action. The Ai is seeking the highest reward, aka the fastest route to the finish. The more movement the more points are deducted, and the large reward is the end. The Q-table: rows have each possible state, and the columns have each action. A good way to visualize it is, the finish would have no negative numbers just a positive reward, the furthest distance from the finish would have the largest negative numbers representing the worst place to be. The Ai would not want to go back to the start after learning that. After learning the Ai has made a policy for that environment. (II, T. B. 2019)
After all the iterations the model is trained and it can be used for inference, basically using it in a real-world environment without labeled examples to learn from.</p>

<p>This is a type of reinforcement learning which uses a Markov decision process. Q learning is commonly used for the maze problem. In the Maze is an Ai that has 4 options: up, down, left, right. The Ai will try all combinations in the maze, all the while updating the Q-Values. The Q value is two values: the state and the action. The Ai is seeking the highest reward, aka the fastest route to the finish. The more movement the more points are deducted, and the reward is the end. The Q-table: rows have each possible state, columns have each action. A good way to visualize it is the finish would have no negative numbers just a positive reward, the furthest distance from the finish would have the largest negative numbers representing the worst place to be. The Ai would not want to go back to the start after learning that. </p>

<p>After learning the Ai has made a policy for that environment in the Q table usually set to zero in the beginning.</p>

<p>Sometimes an epsilon greedy algorithm is used to ensure that AI is choosing more options rather than the best to discover different outcomes.</p>

<p>The temporal difference value is determined by the current state and the state just taken.</p>

<p>Q learning uses the temporal difference equation in the Bellman equation to update the Q-value for the most recent action. </p>

<hr />

<h1 id="contact">Contact</h1>

<p>My name: Shane Flaten </p>

<h1 id="references">REFERENCES</h1>

<p>Thanks to Southern New Hampshire University for providing the reference materials and guides.</p>

<p>https://towardsdatascience.com/fixing-the-keyerror-acc-and-keyerror-val-acc-errors-in-keras-2-3-x-or-newer-b29b52609af9</p>

<p>Relu
https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7</p>

<p>Optimize
https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/</p>

<p>https://medium.com/@blessedmarcel1/how-to-install-jupyter-notebook-on-mac-using-homebrew-528c39fd530f</p>

<p>Instructor: Jeff Heaton, McKelvey School of Engineering, Washington University in St. Louis
https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup.ipynb</p>

<p>Jeff Heaton, Ph.D., Data scientist and adjunct instructor.
https://sites.wustl.edu/jeffheaton/</p>

<p>Q- Algo https://www.youtube.com/watch?v=__t2XRxXGxI</p>

<p>Python example. https://www.youtube.com/watch?v=iKdlKYG78j4</p>
</body>
</html>

