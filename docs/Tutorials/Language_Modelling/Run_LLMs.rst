
===========================
Large Language Models on Snellius 
===========================

In this post we demonstrate how to run a large language model on Snellius. The main purpose of this small experiment was explorative in nature; to which extent can we perform
generation or latent extraction on Snellius? How much compute is needed for a single prompt?

We will mainly discuss: 

1. How we got it working on Snellius.
2. how to run it.
3. a few examples.

The main `repository <https://github.com/sara-nl/Galactica_Snellius>`_ and the tested downloaded models can be found on Snellius under ``/projects/0/hpmlprjs/GALACTICA/``.
For now, we named it GALACTICA as it was solely intended for the new Meta's `Galactica <https://galactica.org/>`_ scientific models. Although, we could use any causal language model uploaded to the Huggingface `Huggingface hub <https://huggingface.co/models?sort=downloads&search=language+model>`_. 
More specifically, any model that can be loaded using ``AutoTokenizer`` and ``AutoModelForCausalLM``. Do note 

.. note::
  
  Testing is still necessary as some models break under specific ``PyTorch``, ``transformers`` or ``DeepSpeed`` versions. 

.. warning::
  This blog is mainly intended for the HPML members for now. A more public version is coming soon, GPUs near you.

For now we have tested four different language models:

* `BLOOM <https://huggingface.co/bigscience/bloom>`_, a multi-language language model (40+ languages)
* `Galactica 6.7b <https://huggingface.co/facebook/galactica-6.7b>`_, the galactic models are a family of LMs trained solely on scientific data 
* `OPT-30b <https://huggingface.co/facebook/opt-30b>`_, LM trained on 800GB of text data (180B tokens).
* `GPT-NeoX-20b <https://huggingface.co/EleutherAI/gpt-neox-20b>`_, LM trained by EleutherAI on `The Pile <https://arxiv.org/abs/2101.00027>`_

These models all have one clearly overlapping feature; they are decoder-transformers similar in shape to GPT-2 and GPT-3. It stands to overemphasize that each has their own qualities and 
desired properties and as such, it would be beneficial to keep a few of these models on Snellius as the need arises.

1. How we got it running on Snellius
------------------------------------

We will see how we downloaded and loaded the model for generation.

Let's take `galactica <https://huggingface.co/facebook/galactica-6.7b>`_ uploaded by Meta on Huggingface as an example. The sharded model can be found under ``files and versions``. We first need to have `git lfs <https://git-lfs.github.com/>`_ installed to be able to download these files on our disk.

We can use

  ::

    git lfs clone https://huggingface.co/facebook/galactica-6.7b/

or we can just use ``git clone`` in this case. 

.. note::
  Using git lfs for larger language models such as BLOOM-176b, we would first be downloading specific binaries that would need to be constructed afterwards by running ``git lfs checkout``.

Let's look at how to load this model using Hugginface. We use ``transformers==4.21`` and ``accelerate``, which is Hugginface's own distributed computing framework that will make our lives easier for now.

Loading the Tokenizer and Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


To avoid bloat and confusion we show the important parts only, please take a look at ``./GALACTICA/lm_gen.py`` for more details.

.. code-block:: python

  tokenizer = AutoTokenizer.from_pretrained(args.model_path)

  kwargs = dict(device_map="auto", load_in_8bit=False)

  model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)


Here we see how we prepare the tokenizer and load the model for the given model path. In this case we use ``/GALACTICA/langauge_models/galactica-6.7b``, in which we can find the model weights and the tokenizer. In kwargs we can see ``device_map="auto"`` and ``load_in_8bit=False``. 

With the former we tell the accelerate framework to load the checkpoint automatically. The `accelerate <https://huggingface.co/docs/accelerate/index>`_ framework enables us to run a model in any distributed configuration, it supports sharded models and full checkpoints. The model gets loaded first by initializing a model with ``meta`` (read: empty) weights and then it determines how to load the sharded model across the available GPUs. It employs a simple pipeline parallelism method and while this is not the most efficient method, it's the most flexible for a large variety of models. See this `language modeling guide <https://huggingface.co/docs/accelerate/usage_guides/big_modeling>`_
for a quick glance in how this works. For instance, with ``./GALACTICA/lm_gen.py`` we could load BLOOM 176b model with only one GPU! It might not be the most efficient execution, but hey, it works :).

The latter argument ``load_in_8bit`` makes it possible to load in a model while using less memory. This approach 8-bit quantizes the model with super minimal performance loss. The main idea is to make large language models more accessible with a smaller infrastructure. For instance, this method allows us to load the full BLOOM 176b model on eight A100 40GB GPUs, as opposed to using 16 A100 GPUs. 
However, as nothing is free in life, this comes at the cost of inference time. We can expect forward propagation slow downs of 16-40%. I encourage you to read this `blog post <https://huggingface.co/blog/hf-bitsandbytes-integration>`_ as it's a good read (or, the `paper <https://arxiv.org/abs/2208.07339>`_).


Generation
~~~~~~~~~~

As we tokenize our input and load our model we can easily generate a piece of text given our input by using Huggingface's generate function which is implemented for CausalLMs:

.. code-block:: python

  generate_kwargs = dict(max_new_tokens=args.num_tokens, do_sample=True, temperature=args.temperature)

  outputs = model.generate(**input_tokens, **generate_kwargs)

I trust that most of these arguments are familiar to us. The ``input tokens`` is a dictionary containing the tokenized input text (``input_ids``), an optional ``attention mask`` and ``token_type_ids``. For the record, ``token_type_ids`` is not accepted by galactica-type models. Most of the time we are only interested in the ``input_ids``, but some models require the other tensors as input as well.

DeepSpeed-Inference
~~~~~~~~~~~~~~~~~~~

The script  ``./GALACTICA/lm_gen_ds.py`` contains code to run model inference with deepspeed. The biggest difference with ``./GALACTICA/lm_gen.py`` is the way deepspeed has to be initialized. Luckily, for our purposes for now this can remain minimal:

.. code-block:: python

  model = deepspeed.init_inference(
          model=model,      # Transformers models
          dtype=torch.float16, # dtype of the weights (fp16)
          replace_method=None, # Lets DS autmatically identify the layer to replace
          replace_with_kernel_inject=False, # replace the model with the kernel injector
      )

Deepspeed deploys Tensor parallelism that mainly distributes each layer ''horizontally''; it splits up the layer and distributes it across the GPUs, each shard then lives on its appointed gpu. Additionally, it gives us the capability to replace some modules with specialized CUDA kernels to run these layers faster. I've run this but we are not getting the correct output. This should be fixable though.

We have been having OOM problems running ``lm_gen`` with the ``deepspeed`` launcher. The galactica-6.7b model and any smaller model should work without the deepspeed launcher but we are yet to fix this for models such as gpt-neox-20b or bigger. We consistently see a 2x speedup using Deepspeed. Check out this `tutorial <https://www.philschmid.de/gptj-deepspeed-inference>`_ that helped us setting this up. 

Deepspeed ZeRO is an add-on to the usual DeepSpeed pipeline, it also performs sharding in a tensor parallelism fashion but with, what they call, ''stage 3'' it is able to do some intelligent tensor off-loading. This can come in particularly handy with large models such as BLOOM 176b or OPT-175b. We haven't been able to get this one off the grounds for reasons unknown; it seems to get stuck forever, while generating with regular deepspeed takes a few seconds.

See the following links for more information about ``ZeRO stage-3``:

1. https://www.deepspeed.ai/2021/03/07/zero3-offload.html
2. https://www.deepspeed.ai/tutorials/zero/
3. https://www.deepspeed.ai/2022/09/09/zero-inference.html


2. How to run as a module on Snellius
-------------------------------------

To module load OptimizedLMs add the following line to your bashrc:
  
  ::

    export MODULEPATH="$MODULEPATH:/projects/0/hpmlprjs/scripts
    source ~/.bashrc

Now we can load the module you linked to in your .bashrc.

  ::

    module load OptimizedLMs

And then run with 

  ::

    lm_gen model_choice input output num_tokens temperature 

Anoter way is to load and install your own packages:

The scripts ``./GALACTICA/lm_gen.py`` and ``./GALACTICA/lm_gen_ds.py`` can be run as is with the correct dependencies.
  
  ::

    module load 2021
    module load Python/3.9.5-GCCcore-10.3.0
    module load PyTorch/1.11.0-foss-2021a-CUDA-11.6.0
    module load Miniconda3/4.9.2

    pip install mpi4py, deepspeed, pydantic
    pip install transformers==4.24, accelerate 

And then run:
  
  ::

    python lm_gen.py --model_path ./language_models/galactica-6.7b/ --batch_size 2 --num_tokens 1000 --input_file ./texts/inputs/geny.txt --temperature 0.95 --output_file ./texts/generations/out

Supported Models
~~~~~~~~~~~~~~~~

For now, we have briefly tested the following models with ``accelerate``.

1. galactica-6.7b
2. opt-30b
3. gpt-neox-20b
4. BLOOM

The weights of these models live under ``/projects/0/hpmlprjs/GALACTICA/language_models/``.
.. Attention::

  As of now, deepspeed-inference is only compatible with galactica-6.7b.

3. Examples
-----------

Let's run a few examples. 

::

  lm_gen galactica-6.7b alpha.txt out 75 0.95

Where ``alpha.txt`` contains:

  ::

    "The function of proteins is mainly dictated by its three dimensional structure. Evolution has played its part in"

Output:

The function of proteins is mainly dictated by its three dimensional structure. Evolution has played its part in selecting the best possible protein structure that can perform its functions. This
structure is called native structure and it corresponds to the minimum of potential. There are several methods to compute the structure of a protein starting from amino acid sequence. With the help of evolutionary knowledge, experimental information and many other techniques like computational tools etc. we have made significant progress in prediction of


This took 5.5s to generate excluding model loading (the model fits in memory). We actually generated a batch of 4 examples in 5.5s. With ``lm_gen_ds`` we generate this same batch size in 2.7s! For reference, running opt-30b with ``lm_gen`` takes 8s.

If you feel like it, you  can run ``lm_gen BLOOM input out 50 0.95`` and see how it takes ~40 minutes to run.
