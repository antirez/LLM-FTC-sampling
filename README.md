# First Token Cutoff LLM sampling

This code implements the tokenizer described [in this blog post](http://antirez.com/news/142) using MLX.
The implementation is just a modification of the mlx example for LLMs inference.

## Usage

First, install MLX. Note that this is going to work only for Apple Silicon:

    $ pip install mlx

Than convert the Pytorch model into MLX format:

    $ python3 -m mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.2

Finally try the inference:

    $ python3 -m mlx_lm.generate --model mlx_model --prompt "<s>[INST] Who was Leonardo Da Vinci? [/INST]" --max-tokens 500

The default cutoff is 0.7 (tokens up to 70% worse than the best scoring token are accepted for sampling), but you can change this with the `--sampling-cutoff` option in the command line. A cutoff of 0 will make the generation deterministic, always selecting the first token. A cutoff of 1 will consider all the possible tokens and makes no sense. More interesting values are between 0.05 and 0.99, depending on the variability you want.

## Output colorization

If you add the `--colorize` option in the generate command line above, the output of the LLM will be colorized based on the probability of the best token (regardless of the token that is sampled). These are the intervals used:

```
if t0 > 0.95:
    color = 'white'
elif t0 > 0.70:
    color = 'green'
elif t0 > 0.30:
    color = 'yellow'
else:
    color = 'red'
```

First token strength is an interesting hint on the model internal state, especially if the model is outputting dates or other factual information: it is often possible to tell, in such cases, if the model is likely hallucinating or not.

## Sampling algorithm used

This is how the algorithm works:

* Compute softmax() of logits.
* Sort tokens by probability.
* Given T0, the probability of the best token, compute the ratio of all the other tokens as:

    r = 1 - (T[i] / T0)

* Select only tokens for which r <= co
* Perform weighted random pick among the selected tokens.

Note that in this way, regardless of the fact that tokens may have a smooth monotonically decreasing value, there is a hard limit to the tokens we can include in the set of possibilities. Instead with other methods that try to identify high-score clusters, this is not the case.

## Hacking with the implementation

The implementation of the sampler is contained in the `sample` function of `mlx_lm/utils.py`.
Modifying it is straightforward.
