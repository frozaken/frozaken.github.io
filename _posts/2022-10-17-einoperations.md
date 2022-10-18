---
layout: post
title: Please use einsum and einops
date: 2022-10-17 00:00:00-0000
description: A short post about why einsum and einops is a great idea to use in deep learning
tags: pytorch python deep-learning einops
categories: pytorch
---
In this post I will go over one of the most readable ways of performing tensor algebra in many of the modern deep learning libraries; einstein notation.

## A short motivator

If you ever tried to implement an attention mechanism {% cite attention %}, you may have done something like this; Assume $$Q,K\in \mathbb{R}^{n\times d}$$ ie, both are sequences of size $$n$$, with tokens embedded in $$\mathbb{R}^d$$. To calculate the attention distributions, we perform the simple operation

$$
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)
$$

This is easy to implement; most end up with something like this
{% highlight python %}
Q = torch.randn((n,d))
K = torch.randn((n,d))

dist = (Q@K.T/d.sqrt()).softmax(dim=1)
{% endhighlight %}

The question is whether we are happy with this implementation. I would say no, most modern deep learning researchers will apply batches of sequences for parallel processing, instead of one at a time. This means that we in practice assume $$Q,K\in \mathbb{R}^{b\times n\times d}$$ where $$b$$ is the batch size. Suddenly the distributions are not so easy to calculate without sacrificing readability.
{% highlight python %}
Q = torch.randn((b,n,d))
K = torch.randn((b,n,d))

dist = torch.bmm(Q/d.sqrt(), K.permute((0,2,1))).softmax(dim=-1)
{% endhighlight %}

If you simply glance over this code, it is not nearly as obvious that this is a batched matrix multiplication where the second operand is transposed. Atleast not nearly as obvious as the non-batched `Q@K.T`. The einsum operations deals with this problem; allowing for flexible and complicated tensor computations, whilst maintaining a high degree of readability. Using einsum, this becomes much more readable

{% highlight python %}
Q = torch.randn((b,n,d))
K = torch.randn((b,n,d))

dist = torch.einsum('bid,bjd->bij', Q/d.sqrt(), K).softmax(dim=-1)
{% endhighlight %}

## A simple TL;DR of einsum
I understand how if you have never seen einsums before, the third example might seem even less readable than the first two. I hope that by the end of this post, you will begin to see the potential of expressing everything using this simple interface. To understand how einsum works, consider first the simple operation of matrix multiplication between matrices $$A,B$$ where $$AB=C$$, by definition we have

$$
\sum_k A_{ik}B_{kj} = C_{ij}
$$

Notice here that the indices for $$A$$ is `ik`, the indices for $$B$$ is `kj`, and the output indices are `ij`. It turns out that these indices (as long as the follow certain rules), is enough to completely specify the operation. The principle of einsum, is to first specify the indices of the operands, followed by an arrow and the indices of the result. Thus to calculate $$C=AB$$, we would simply list the indices we just derived `torch.einsum('ik,kj->ij',A,B)`. Notice then how easily a transpose can be added. Since a transposition is simply a reversal of indices, then one can calculate $$AB^\top$$ by simply writing `jk` instead of `kj` as indexing $$B$$ with indices `jk` is the same as indexing $$B^\top$$ with indices `kj`. Thus our transposed matrix product is given by
{% highlight python %}
torch.einsum('ik,jk->ij',A,B)
{% endhighlight %}

Notice how suddenly all operations that relate to the order of the dimensions in our tensors became superfluous since we can just specify the indices in a different order. If we go back to the example of attention, we saw how we without batching can perform attention as the soft-max over a simple transposed matrix product `Q@K.T`. Lets use the expression we just derived `torch.einsum('id,jd->ij',Q/d.sqrt(),K)`, and instead of using `k` as an index, we will use `d`, to remind ourselves that we are summing over the embedding dimension. The big revelation is now that turning this into a batch operation is extremely simple. All we have to do, is add a batch index we do not sum over
{% highlight python %}
torch.einsum('bid,bjd->bij',Q/d.sqrt(),K)
{% endhighlight %}


It seems almost magical (for those used to batching more complicated operations), that we can turn these matrix operations into batched operations by simply adding an index to the computation. 

We can also specify an einsum with does not sum over anything. The simplest of these operations is the transpose itself. by specifying a reversal of indices, we get a transpose as we would expect 
{% highlight python %}
torch.einsum('ij->ji', A)
{% endhighlight %}
We can also reduce all dimensions, in which case we get a simple sum over the entire tensor.
{% highlight python %}
torch.einsum('ij->', A)
{% endhighlight %}

## Why you need einops

There are some limitations with this however, we cannot really perform operations that significantly changes the shape, such as taking a dataset with $$N=K\cdot b$$ samples, and splitting it into $$K$$ batches. This is where the python library <a href="https://einops.rocks/">einops</a> comes in handy. It essentially extends the classical einsum, into a much more flexible tool to manipulate shapes.

Consider the problem of having a batch of size $$b$$ containing RBG images of size $$128\times128$$. A common operation {% cite vit %} is to cut this image into patches of size $$h\times w$$. One of the first results that pop up on google when searching this, is <a href="https://discuss.pytorch.org/t/slicing-image-into-square-patches/8772/2">this discussion</a>, which run the following (completely unreadable) code

{% highlight python %}
patches = img_t.data.unfold(0, 3, 3).unfold(1, 8, 8).unfold(2, 8, 8)
{% endhighlight %}

Lets see if we can do better. The thought process needed to write einops is very similar to einsum. The one difference I would point out after using both for quite some time, is that einops thinks more in equations between shapes, rather than indices explicitly as in einsums. Lets try and cut out some $$4\times 8$$ patches from these images. First, we realize that the tensors we are dealing with have shape `(b, 128, 128, 3)`, and we would like them to have shape `(b, 512, 4, 8, 3)`. We can write this as an equation. We denote the number of patches in horizontal direction `nw` and the number in the vertical direction `nh`. We can then write the original shape as `(b, nh*4, nw*8, 3)`, and the resulting shape as `(b, nh*nw, 4, 8, 3)`. This is exactly what we do with the rearrange operation from einops

{% highlight python %}
from einops import rearrange

patches = rearrange(ims, 'b (nh h) (nw w) c -> b (nh nw) h w c', h=4, w = 8)
{% endhighlight %}

This will result in a tensor where `patches[0,0]==ims[0, :4, :8]`, which is exactly what we were looking for. Notice here than instead of the explicit multiplication `nh*4`, einops uses the notation `(n k)` to specify the multiplication of `n` and `k`. Notice how we also parameterize the entire operation by arguments to rearrange. Now we can easily dynamically change the patch size we want to extract, by using the two free variables `h` and `w` in the einops equation. It is also worth noting that einops enforces space between each variables in the equation unlike einsum. This allows for multicharacter variables, but is slightly less compact.

Another common operation that I have seen countless deep learning repositories perform, is changing a batch of images from channel-last to channel-first. Rearrange is also perfect for this, as it gives the very readable equation

{% highlight python %}
rearrange(ims, 'b h w c -> b c h w')
{% endhighlight %}

Notice how we do not even need to know the signature of `ims`, as it is listed very explicitly on the left hand side of the equation `b h w c`. This is the major advantage of einops in my opinion, it is very easy to glance over a set of operations, and have a fairly good idea of how the shapes of tensors change throught the network.

What is interesting for us, is that einops actually lets us define these operations before we even have an operand. For many people trying to work with CNN's for the first time, they include a flattening operation after their last convolution like so

{% highlight python %}
class Model:
    def __init__(self, ...)
        ...
        self.flatten = torch.nn.Flatten()
        ...
{% endhighlight %}

Whilst this definetly does the job, it does not say much about what we are actually flattening. Einops makes this much more readable, by letting us specify the operation upfront

{% highlight python %}
from einops.layers.torch import Rearrange

class Model:
    def __init__(self, ...)
        ...
        self.flatten = Rearrange('b c h w -> b (c h w)')
        ...
{% endhighlight %}

There is certainly overhead in this implementation for a newcomer, as the `rearrange` operation requires much more active thought about how your shapes change. With the flatten operation, you do not really care much about the input nor the output shape. This does lead to more bugs than the `rearrange` operation, since giving it any tensor with a different signature than the left hand side of the equation will throw an error. As deep learning engineers, I think we really should appreciate anything that throws an error - silent errors are very common in DL, and I believe newcomers aswell as veterans should get used to being very aware of their shapes.

If this short teaser sparked your curiosity, then i highly reccomend you check out <a href="https://einops.rocks/">einops</a>. They have many more use cases (and operations) than the ones covered here.

References
----------
{% bibliography --cited %}