**LBNets: The first ever Logic-based Transformers/Neural Networks.**

Text-based transformers are great at predicting the next token. They are extremely useful in environments where extremely advanced reasoning is not always top priority. 
So, why do we need LBNets?

**Here's the answer:**

**1. LBNets are the first-ever implementation of logic/reasoning bolted into the model, not just a wrapper like Chain of Thought.**
**2. This will allow for all text-based models to reason far better than the previous pure predict-the-next-token transformers.**

Just to be clear, here are some of the big differences between regular Transformers and LBNets:
1. Transformers work like this:

   [INPUT TOKEN]
         |
   [OUTPUT TOKEN]

2. LBNets work like this:

   [INPUT TOKEN]
         |
   [REASONING TOKEN]
         |
   [REASONING TOKEN]
         |
   [OUTPUT TOKEN]

   For now, there can be a max of 16 reasoning tokens (this is to reduce amount of compute needed)

**To get started:**
1. clone this repo: git clone https://github.com/Aclevo/LBNets --recursive
2. Install LBNets (will still show up as transformers for now): python3 setup.py install

Please know that LBNets is still in extreme beta mode and is nowhere near production-ready.
If you encounter an issue with anything related to LBNets, create an issue. If you would like to request a feature, create a PR.
