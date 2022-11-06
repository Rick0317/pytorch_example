# Possible Errors
## The output of each learning iteration is the same
Solution ideas:  
* You don't have optimizer.step() inside your epoch loop.
* Over learning.