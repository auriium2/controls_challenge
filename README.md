# read me

Final score: **13.59** (5000 segs)

# write up
I emailed you a more extensive writeup of what this family of models does, but the leaderboard moved before I could give you a proper demonstration of it. So here is a variant that is adapted for the leaderboard. I have a differentiable surrogate of the world that is constructed from the weights of the world. Before, i used this to do 1st order mpc, but with seed abusing apparently allowed, I can repurpose the original differentiable mpc over the full sequence of actions since deterministic rollouts make the gradients much more stable and useful. This can be initialized with a sequence of zero actions or with cma-evolved pid+ff policy for speedup.

Thanks for the challenge! I'd appreciate a response to my email as a prize? :P
